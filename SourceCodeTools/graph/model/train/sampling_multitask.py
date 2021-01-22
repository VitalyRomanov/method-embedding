from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import dgl
from math import ceil
from time import time
from os.path import join
import logging

from SourceCodeTools.graph.model.train.utils import BestScoreTracker  # create_elem_embedder
from SourceCodeTools.graph.model.LinkPredictor import LinkPredictor
from SourceCodeTools.embed import token_hasher


def _compute_accuracy(pred_, true_):
    return torch.sum(pred_ == true_).item() / len(true_)


class SimpleNodeEmbedder(nn.Module):
    def __init__(self, dataset, emb_size, dtype=None, n_buckets=500000, pretrained=None):
        super(SimpleNodeEmbedder, self).__init__()

        self.emb_size = emb_size
        self.dtype = dtype
        if dtype is None:
            self.dtype = torch.float32
        self.n_buckets = n_buckets

        self.buckets = None

        leaf_types = {'subword', "Op", "Constant", "Name"}

        nodes_with_embeddings = dataset.nodes[
            dataset.nodes['type_backup'].apply(lambda type_: type_ in leaf_types)
        ][['global_graph_id', 'typed_id', 'type', 'type_backup', 'name']]

        type_name = list(zip(nodes_with_embeddings['type_backup'], nodes_with_embeddings['name']))

        self.node_info = dict(zip(
            list(zip(nodes_with_embeddings['type'], nodes_with_embeddings['typed_id'])),
            type_name
        ))

        assert len(nodes_with_embeddings) == len(self.node_info)

        self.node_info_global = dict(zip(
            nodes_with_embeddings['global_graph_id'],
            type_name
        ))

        if pretrained is None:
            self._create_buckets()
        else:
            self._create_buckets_from_pretrained(pretrained)

    def _create_buckets(self):
        self.buckets = nn.Embedding(self.n_buckets + 1, self.emb_size, padding_idx=self.n_buckets)

    def _create_buckets_from_pretrained(self, pretrained):

        assert pretrained.n_dims == self.emb_size

        import numpy as np

        embs_init = np.random.randn(self.n_buckets, self.emb_size).astype(np.float32)

        for word in pretrained.keys():
            ind = token_hasher(word, self.n_buckets)
            embs_init[ind, :] = pretrained[word]

        from SourceCodeTools.embed.python_op_to_bpe_subwords import python_ops_to_bpe

        def op_embedding(op_tokens):
            embedding = None
            for token in op_tokens:
                token_emb = pretrained.get(token, None)
                if embedding is None:
                    embedding = token_emb
                else:
                    embedding = embedding + token_emb
            return embedding

        for op, op_tokens in python_ops_to_bpe.items():
            op_emb = op_embedding(op_tokens)
            if op_emb is not None:
                op_ind = token_hasher(op, self.n_buckets)
                embs_init[op_ind, :] = op_emb

        weights_with_pad = torch.tensor(np.vstack([embs_init, np.zeros((1, self.emb_size), dtype=np.float32)]))

        self.buckets = nn.Embedding.from_pretrained(weights_with_pad, freeze=False, padding_idx=self.n_buckets)

    def _get_embedding_from_node_info(self, keys, node_info):
        idxs = []

        for key in keys:
            if key in node_info:
                real_type, name = node_info[key]
                idxs.append(token_hasher(name, self.n_buckets))
            else:
                idxs.append(self.n_buckets)

        return self.buckets(torch.LongTensor(idxs))

    def _get_embeddings_with_type(self, node_type, ids):
        type_ids = ((node_type, id_) for id_ in ids)
        return self._get_embedding_from_node_info(type_ids, self.node_info)

    def _get_embeddings_global(self, ids):
        return self._get_embedding_from_node_info(ids, self.node_info_global)

    def get_embeddings(self, node_type=None, node_ids=None):
        assert node_ids is not None
        if node_type is None:
            return self._get_embeddings_global(node_ids)
        else:
            return self._get_embeddings_with_type(node_type, node_ids)

    def forward(self, node_type=None, node_ids=None, train_embeddings=True):
        if train_embeddings:
            return self.get_embeddings(node_type, node_ids.tolist())
        else:
            with torch.set_grad_enabled(False):
                return self.get_embeddings(node_type, node_ids.tolist())

class NodeEmbedder(nn.Module):
    def __init__(self, dataset, emb_size, tokenizer_path, dtype=None, n_buckets=100000, pretrained=None):
        super(NodeEmbedder, self).__init__()

        self.emb_size = emb_size
        self.dtype = dtype
        if dtype is None:
            self.dtype = torch.float32
        self.n_buckets = n_buckets

        self.bpe_tokenizer = None
        self.op_tokenizer = None
        # self.graph_id_to_pretrained_name = None
        self.pretrained_name_to_ind = None
        self.pretrained_embeddings = None
        self.buckets = None

        self.leaf_types = {'subword', "Op", "Constant", "Name"}

        nodes_with_embeddings = dataset.nodes[
            dataset.nodes['type_backup'].apply(lambda type_: type_ in self.leaf_types)
        ][['global_graph_id', 'typed_id', 'type', 'type_backup', 'name']]

        type_name = list(zip(nodes_with_embeddings['type_backup'], nodes_with_embeddings['name']))

        self.node_info = dict(zip(
            list(zip(nodes_with_embeddings['type'], nodes_with_embeddings['typed_id'])),
            type_name
        ))

        assert len(nodes_with_embeddings) == len(self.node_info)

        self.node_info_global = dict(zip(
            nodes_with_embeddings['global_graph_id'],
            type_name
        ))

        # self._create_ops_tokenization(nodes_with_embeddings)
        self._create_buckets()

        if pretrained is not None:
            self._create_pretrained_embeddings(nodes_with_embeddings, pretrained)

        self._create_zero_embedding()
        self._init_tokenizer(tokenizer_path)

    def _create_zero_embedding(self):
        self.zero = torch.zeros((self.emb_size, ), requires_grad=False)

    def _create_pretrained_embeddings(self, nodes, pretrained):
        # self.graph_id_to_pretrained_name = dict(zip(nodes['global_graph_id'], nodes['name']))
        self.pretrained_name_to_ind = pretrained.ind
        embed = nn.Parameter(torch.tensor(pretrained.e, dtype=self.dtype))
        # nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(embed)
        self.pretrained_embeddings = embed

    def _create_ops_tokenization(self, nodes_with_embeddings):
        ops = nodes_with_embeddings.query("type_backup == 'Op'")
        from SourceCodeTools.embed.python_op_to_bpe_subwords import op_tokenizer

        self.ops_tokenized = dict(zip(ops['name'], ops['name'].apply(op_tokenizer)))

    def _create_buckets(self):
        embed = nn.Parameter(torch.Tensor(self.n_buckets, self.emb_size))
        # nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(embed)
        self.buckets = embed

    def _init_tokenizer(self, tokenizer_path):
        from SourceCodeTools.embed.bpe import load_bpe_model, make_tokenizer
        self.bpe_tokenizer = make_tokenizer(load_bpe_model(tokenizer_path))
        from SourceCodeTools.embed.python_op_to_bpe_subwords import op_tokenizer
        self.op_tokenizer = op_tokenizer

    def _tokenize(self, type_, name):
        tokenized = None
        if type_ == "Op":
            try_tokenized = self.op_tokenizer(name)
            if try_tokenized == name:
                tokenized = None

        if tokenized is None:
            tokenized = self.bpe_tokenizer(name)
        return tokenized

    def _get_pretrained_or_none(self, name):
        if self.pretrained_name_to_ind is not None and name in self.pretrained_name_to_ind:
            return self.pretrained_embeddings[self.pretrained_name_to_ind[name], :]
        else:
            return None

    def _get_from_buckets(self, name):
        return self.buckets[token_hasher(name, self.n_buckets), :]

    def _get_from_tokenized(self, type_, name):
        tokens = self._tokenize(type_, name)
        embedding = None
        for token in tokens:
            token_emb = self._get_pretrained_or_none(token)
            if token_emb is None:
                token_emb = self._get_from_buckets(token)

            if embedding is None:
                embedding = token_emb
            else:
                embedding = embedding + token_emb
        return embedding

    def _get_embedding(self, type_id, node_info):
        if type_id in node_info:
            real_type, name = self.node_info[type_id]
            embedding = self._get_pretrained_or_none(name)

            if embedding is None:
                embedding = self._get_from_tokenized(real_type, name)
        else:
            embedding = self.zero

        return embedding

    def _get_embeddings_with_type(self, node_type, ids):
        embeddings = []
        for id_ in ids:
            type_id = (node_type, id_)
            embeddings.append(self._get_embedding(type_id, self.node_info))
        embeddings = torch.stack(embeddings)
        return embeddings

    def _get_embeddings_global(self, ids):
        embeddings = []
        for global_id in ids:
            embeddings.append(self._get_embedding(global_id, self.node_info_global))
        embeddings = torch.stack(embeddings)
        return embeddings

    def get_embeddings(self, node_type=None, node_ids=None):
        assert node_ids is not None
        if node_type is None:
            return self._get_embeddings_global(node_ids)
        else:
            return self._get_embeddings_with_type(node_type, node_ids)

    def forward(self, node_type=None, node_ids=None, train_embeddings=True):
        if train_embeddings:
            return self.get_embeddings(node_type, node_ids.tolist())
        else:
            with torch.set_grad_enabled(False):
                return self.get_embeddings(node_type, node_ids.tolist())


class SamplingMultitaskTrainer:

    def __init__(self,
                 dataset=None, model_name=None, model_params=None,
                 trainer_params=None, restore=None, device=None,
                 pretrained_embeddings_path=None,
                 tokenizer_path=None
                 ):

        self.graph_model = model_name(dataset.g, **model_params).to(device)
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.device = device
        self.epoch = 0
        self.dtype = torch.float32

        # self.ee_node_name = create_elem_embedder(
        #     dataset.load_node_names(), dataset.nodes,
        #     self.elem_emb_size, compact_dst=True
        # ).to(device)
        #
        # self.ee_var_use = create_elem_embedder(
        #     dataset.load_var_use(), dataset.nodes,
        #     self.elem_emb_size, compact_dst=True
        # ).to(device)
        #
        # self.ee_api_call = create_elem_embedder(
        #     dataset.load_api_call(), dataset.nodes,
        #     self.elem_emb_size, compact_dst=False
        # ).to(device)

        from SourceCodeTools.graph.model.ElementEmbedder import ElementEmbedderWithBpeSubwords
        self.ee_node_name = ElementEmbedderWithBpeSubwords(
            elements=dataset.load_node_names(), nodes=dataset.nodes, emb_size=self.elem_emb_size,
            tokenizer_path=tokenizer_path
        ).to(device)

        self.ee_var_use = ElementEmbedderWithBpeSubwords(
            elements=dataset.load_var_use(), nodes=dataset.nodes, emb_size=self.elem_emb_size,
            tokenizer_path=tokenizer_path
        ).to(device)

        from SourceCodeTools.graph.model.ElementEmbedderBase import ElementEmbedderBase
        self.ee_api_call = ElementEmbedderBase(
            elements=dataset.load_api_call(), nodes=dataset.nodes, compact_dst=False
        )



        self.lp_node_name = LinkPredictor(self.ee_node_name.emb_size + self.graph_model.emb_size).to(device)
        self.lp_var_use = LinkPredictor(self.ee_var_use.emb_size + self.graph_model.emb_size).to(device)
        self.lp_api_call = LinkPredictor(self.graph_model.emb_size + self.graph_model.emb_size).to(device)

        if restore:
            self.restore_from_checkpoint(self.model_base_path)

        self.optimizer = self._create_optimizer()

        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.991)
        self.best_score = BestScoreTracker()

        self._create_loaders(*self._get_training_targets())

        if pretrained_embeddings_path is not None:
            self.create_node_embedder(dataset, tokenizer_path, pretrained_path=pretrained_embeddings_path)

    def create_node_embedder(self, dataset, tokenizer_path, n_dims=None, pretrained_path=None):
        from SourceCodeTools.embed.fasttext import load_w2v_map

        if pretrained_path is not None:
            pretrained = load_w2v_map(pretrained_path)
        else:
            pretrained = None

        if pretrained_path is None and n_dims is None:
            raise ValueError(f"Specify embedding dimensionality or provide pretrained embeddings")
        elif pretrained_path is not None and n_dims is not None:
            assert n_dims == pretrained.n_dims, f"Requested embedding size and pretrained embedding " \
                                                f"size should match: {n_dims} != {pretrained.n_dims}"
        elif pretrained_path is not None and n_dims is None:
            n_dims = pretrained.n_dims

        if pretrained is not None:
            logging.info(f"Loading pretrained embeddings...")
        logging.info(f"Input embedding size is {n_dims}")

        self.node_embedder = SimpleNodeEmbedder(
            dataset=dataset,
            emb_size=n_dims,
            # tokenizer_path=tokenizer_path,
            dtype=self.dtype,
            pretrained=pretrained
        )

        # self.node_embedder(node_type="node_", node_ids=torch.LongTensor([0]))
        # self.node_embedder(node_type="node_", node_ids=torch.LongTensor([13749]))
        # self.node_embedder(node_type="node_", node_ids=torch.LongTensor([13754]))

        # node_, 0 matplotlib
        # node_ 13749        Renderer
        # node_  13754 ▁renderer

        # print()

    @property
    def lr(self):
        return self.trainer_params['lr']

    @property
    def batch_size(self):
        return self.trainer_params['batch_size']

    @property
    def sampling_neighbourhood_size(self):
        return self.trainer_params['sampling_neighbourhood_size']

    @property
    def neg_sampling_factor(self):
        return self.trainer_params['neg_sampling_factor']

    @property
    def epochs(self):
        return self.trainer_params['epochs']

    @property
    def elem_emb_size(self):
        return self.trainer_params['elem_emb_size']

    @property
    def node_name_file(self):
        return self.trainer_params['node_name_file']

    @property
    def var_use_file(self):
        return self.trainer_params['var_use_file']

    @property
    def call_seq_file(self):
        return self.trainer_params['call_seq_file']

    @property
    def model_base_path(self):
        return self.trainer_params['model_base_path']

    @property
    def pretraining(self):
        return self.epoch >= self.trainer_params['pretraining_phase']

    # def _extract_embed(self, node_embed, input_nodes):
    #     emb = {}
    #     for node_type, nid in input_nodes.items():
    #         emb[node_type] = node_embed[node_type][nid]
    #     return emb

    def _extract_embed(self, input_nodes):
        emb = {}
        for node_type, nid in input_nodes.items():
            emb[node_type] = self.node_embedder(
                node_type=node_type, node_ids=nid,
                train_embeddings=self.pretraining
            )
        return emb

    def _logits_batch(self, input_nodes, blocks):

        cumm_logits = []

        if self.use_types:
            # emb = self._extract_embed(self.graph_model.node_embed(), input_nodes)
            emb = self._extract_embed(input_nodes)
        else:
            if self.ntypes is not None:
                # single node type
                key = next(iter(self.ntypes))
                input_nodes = {key: input_nodes}
                # emb = self._extract_embed(self.graph_model.node_embed(), input_nodes)
                emb = self._extract_embed(input_nodes)
            else:
                emb = self.node_embedder(node_ids=input_nodes, train_embeddings=self.pretraining)
                # emb = self.graph_model.node_embed()[input_nodes]

        logits = self.graph_model(emb, blocks)

        if self.use_types:
            for ntype in self.graph_model.g.ntypes:

                logits_ = logits.get(ntype, None)
                if logits_ is None:
                    continue

                cumm_logits.append(logits_)
        else:
            if self.ntypes is not None:
                # single node type
                key = next(iter(self.ntypes))
                logits_ = logits[key]
            else:
                logits_ = logits

            cumm_logits.append(logits_)

        return torch.cat(cumm_logits)

    def _logits_embedder(self, node_embeddings, elem_embedder, link_predictor, seeds, negative_factor=1):
        k = negative_factor
        indices = seeds
        batch_size = len(seeds)

        node_embeddings_batch = node_embeddings
        element_embeddings = elem_embedder(elem_embedder[indices.tolist()].to(self.device))

        positive_batch = torch.cat([node_embeddings_batch, element_embeddings], 1)
        labels_pos = torch.ones(batch_size, dtype=torch.long)

        node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
        negative_random = elem_embedder(elem_embedder.sample_negative(batch_size * k).to(self.device))

        negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
        labels_neg = torch.zeros(batch_size * k, dtype=torch.long)

        batch = torch.cat([positive_batch, negative_batch], 0)
        labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)

        logits = link_predictor(batch)

        return logits, labels

    def _handle_non_unique(self, non_unique_ids):
        id_list = non_unique_ids.tolist()
        unique_ids = list(set(id_list))
        new_position = dict(zip(unique_ids, range(len(unique_ids))))
        slice_map = torch.tensor(list(map(lambda x: new_position[x], id_list)), dtype=torch.long)
        return torch.tensor(unique_ids, dtype=torch.long), slice_map

    def _logits_nodes(self, node_embeddings,
                      elem_embedder, link_predictor, create_dataloader,
                      src_seeds, negative_factor=1):
        k = negative_factor
        indices = src_seeds
        batch_size = len(src_seeds)

        node_embeddings_batch = node_embeddings
        next_call_indices = elem_embedder[indices.tolist()]  # this assumes indices is torch tensor

        # dst targets are not unique
        unique_dst, slice_map = self._handle_non_unique(next_call_indices)
        assert unique_dst[slice_map].tolist() == next_call_indices.tolist()

        dataloader = create_dataloader(unique_dst)
        input_nodes, dst_seeds, blocks = next(iter(dataloader))
        blocks = [blk.to(self.device) for blk in blocks]
        assert dst_seeds.shape == unique_dst.shape
        assert dst_seeds.tolist() == unique_dst.tolist()
        unique_dst_embeddings = self._logits_batch(input_nodes, blocks)  # use_types, ntypes)
        next_call_embeddings = unique_dst_embeddings[slice_map.to(self.device)]
        positive_batch = torch.cat([node_embeddings_batch, next_call_embeddings], 1)
        labels_pos = torch.ones(batch_size, dtype=torch.long)

        node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
        negative_indices = torch.tensor(elem_embedder.sample_negative(
            batch_size * k), dtype=torch.long)  # embeddings are sampled from 3/4 unigram distribution
        unique_negative, slice_map = self._handle_non_unique(negative_indices)
        assert unique_negative[slice_map].tolist() == negative_indices.tolist()

        dataloader = create_dataloader(unique_negative)
        input_nodes, dst_seeds, blocks = next(iter(dataloader))
        blocks = [blk.to(self.device) for blk in blocks]
        assert dst_seeds.shape == unique_negative.shape
        assert dst_seeds.tolist() == unique_negative.tolist()
        unique_negative_random = self._logits_batch(input_nodes, blocks)  # use_types, ntypes)
        negative_random = unique_negative_random[slice_map.to(self.device)]
        negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
        labels_neg = torch.zeros(batch_size * k, dtype=torch.long)

        batch = torch.cat([positive_batch, negative_batch], 0)
        labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)

        logits = link_predictor(batch)

        return logits, labels

    def _logits_node_name(self, input_nodes, seeds, blocks):
        src_embs = self._logits_batch(input_nodes, blocks)
        logits, labels = self._logits_embedder(src_embs, self.ee_node_name, self.lp_node_name, seeds)
        return logits, labels

    def _logits_var_use(self, input_nodes, seeds, blocks):
        src_embs = self._logits_batch(input_nodes, blocks)
        logits, labels = self._logits_embedder(src_embs, self.ee_var_use, self.lp_var_use, seeds)
        return logits, labels

    def _logits_api_call(self, input_nodes, seeds, blocks):
        src_embs = self._logits_batch(input_nodes, blocks)
        logits, labels = self._logits_nodes(src_embs, self.ee_api_call, self.lp_api_call, self._create_api_call_loader,
                                            seeds)
        return logits, labels

    def _get_training_targets(self):
        if hasattr(self.graph_model.g, 'ntypes'):
            self.ntypes = self.graph_model.g.ntypes
            # labels = {ntype: self.graph_model.g.nodes[ntype].data['labels'] for ntype in self.ntypes}
            self.use_types = True

            if len(self.graph_model.g.ntypes) == 1:
                # key = next(iter(labels.keys()))
                # labels = labels[key]
                self.use_types = False
        
            train_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['train_mask']).squeeze()
                for ntype in self.ntypes
            }
            val_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['val_mask']).squeeze()
                for ntype in self.ntypes
            }
            test_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['test_mask']).squeeze()
                for ntype in self.ntypes
            }
        else:
            self.ntypes = None
            # labels = g.ndata['labels']
            train_idx = self.graph_model.g.ndata['train_mask']
            val_idx = self.graph_model.g.ndata['val_mask']
            test_idx = self.graph_model.g.ndata['test_mask']
            self.use_types = False

        return train_idx, val_idx, test_idx

    def _evaluate_embedder(self, ee, lp, loader, neg_sampling_factor=1):

        total_loss = 0
        total_acc = 0
        count = 0

        for input_nodes, seeds, blocks in loader:
            blocks = [blk.to(self.device) for blk in blocks]

            src_embs = self._logits_batch(input_nodes, blocks)
            logits, labels = self._logits_embedder(src_embs, ee, lp, seeds, neg_sampling_factor)

            logp = nn.functional.log_softmax(logits, 1)
            loss = nn.functional.cross_entropy(logp, labels)
            acc = _compute_accuracy(logp.argmax(dim=1), labels)

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count

    def _evaluate_nodes(self, ee, lp, create_api_call_loader, loader,
                        neg_sampling_factor=1
                        ):

        total_loss = 0
        total_acc = 0
        count = 0

        for input_nodes, seeds, blocks in loader:
            blocks = [blk.to(self.device) for blk in blocks]

            src_embs = self._logits_batch(input_nodes, blocks)
            logits, labels = self._logits_nodes(src_embs, ee, lp, create_api_call_loader, seeds, neg_sampling_factor)

            logp = nn.functional.log_softmax(logits, 1)
            loss = nn.functional.cross_entropy(logp, labels)
            acc = _compute_accuracy(logp.argmax(dim=1), labels)

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count

    def _evaluate_objectives(
            self, loader_node_name, loader_var_use,
            loader_api_call, neg_sampling_factor
    ):

        node_name, node_name_acc = self._evaluate_embedder(
            self.ee_node_name, self.lp_node_name, loader_node_name, neg_sampling_factor=neg_sampling_factor
        )

        var_use_loss, var_use_acc = self._evaluate_embedder(
            self.ee_var_use, self.lp_var_use, loader_var_use, neg_sampling_factor=neg_sampling_factor
        )

        api_call_loss, api_call_acc = self._evaluate_nodes(self.ee_api_call, self.lp_api_call,
                                                           self._create_api_call_loader, loader_api_call,
                                                           neg_sampling_factor=neg_sampling_factor)

        loss = node_name + var_use_loss + api_call_loss

        return loss, node_name_acc, var_use_acc, api_call_acc

    def _idx_len(self, idx):
        if isinstance(idx, dict):
            length = 0
            for key in idx:
                length += len(idx[key])
        else:
            length = len(idx)
        return length

    def _get_loaders(self, train_idx, val_idx, test_idx, batch_size):
        layers = len(self.graph_model.layers)
        # train sampler
        sampler = dgl.dataloading.MultiLayerNeighborSampler([self.sampling_neighbourhood_size] * layers)
        loader = dgl.dataloading.NodeDataLoader(
            self.graph_model.g, train_idx, sampler, batch_size=batch_size, shuffle=False, num_workers=0)

        # validation sampler
        # we do not use full neighbor to save computation resources
        val_sampler = dgl.dataloading.MultiLayerNeighborSampler([self.sampling_neighbourhood_size] * layers)
        val_loader = dgl.dataloading.NodeDataLoader(
            self.graph_model.g, val_idx, val_sampler, batch_size=batch_size, shuffle=False, num_workers=0)

        # we do not use full neighbor to save computation resources
        test_sampler = dgl.dataloading.MultiLayerNeighborSampler([self.sampling_neighbourhood_size] * layers)
        test_loader = dgl.dataloading.NodeDataLoader(
            self.graph_model.g, test_idx, test_sampler, batch_size=batch_size, shuffle=False, num_workers=0)

        return loader, val_loader, test_loader

    def _create_loaders(self, train_idx, val_idx, test_idx):

        train_idx_node_name, val_idx_node_name, test_idx_node_name = self.ee_node_name.create_idx_pools(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )
        train_idx_var_use, val_idx_var_use, test_idx_var_use = self.ee_var_use.create_idx_pools(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )
        train_idx_api_call, val_idx_api_call, test_idx_api_call = self.ee_api_call.create_idx_pools(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )

        logging.info(
            f"Pool sizes : train {self._idx_len(train_idx_node_name)}, "
            f"val {self._idx_len(val_idx_node_name)}, "
            f"test {self._idx_len(test_idx_node_name)}."
        )
        logging.info(
            f"Pool sizes : train {self._idx_len(train_idx_var_use)}, "
            f"val {self._idx_len(val_idx_var_use)}, "
            f"test {self._idx_len(test_idx_var_use)}."
        )
        logging.info(
            f"Pool sizes : train {self._idx_len(train_idx_api_call)}, "
            f"val {self._idx_len(val_idx_api_call)}, "
            f"test {self._idx_len(test_idx_api_call)}."
        )

        batch_size_node_name = self.batch_size

        def estimate_batch_size(indexes):
            return ceil(self._idx_len(indexes) / ceil(self._idx_len(train_idx_node_name) / batch_size_node_name))

        batch_size_var_use = estimate_batch_size(train_idx_var_use)
        batch_size_api_call = estimate_batch_size(train_idx_api_call)

        self.loader_node_name, self.test_loader_node_name, self.val_loader_node_name = self._get_loaders(
            train_idx=train_idx_node_name, val_idx=val_idx_node_name, test_idx=test_idx_node_name,
            batch_size=batch_size_node_name
        )
        self.loader_var_use, self.test_loader_var_use, self.val_loader_var_use = self._get_loaders(
            train_idx=train_idx_var_use, val_idx=val_idx_var_use, test_idx=test_idx_var_use,
            batch_size=batch_size_var_use
        )
        self.loader_api_call, self.test_loader_api_call, self.val_loader_api_call = self._get_loaders(
            train_idx=train_idx_api_call, val_idx=val_idx_api_call, test_idx=test_idx_api_call,
            batch_size=batch_size_api_call
        )

    def _create_api_call_loader(self, indices):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [self.sampling_neighbourhood_size] * len(self.graph_model.layers))
        return dgl.dataloading.NodeDataLoader(
            self.graph_model.g, indices, sampler, batch_size=len(indices), num_workers=0)

    def _create_optimizer(self):
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(
            [
                {'params': self.graph_model.parameters()},
                {'params': self.ee_node_name.parameters()},
                {'params': self.ee_var_use.parameters()},
                # {'params': self.ee_api_call.parameters()},
                {'params': self.lp_node_name.parameters()},
                {'params': self.lp_var_use.parameters()},
                {'params': self.lp_api_call.parameters()},
            ], lr=self.lr
        )
        return optimizer

    def train_all(self):
        """
        Training procedure for the model with node classifier
        :return:
        """

        for epoch in range(self.epochs):
            self.epoch = epoch

            start = time()

            for i, ((input_nodes_node_name, seeds_node_name, blocks_node_name),
                    (input_nodes_var_use, seeds_var_use, blocks_var_use),
                    (input_nodes_api_call, seeds_api_call, blocks_api_call)) in \
                    enumerate(zip(
                        self.loader_node_name,
                        self.loader_var_use,
                        self.loader_api_call)):

                blocks_node_name = [blk.to(self.device) for blk in blocks_node_name]
                blocks_var_use = [blk.to(self.device) for blk in blocks_var_use]
                blocks_api_call = [blk.to(self.device) for blk in blocks_api_call]

                logits_node_name, labels_node_name = self._logits_node_name(
                    input_nodes_node_name, seeds_node_name, blocks_node_name
                )

                logits_var_use, labels_var_use = self._logits_var_use(
                    input_nodes_var_use, seeds_var_use, blocks_var_use
                )

                logits_api_call, labels_api_call = self._logits_api_call(
                    input_nodes_api_call, seeds_api_call, blocks_api_call
                )

                train_acc_node_name = _compute_accuracy(logits_node_name.argmax(dim=1), labels_node_name)
                train_acc_var_use = _compute_accuracy(logits_var_use.argmax(dim=1), labels_var_use)
                train_acc_api_call = _compute_accuracy(logits_api_call.argmax(dim=1), labels_api_call)

                train_logits = torch.cat([logits_node_name, logits_var_use, logits_api_call], 0)
                train_labels = torch.cat([labels_node_name, labels_var_use, labels_api_call], 0)

                logp = nn.functional.log_softmax(train_logits, 1)
                loss = nn.functional.nll_loss(logp, train_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            _, val_acc_node_name, val_acc_var_use, val_acc_api_call = self._evaluate_objectives(
                self.val_loader_node_name, self.val_loader_var_use, self.val_loader_api_call, self.neg_sampling_factor
            )

            _, test_acc_node_name, test_acc_var_use, test_acc_api_call = self._evaluate_objectives(
                self.test_loader_node_name, self.test_loader_var_use, self.test_loader_api_call,
                self.neg_sampling_factor
            )

            end = time()

            self.best_score.track_best(epoch=epoch, loss=loss.item(), train_acc_node_name=train_acc_node_name,
                                       val_acc_node_name=val_acc_node_name, test_acc_node_name=test_acc_node_name,
                                       train_acc_var_use=train_acc_var_use, val_acc_var_use=val_acc_var_use,
                                       test_acc_var_use=test_acc_var_use, train_acc_api_call=train_acc_api_call,
                                       val_acc_api_call=val_acc_api_call, test_acc_api_call=test_acc_api_call,
                                       time=end - start)

            self.save_checkpoint(self.model_base_path)

            self.lr_scheduler.step()

    def save_checkpoint(self, checkpoint_path=None, checkpoint_name=None, **kwargs):

        checkpoint_path = join(checkpoint_path, "saved_state.pt")

        param_dict = {
            'graph_model': self.graph_model.state_dict(),
            'ee_node_name': self.ee_node_name.state_dict(),
            'ee_var_use': self.ee_var_use.state_dict(),
            # 'ee_api_call': self.ee_api_call.state_dict(),
            "lp_node_name": self.lp_node_name.state_dict(),
            "lp_var_use": self.lp_var_use.state_dict(),
            "lp_api_call": self.lp_api_call.state_dict(),
            "epoch": self.epoch
        }

        if len(kwargs) > 0:
            param_dict.update(kwargs)

        torch.save(param_dict, checkpoint_path)

    def restore_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(join(checkpoint_path, "saved_state.pt"))
        self.graph_model.load_state_dict(checkpoint['graph_model'])
        self.ee_node_name.load_state_dict(checkpoint['ee_node_name'])
        self.ee_var_use.load_state_dict(checkpoint['ee_var_use'])
        # self.ee_api_call.load_state_dict(checkpoint['ee_api_call'])
        self.lp_node_name.load_state_dict(checkpoint['lp_node_name'])
        self.lp_var_use.load_state_dict(checkpoint['lp_var_use'])
        self.lp_api_call.load_state_dict(checkpoint['lp_api_call'])
        logging.info(f"Restored from epoch {checkpoint['epoch']}")

    def final_evaluation(self):

        loss, train_acc_node_name, train_acc_var_use, train_acc_api_call = self._evaluate_objectives(
            self.loader_node_name, self.loader_var_use, self.loader_api_call, 1
        )

        _, val_acc_node_name, val_acc_var_use, val_acc_api_call = self._evaluate_objectives(
            self.val_loader_node_name, self.val_loader_var_use, self.val_loader_api_call, 1
        )

        _, test_acc_node_name, test_acc_var_use, test_acc_api_call = self._evaluate_objectives(
            self.test_loader_node_name, self.test_loader_var_use, self.test_loader_api_call, 1
        )

        scores = {
            # "loss": loss.item(),
            "train_acc_node_name": train_acc_node_name,
            "val_acc_node_name": val_acc_node_name,
            "test_acc_node_name": test_acc_node_name,
            "train_acc_var_use": train_acc_var_use,
            "val_acc_var_use": val_acc_var_use,
            "test_acc_var_use": test_acc_var_use,
            "train_acc_api_call": train_acc_api_call,
            "val_acc_api_call": val_acc_api_call,
            "test_acc_api_call": test_acc_api_call,
        }

        print(
            f'Final Eval : node name Train Acc {scores["train_acc_node_name"]:.4f}, '
            f'node name Val Acc {scores["val_acc_node_name"]:.4f}, '
            f'node name Test Acc {scores["test_acc_node_name"]:.4f}, '
            f'var use Train Acc {scores["train_acc_var_use"]:.4f}, '
            f'var use Val Acc {scores["val_acc_var_use"]:.4f}, '
            f'var use Test Acc {scores["test_acc_var_use"]:.4f}, '
            f'api call Train Acc {scores["train_acc_api_call"]:.4f}, '
            f'api call Val Acc {scores["val_acc_api_call"]:.4f}, '
            f'api call Test Acc {scores["test_acc_api_call"]:.4f}'
        )

        return scores

    def eval(self):
        self.graph_model.eval()
        self.ee_node_name.eval()
        self.ee_var_use.eval()
        # self.ee_api_call.eval()
        self.lp_node_name.eval()
        self.lp_var_use.eval()
        self.lp_api_call.eval()

    def to(self, device):
        # self.graph_model.to(device)
        self.ee_node_name.to(device)
        self.ee_var_use.to(device)
        # self.ee_api_call.to(device)
        self.lp_node_name.to(device)
        self.lp_var_use.to(device)
        self.lp_api_call.to(device)


def select_device(args):
    device = 'cpu'
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = 'cuda:%d' % args.gpu
    return device


def training_procedure(
        dataset, model_name, model_params, args, model_base_path
) -> Tuple[SamplingMultitaskTrainer, dict]:

    device = select_device(args)

    model_params['num_classes'] = args.node_emb_size

    trainer_params = {
        'lr': model_params.pop('lr'),
        'batch_size': args.batch_size,
        'sampling_neighbourhood_size': args.num_per_neigh,
        'neg_sampling_factor': args.neg_sampling_factor,
        'epochs': args.epochs,
        # 'node_name_file': args.fname_file,
        # 'var_use_file': args.varuse_file,
        # 'call_seq_file': args.call_seq_file,
        'elem_emb_size': args.elem_emb_size,
        'model_base_path': model_base_path,
        'pretraining_phase': args.pretraining_phase
    }

    trainer = SamplingMultitaskTrainer(
        dataset=dataset,
        model_name=model_name,
        model_params=model_params,
        trainer_params=trainer_params,
        restore=args.restore_state,
        device=device,
        pretrained_embeddings_path=args.pretrained,
        tokenizer_path=args.tokenizer
    )

    try:
        trainer.train_all()
    except KeyboardInterrupt:
        print("Training interrupted")
    except Exception as e:
        raise e

    trainer.eval()
    scores = trainer.final_evaluation()

    trainer.to('cpu')

    return trainer, scores
