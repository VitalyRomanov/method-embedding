import logging
from collections import OrderedDict
from itertools import chain

import dgl
import torch

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedderWithBpeSubwords
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.LinkPredictor import LinkPredictor

import torch.nn as nn

def _compute_accuracy(pred_, true_):
    return torch.sum(pred_ == true_).item() / len(true_)

class Objective(nn.Module):
    def __init__(
            self, name, objective_type, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="nn", masker: SubwordMasker=None
    ):
        """
        :param name: name for reference
        :param objective_type: one of: graph_link_prediction|graph_link_classification|subword_ranker|node_classification
        :param graph_model:
        :param nodes:
        :param data_loading_func:
        :param device:
        :param sampling_neighbourhood_size:
        :param tokenizer_path:
        :param target_emb_size:
        :param link_predictor_type:
        """
        super(Objective, self).__init__()
        self.name = name
        self.type = objective_type
        self.graph_model = graph_model
        self.sampling_neighbourhood_size = sampling_neighbourhood_size
        self.batch_size = batch_size
        self.node_embedder = node_embedder
        self.device = device
        self.masker = masker

        if self.type not in {"graph_link_prediction", "graph_link_classification", "subword_ranker", "classification"}:
            raise NotImplementedError()

        # create target embedder
        if self.type == "graph_link_prediction" or self.type == "graph_link_classification":
            self.target_embedder = ElementEmbedderBase(
                elements=data_loading_func(), nodes=nodes, compact_dst=False, dst_to_global=True
            )
        elif self.type == "subword_ranker":
            self.target_embedder = ElementEmbedderWithBpeSubwords(
                elements=data_loading_func(), nodes=nodes, emb_size=target_emb_size,
                tokenizer_path=tokenizer_path
            ).to(self.device)
        elif self.type == "node_classification":
            self.target_embedder = None

        # create link predictors
        if self.type in {"graph_link_prediction", "graph_link_classification", "subword_ranker"}:
            if link_predictor_type == "nn":
                self.link_predictor = LinkPredictor(target_emb_size + self.graph_model.emb_size).to(self.device)
            elif link_predictor_type == "inner_prod":
                pass
            else:
                raise NotImplementedError()
        else:
            # for node classifier
            self.link_predictor = LinkPredictor(self.graph_model.emb_size).to(self.device)

        # create loaders
        if self.type in {"graph_link_prediction", "graph_link_classification", "subword_ranker"}:
            train_idx, val_idx, test_idx = self._get_training_targets()
            train_idx, val_idx, test_idx = self.target_embedder.create_idx_pools(
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
            )
        else:
            raise NotImplementedError()
        logging.info(
            f"Pool sizes for {self.name}: train {self._idx_len(train_idx)}, "
            f"val {self._idx_len(val_idx)}, "
            f"test {self._idx_len(test_idx)}."
        )
        self.train_loader, self.test_loader, self.val_loader = self._get_loaders(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            batch_size=self.batch_size  # batch_size_node_name
        )

    def _idx_len(self, idx):
        if isinstance(idx, dict):
            length = 0
            for key in idx:
                length += len(idx[key])
        else:
            length = len(idx)
        return length

    def _handle_non_unique(self, non_unique_ids):
        id_list = non_unique_ids.tolist()
        unique_ids = list(set(id_list))
        new_position = dict(zip(unique_ids, range(len(unique_ids))))
        slice_map = torch.tensor(list(map(lambda x: new_position[x], id_list)), dtype=torch.long)
        return torch.tensor(unique_ids, dtype=torch.long), slice_map

    def _get_training_targets(self):
        """
        Set use_type flag based on the number of types in the graph.
        :return: Return train, validation and test indexes as typed
        ids. For graphs with single node type typed and global graph ids match.
        """
        if hasattr(self.graph_model.g, 'ntypes'):
            self.ntypes = self.graph_model.g.ntypes
            # labels = {ntype: self.graph_model.g.nodes[ntype].data['labels'] for ntype in self.ntypes}
            self.use_types = True

            if len(self.graph_model.g.ntypes) == 1:
                # key = next(iter(labels.keys()))
                # labels = labels[key]
                self.use_types = False

            train_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['train_mask'], as_tuple=False).squeeze()
                for ntype in self.ntypes
            }
            val_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['val_mask'], as_tuple=False).squeeze()
                for ntype in self.ntypes
            }
            test_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['test_mask'], as_tuple=False).squeeze()
                for ntype in self.ntypes
            }
        else:
            # not sure when this is called
            raise NotImplementedError()
            # self.ntypes = None
            # # labels = g.ndata['labels']
            # train_idx = self.graph_model.g.ndata['train_mask']
            # val_idx = self.graph_model.g.ndata['val_mask']
            # test_idx = self.graph_model.g.ndata['test_mask']
            # self.use_types = False

        return train_idx, val_idx, test_idx

    def _get_loaders(self, train_idx, val_idx, test_idx, batch_size):
        # train sampler
        layers = self.graph_model.num_layers
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

    def reset_iterator(self, data_split):
        iter_name = f"{data_split}_loader_iter"
        setattr(self, iter_name, iter(getattr(self, f"{data_split}_loader")))

    def loader_next(self, data_split):
        iter_name = f"{data_split}_loader_iter"
        if not hasattr(self, iter_name):
            setattr(self, iter_name, iter(getattr(self, f"{data_split}_loader")))
        return next(getattr(self, iter_name))

    def _create_loader(self, indices):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [self.sampling_neighbourhood_size] * self.graph_model.num_layers)
        return dgl.dataloading.NodeDataLoader(
            self.graph_model.g, indices, sampler, batch_size=len(indices), num_workers=0)

    def _extract_embed(self, input_nodes, train_embeddings=True, masked=None):
        emb = {}
        for node_type, nid in input_nodes.items():
            emb[node_type] = self.node_embedder(
                node_type=node_type, node_ids=nid,
                train_embeddings=train_embeddings, masked=masked
            ).to(self.device)
        return emb

    def _logits_batch(self, input_nodes, blocks, train_embeddings=True, masked=None):

        cumm_logits = []

        if self.use_types:
            # emb = self._extract_embed(self.graph_model.node_embed(), input_nodes)
            emb = self._extract_embed(input_nodes, train_embeddings, masked=masked)
        else:
            if self.ntypes is not None:
                # single node type
                key = next(iter(self.ntypes))
                input_nodes = {key: input_nodes}
                # emb = self._extract_embed(self.graph_model.node_embed(), input_nodes)
                emb = self._extract_embed(input_nodes, train_embeddings, masked=masked)
            else:
                emb = self.node_embedder(node_ids=input_nodes, train_embeddings=train_embeddings, masked=masked)
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

    def seeds_to_global(self, seeds):
        if type(seeds) is dict:
            indices = [self.graph_model.g.nodes[ntype].data["global_graph_id"][seeds[ntype]] for ntype in seeds]
            return torch.cat(indices, dim=0)
        else:
            return seeds

    def _logits_embedder(self, node_embeddings, elem_embedder, link_predictor, seeds, negative_factor=1):
        k = negative_factor
        indices = self.seeds_to_global(seeds)
        batch_size = len(indices)

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

    def _logits_nodes(self, node_embeddings,
                      elem_embedder, link_predictor, create_dataloader,
                      src_seeds, negative_factor=1, train_embeddings=True):
        k = negative_factor
        indices = self.seeds_to_global(src_seeds)
        batch_size = len(indices)

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
        unique_dst_embeddings = self._logits_batch(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
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
        unique_negative_random = self._logits_batch(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
        negative_random = unique_negative_random[slice_map.to(self.device)]
        negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
        labels_neg = torch.zeros(batch_size * k, dtype=torch.long)

        batch = torch.cat([positive_batch, negative_batch], 0)
        labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)

        logits = link_predictor(batch)

        return logits, labels

    def seeds_to_python(self, seeds):
        if isinstance(seeds, dict):
            python_seeds = {}
            for key, val in seeds.items():
                python_seeds[key] = val.tolist()
        else:
            python_seeds = seeds.tolist()
        return python_seeds

    def forward(self, input_nodes, seeds, blocks, train_embeddings=True):
        masked = None
        if self.type in {"subword_ranker"}:
            masked = self.masker.get_mask(self.seeds_to_python(seeds))
        graph_emb = self._logits_batch(input_nodes, blocks, train_embeddings, masked=masked)
        if self.type in {"subword_ranker"}:
            logits, labels = self._logits_embedder(graph_emb, self.target_embedder, self.link_predictor, seeds)
        elif self.type in {"graph_link_prediction", "graph_link_classification"}:
            logits, labels = self._logits_nodes(graph_emb, self.target_embedder, self.link_predictor,
                                                self._create_loader, seeds, train_embeddings=train_embeddings)
        else:
            raise NotImplementedError()

        acc = _compute_accuracy(logits.argmax(dim=1), labels)
        logp = nn.functional.log_softmax(logits, 1)
        loss = nn.functional.nll_loss(logp, labels)

        return loss, acc

    def _evaluate_embedder(self, ee, lp, data_split, neg_sampling_factor=1):

        total_loss = 0
        total_acc = 0
        ndcg_at = [1, 3, 5, 10]
        total_ndcg = {f"ndcg@{k}": 0. for k in ndcg_at}
        count = 0

        for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
            blocks = [blk.to(self.device) for blk in blocks]

            masked = None
            if self.type == "subword_ranker":
                masked = self.masker.get_mask(self.seeds_to_python(seeds))

            src_embs = self._logits_batch(input_nodes, blocks, masked=masked)
            logits, labels = self._logits_embedder(src_embs, ee, lp, seeds, neg_sampling_factor)

            ndcg = self.target_embedder.score_candidates(self.seeds_to_global(seeds), src_embs, self.link_predictor, at=ndcg_at)
            for key, val in ndcg.items():
                total_ndcg[key] = total_ndcg[key] + val

            logp = nn.functional.log_softmax(logits, 1)
            loss = nn.functional.cross_entropy(logp, labels)
            acc = _compute_accuracy(logp.argmax(dim=1), labels)

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count, {key: val / count for key, val in total_ndcg.items()}

    def _evaluate_nodes(self, ee, lp, create_api_call_loader, data_split, neg_sampling_factor=1):

        total_loss = 0
        total_acc = 0
        count = 0

        for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
            blocks = [blk.to(self.device) for blk in blocks]

            masked = None
            if self.type == "subword_ranker":
                masked = self.masker.get_mask(self.seeds_to_python(seeds))

            src_embs = self._logits_batch(input_nodes, blocks, masked=masked)
            logits, labels = self._logits_nodes(src_embs, ee, lp, create_api_call_loader, seeds, neg_sampling_factor)

            logp = nn.functional.log_softmax(logits, 1)
            loss = nn.functional.cross_entropy(logp, labels)
            acc = _compute_accuracy(logp.argmax(dim=1), labels)

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count

    def evaluate(self, data_split, neg_sampling_factor=1):
        if self.type in {"subword_ranker"}:
            loss, acc, ndcg = self._evaluate_embedder(
                self.target_embedder, self.link_predictor, data_split=data_split, neg_sampling_factor=neg_sampling_factor
            )
        elif self.type in {"graph_link_prediction", "graph_link_classification"}:
            loss, acc = self._evaluate_nodes(
                self.target_embedder, self.link_predictor, self._create_loader, data_split=data_split,
                neg_sampling_factor=neg_sampling_factor
            )
            ndcg = None
        else:
            raise NotImplementedError()

        return loss, acc, ndcg

    def parameters(self, recurse: bool = True):
        if self.type in {"subword_ranker"}:
            return chain(self.target_embedder.parameters(), self.link_predictor.parameters())
        elif self.type in {"graph_link_prediction", "graph_link_classification"}:
            return self.link_predictor.parameters()
        else:
            raise NotImplementedError()

    def custom_state_dict(self):
        state_dict = OrderedDict()
        if self.type in {"subword_ranker"}:
            for k, v in self.target_embedder.state_dict().items():
                state_dict[f"target_embedder.{k}"] = v
            for k, v in self.link_predictor.state_dict().items():
                state_dict[f"link_predictor.{k}"] = v
            # state_dict["target_embedder"] = self.target_embedder.state_dict()
            # state_dict["link_predictor"] = self.link_predictor.state_dict()
        elif self.type in {"graph_link_prediction", "graph_link_classification"}:
            for k, v in self.link_predictor.state_dict().items():
                state_dict[f"link_predictor.{k}"] = v
            # state_dict["link_predictor"] = self.link_predictor.state_dict()
        else:
            raise NotImplementedError()

        return state_dict

    def custom_load_state_dict(self, state_dicts):
        if self.type in {"subword_ranker"}:
            self.target_embedder.load_state_dict(
                self.get_prefix("target_embedder", state_dicts)
            )
            self.link_predictor.load_state_dict(
                self.get_prefix("link_predictor", state_dicts)
            )
        elif self.type in {"graph_link_prediction", "graph_link_classification"}:
            self.link_predictor.load_state_dict(
                self.get_prefix("link_predictor", state_dicts)
            )
        else:
            raise NotImplementedError()

    def get_prefix(self, prefix, state_dict):
        return {key.replace(f"{prefix}.", ""): val for key, val in state_dict.items() if key.startswith(prefix)}