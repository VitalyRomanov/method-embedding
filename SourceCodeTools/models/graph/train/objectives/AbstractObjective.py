import logging
from abc import abstractmethod
from collections import defaultdict

import dgl
import torch
from torch.nn import CosineEmbeddingLoss
from tqdm import tqdm

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.mltools.torch import compute_accuracy
from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedderWithBpeSubwords, GraphLinkSampler
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.LinkPredictor import CosineLinkPredictor, BilinearLinkPedictor, L2LinkPredictor

import torch.nn as nn


class ZeroEdges(Exception):
    def __init__(self, *args):
        super(ZeroEdges, self).__init__(*args)


class EarlyStoppingTracker:
    def __init__(self, early_stopping_tolerance):
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_counter = 0
        self.early_stopping_value = 0.
        self.early_stopping_trigger = False

    def should_stop(self, metric):
        if metric <= self.early_stopping_value:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_tolerance:
                return True
        else:
            self.early_stopping_counter = 0
            self.early_stopping_value = metric
        return False

    def reset(self):
        self.early_stopping_counter = 0
        self.early_stopping_value = 0.


def sum_scores(s):
    n = len(s)
    if n == 0:
        n += 1
    return sum(s) / n


class AbstractObjective(nn.Module):
    # # set in the init
    # name = None
    # graph_model = None
    # sampling_neighbourhood_size = None
    # batch_size = None
    # target_emb_size = None
    # node_embedder = None
    # device = None
    # masker = None
    # link_predictor_type = None
    # measure_scores = None
    # dilate_scores = None
    # early_stopping_tracker = None
    # early_stopping_trigger = None
    #
    # # set elsewhere
    # target_embedder = None
    # link_predictor = None
    # positive_label = None
    # negative_label = None
    # label_dtype = None
    #
    # train_loader = None
    # test_loader = None
    # val_loader = None
    # num_train_batches = None
    # num_test_batches = None
    # num_val_batches = None
    #
    # ntypes = None

    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
            ns_groups=None
    ):
        super(AbstractObjective, self).__init__()

        self.name = name
        self.graph_model = graph_model
        self.sampling_neighbourhood_size = sampling_neighbourhood_size
        self.batch_size = batch_size
        self.target_emb_size = target_emb_size
        self.node_embedder = node_embedder
        self.device = device
        self.masker = masker
        self.link_predictor_type = link_predictor_type
        self.measure_scores = measure_scores
        self.dilate_scores = dilate_scores
        self.nn_index = nn_index
        self.early_stopping_tracker = EarlyStoppingTracker(early_stopping_tolerance) if early_stopping else None
        self.early_stopping_trigger = False
        self.ns_groups = ns_groups

        self.verify_parameters()

        self.create_target_embedder(data_loading_func, nodes, tokenizer_path)
        self.create_link_predictor()
        self.create_loaders()

        self.target_embedding_fn = self.get_targets_from_embedder
        self.negative_factor = 1
        self.update_embeddings_for_queries = True

    @abstractmethod
    def verify_parameters(self):
        pass

    def create_base_element_sampler(self, data_loading_func, nodes):
        self.target_embedder = ElementEmbedderBase(
            elements=data_loading_func(), nodes=nodes, compact_dst=False, dst_to_global=True
        )

    def create_graph_link_sampler(self, data_loading_func, nodes):
        self.target_embedder = GraphLinkSampler(
            elements=data_loading_func(), nodes=nodes, compact_dst=False, dst_to_global=True,
            emb_size=self.target_emb_size, device=self.device, method=self.link_predictor_type, nn_index=self.nn_index,
            ns_groups=self.ns_groups
        )

    def create_subword_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = ElementEmbedderWithBpeSubwords(
            elements=data_loading_func(), nodes=nodes, emb_size=self.target_emb_size,
            tokenizer_path=tokenizer_path
        ).to(self.device)

    @abstractmethod
    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        # self.create_base_element_sampler(data_loading_func, nodes)
        # self.create_graph_link_sampler(data_loading_func, nodes)
        # self.create_subword_embedder(data_loading_func, nodes, tokenizer_path)
        raise NotImplementedError()

    def create_nn_link_predictor(self):
        self.link_predictor = BilinearLinkPedictor(self.target_emb_size, self.graph_model.emb_size, 2).to(self.device)
        self.positive_label = 1
        self.negative_label = 0
        self.label_dtype = torch.long

    def create_inner_prod_link_predictor(self):
        self.margin = -0.2
        self.target_embedder.set_margin(self.margin)
        self.link_predictor = CosineLinkPredictor(margin=self.margin).to(self.device)
        self.hinge_loss = nn.HingeEmbeddingLoss(margin=1. - self.margin)

        def cosine_loss(x1, x2, label):
            sim = nn.CosineSimilarity()
            dist = 1. - sim(x1, x2)
            return self.hinge_loss(dist, label)

        # self.cosine_loss = CosineEmbeddingLoss(margin=self.margin)
        self.cosine_loss = cosine_loss
        self.positive_label = 1.
        self.negative_label = -1.
        self.label_dtype = torch.float32

    def create_l2_link_predictor(self):
        self.margin = 2.0
        self.target_embedder.set_margin(self.margin)
        self.link_predictor = L2LinkPredictor().to(self.device)
        # self.hinge_loss = nn.HingeEmbeddingLoss(margin=self.margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin)

        def l2_loss(x1, x2, label):
            half = x1.shape[0] // 2
            pos = x2[:half, :]
            neg = x2[half:, :]

            return self.triplet_loss(x1[:half, :], pos, neg)
            # dist = torch.norm(x1 - x2, dim=-1)
            # return self.hinge_loss(dist, label)

        self.l2_loss = l2_loss
        self.positive_label = 1.
        self.negative_label = -1.
        self.label_dtype = torch.float32

    def create_link_predictor(self):
        if self.link_predictor_type == "nn":
            self.create_nn_link_predictor()
        elif self.link_predictor_type == "inner_prod":
            self.create_inner_prod_link_predictor()
        elif self.link_predictor_type == "l2":
            self.create_l2_link_predictor()
        else:
            raise NotImplementedError()

    def create_loaders(self):
        train_idx, val_idx, test_idx = self._get_training_targets()
        train_idx, val_idx, test_idx = self.target_embedder.create_idx_pools(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )
        logging.info(
            f"Pool sizes for {self.name}: train {self._idx_len(train_idx)}, "
            f"val {self._idx_len(val_idx)}, "
            f"test {self._idx_len(test_idx)}."
        )
        self.train_loader, self.test_loader, self.val_loader = self._get_loaders(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
            batch_size=self.batch_size  # batch_size_node_name
        )

        def get_num_nodes(ids):
            return sum(len(ids[key_]) for key_ in ids) // self.batch_size + 1

        self.num_train_batches = get_num_nodes(train_idx)
        self.num_test_batches = get_num_nodes(test_idx)
        self.num_val_batches = get_num_nodes(val_idx)

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

            def get_targets(data_label):
                return {
                    ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data[data_label], as_tuple=False).squeeze()
                    for ntype in self.ntypes
                }

            train_idx = get_targets("train_mask")
            val_idx = get_targets("val_mask")
            test_idx = get_targets("test_mask")

            # train_idx = {
            #     ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['train_mask'], as_tuple=False).squeeze()
            #     for ntype in self.ntypes
            # }
            # val_idx = {
            #     ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['val_mask'], as_tuple=False).squeeze()
            #     for ntype in self.ntypes
            # }
            # test_idx = {
            #     ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['test_mask'], as_tuple=False).squeeze()
            #     for ntype in self.ntypes
            # }
        else:
            # not sure when this is called
            raise NotImplementedError()

        return train_idx, val_idx, test_idx

    def _get_loaders(self, train_idx, val_idx, test_idx, batch_size):

        layers = self.graph_model.num_layers

        def create_loader(ids):
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(layers)
            loader = dgl.dataloading.NodeDataLoader(
                self.graph_model.g, ids, sampler, batch_size=batch_size, shuffle=False, num_workers=0)
            return loader

        train_loader = create_loader(train_idx)
        val_loader = create_loader(val_idx)
        test_loader = create_loader(test_idx)

        return train_loader, val_loader, test_loader

    def reset_iterator(self, data_split):
        iter_name = f"{data_split}_loader_iter"
        setattr(self, iter_name, iter(getattr(self, f"{data_split}_loader")))

    def loader_next(self, data_split):
        iter_name = f"{data_split}_loader_iter"
        if not hasattr(self, iter_name):
            setattr(self, iter_name, iter(getattr(self, f"{data_split}_loader")))
        return next(getattr(self, iter_name))

    def _create_loader(self, indices):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.graph_model.num_layers)
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

    def compute_acc_loss(self, node_embs_, element_embs_, labels):
        logits = self.link_predictor(node_embs_, element_embs_)

        if self.link_predictor_type == "nn":
            logp = nn.functional.log_softmax(logits, dim=1)
            loss = nn.functional.nll_loss(logp, labels)
        elif self.link_predictor_type == "inner_prod":
            loss = self.cosine_loss(node_embs_, element_embs_, labels)
            labels[labels < 0] = 0
        elif self.link_predictor_type == "l2":
            loss = self.l2_loss(node_embs_, element_embs_, labels)
            labels[labels < 0] = 0
            # num_examples = len(labels) // 2
            # anchor = node_embs_[:num_examples, :]
            # positive = element_embs_[:num_examples, :]
            # negative = element_embs_[num_examples:, :]
            # # pos_labels_ = labels[:num_examples]
            # # neg_labels_ = labels[num_examples:]
            # margin = 1.
            # triplet = nn.TripletMarginLoss(margin=margin)
            # self.target_embedder.set_margin(margin)
            # loss = triplet(anchor, positive, negative)
            # logits = (torch.norm(node_embs_ - element_embs_, keepdim=True) < 1.).float()
            # logits = torch.cat([1 - logits, logits], dim=1)
            # labels[labels < 0] = 0
        else:
            raise NotImplementedError()

        acc = compute_accuracy(logits.argmax(dim=1), labels)

        return acc, loss

    def _graph_embeddings(self, input_nodes, blocks, train_embeddings=True, masked=None):

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

    def sample_negative(self, ids, k, neg_sampling_strategy):
        if neg_sampling_strategy is not None:
            negative = self.target_embedder.sample_negative(
                k, ids=ids, strategy=neg_sampling_strategy
            )
        else:
            negative = self.target_embedder.sample_negative(
                k, ids=ids,
            )
        return negative

    def get_targets_from_nodes(
            self, positive_indices, negative_indices=None, train_embeddings=True
    ):
        negative_indices = torch.tensor(negative_indices, dtype=torch.long) if negative_indices is not None else None

        def get_embeddings_for_targets(dst):
            unique_dst, slice_map = self._handle_non_unique(dst)
            assert unique_dst[slice_map].tolist() == dst.tolist()

            dataloader = self._create_loader(unique_dst)
            input_nodes, dst_seeds, blocks = next(iter(dataloader))
            blocks = [blk.to(self.device) for blk in blocks]
            assert dst_seeds.shape == unique_dst.shape
            assert dst_seeds.tolist() == unique_dst.tolist()
            unique_dst_embeddings = self._graph_embeddings(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
            dst_embeddings = unique_dst_embeddings[slice_map.to(self.device)]

            if self.update_embeddings_for_queries:
                self.target_embedder.set_embed(unique_dst.detach().cpu().numpy(),
                                               unique_dst_embeddings.detach().cpu().numpy())

            return dst_embeddings

        positive_dst = get_embeddings_for_targets(positive_indices)
        negative_dst = get_embeddings_for_targets(negative_indices) if negative_indices is not None else None
        return positive_dst, negative_dst

    def get_targets_from_embedder(
            self, positive_indices, negative_indices=None, train_embeddings=True
    ):

        # def get_embeddings_for_targets(dst):
        #     unique_dst, slice_map = self._handle_non_unique(dst)
        #     assert unique_dst[slice_map].tolist() == dst.tolist()
        #     unique_dst_embeddings = self.target_embedder(unique_dst.to(self.device))
        #     dst_embeddings = unique_dst_embeddings[slice_map.to(self.device)]
        #
        #     if self.update_embeddings_for_queries:
        #         self.target_embedder.set_embed(unique_dst.detach().cpu().numpy(),
        #                                        unique_dst_embeddings.detach().cpu().numpy())
        #
        #     return dst_embeddings

        positive_dst = self.target_embedder(positive_indices.to(self.device))
        negative_dst = self.target_embedder(negative_indices.to(self.device)) if negative_indices is not None else None
        #
        # positive_dst = get_embeddings_for_targets(positive_indices)
        # negative_dst = get_embeddings_for_targets(negative_indices) if negative_indices is not None else None

        return positive_dst, negative_dst

    def create_positive_labels(self, ids):
        return torch.full((len(ids),), self.positive_label, dtype=self.label_dtype)

    def create_negative_labels(self, ids, k):
        return torch.full((len(ids) * k,), self.negative_label, dtype=self.label_dtype)

    def prepare_for_prediction(
            self, node_embeddings, seeds, target_embedding_fn, negative_factor=1,
            neg_sampling_strategy=None, train_embeddings=True,
    ):
        k = negative_factor
        indices = self.seeds_to_global(seeds).tolist()
        batch_size = len(indices)

        node_embeddings_batch = node_embeddings
        node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)

        positive_indices = self.target_embedder[indices]
        negative_indices = self.sample_negative(
            k=batch_size * k, ids=indices, neg_sampling_strategy=neg_sampling_strategy
        )

        positive_dst, negative_dst = target_embedding_fn(
            positive_indices, negative_indices, train_embeddings
        )

        # TODO breaks cache in
        #  SourceCodeTools.models.graph.train.objectives.GraphLinkClassificationObjective.TargetLinkMapper.get_labels
        labels_pos = self.create_positive_labels(indices)
        labels_neg = self.create_negative_labels(indices, k)

        src_embs = torch.cat([node_embeddings_batch, node_embeddings_neg_batch], dim=0)
        dst_embs = torch.cat([positive_dst, negative_dst], dim=0)
        labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)
        return src_embs, dst_embs, labels

    # def _logits_embedder(
    #         self, node_embeddings, elem_embedder, link_predictor, seeds, negative_factor=1, neg_sampling_strategy=None
    # ):
    #     k = negative_factor
    #     indices = self.seeds_to_global(seeds).tolist()
    #     batch_size = len(indices)
    #
    #     node_embeddings_batch = node_embeddings
    #     node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
    #
    #     element_embeddings = elem_embedder(elem_embedder[indices].to(self.device))
    #     negative_random = elem_embedder(self.sample_negative(
    #         k=batch_size * k, ids=indices, neg_sampling_strategy=neg_sampling_strategy
    #     ).to(self.device))
    #
    #     labels_pos = torch.full((batch_size,), self.positive_label, dtype=self.label_dtype)
    #     labels_neg = torch.full((batch_size * k,), self.negative_label, dtype=self.label_dtype)
    #
    #     src_embs = torch.cat([node_embeddings_batch, node_embeddings_neg_batch], dim=0)
    #     dst_embs = torch.cat([element_embeddings, negative_random], dim=0)
    #     labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)
    #     return src_embs, dst_embs, labels
    #
    # def _logits_nodes(
    #         self, node_embeddings, elem_embedder, link_predictor, create_dataloader,
    #         src_seeds, negative_factor=1, train_embeddings=True, neg_sampling_strategy=None,
    #         update_embeddings_for_queries=False
    # ):
    #     k = negative_factor
    #     indices = self.seeds_to_global(src_seeds).tolist()
    #     batch_size = len(indices)
    #
    #     node_embeddings_batch = node_embeddings
    #     node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
    #
    #     next_call_indices = elem_embedder[indices]  # this assumes indices is torch tensor
    #     negative_indices = torch.tensor(self.sample_negative(
    #         k=batch_size * k, ids=indices, neg_sampling_strategy=neg_sampling_strategy
    #     ), dtype=torch.long)
    #
    #     # dst targets are not unique
    #     def get_embeddings_for_targets(dst, update_embeddings_for_queries):
    #         unique_dst, slice_map = self._handle_non_unique(dst)
    #         assert unique_dst[slice_map].tolist() == dst.tolist()
    #
    #         dataloader = create_dataloader(unique_dst)
    #         input_nodes, dst_seeds, blocks = next(iter(dataloader))
    #         blocks = [blk.to(self.device) for blk in blocks]
    #         assert dst_seeds.shape == unique_dst.shape
    #         assert dst_seeds.tolist() == unique_dst.tolist()
    #         unique_dst_embeddings = self._graph_embeddings(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
    #         dst_embeddings = unique_dst_embeddings[slice_map.to(self.device)]
    #
    #         if update_embeddings_for_queries:
    #             self.target_embedder.set_embed(unique_dst.detach().cpu().numpy(),
    #                                            unique_dst_embeddings.detach().cpu().numpy())
    #
    #         return dst_embeddings
    #
    #     next_call_embeddings = get_embeddings_for_targets(next_call_indices, update_embeddings_for_queries)
    #     negative_random = get_embeddings_for_targets(negative_indices, update_embeddings_for_queries)
    #
    #     labels_pos = torch.full((batch_size,), self.positive_label, dtype=self.label_dtype)
    #     labels_neg = torch.full((batch_size * k,), self.negative_label, dtype=self.label_dtype)
    #
    #     src_embs = torch.cat([node_embeddings_batch, node_embeddings_neg_batch], dim=0)
    #     dst_embs = torch.cat([next_call_embeddings, negative_random], dim=0)
    #     labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)
    #     return src_embs, dst_embs, labels

    def seeds_to_python(self, seeds):
        if isinstance(seeds, dict):
            python_seeds = {}
            for key, val in seeds.items():
                python_seeds[key] = val.tolist()
        else:
            python_seeds = seeds.tolist()
        return python_seeds

    def forward(self, input_nodes, seeds, blocks, train_embeddings=True, neg_sampling_strategy=None):
        masked = self.masker.get_mask(self.seeds_to_python(seeds)) if self.masker is not None else None
        graph_emb = self._graph_embeddings(input_nodes, blocks, train_embeddings, masked=masked)
        node_embs_, element_embs_, labels = self.prepare_for_prediction(
            graph_emb, seeds, self.target_embedding_fn, negative_factor=self.negative_factor,
            neg_sampling_strategy=neg_sampling_strategy,
            train_embeddings=train_embeddings
        )

        acc, loss = self.compute_acc_loss(node_embs_, element_embs_, labels)

        return loss, acc

    def evaluate_objective(self, data_split, neg_sampling_strategy=None, negative_factor=1):
        # total_loss = 0
        # total_acc = 0
        at = [1, 3, 5, 10]
        # total_ndcg = {f"ndcg@{k}": 0. for k in ndcg_at}
        # ndcg_count = 0
        count = 0

        scores = defaultdict(list)

        for input_nodes, seeds, blocks in tqdm(
                getattr(self, f"{data_split}_loader"), total=getattr(self, f"num_{data_split}_batches")
        ):
            blocks = [blk.to(self.device) for blk in blocks]

            if self.masker is None:
                masked = None
            else:
                masked = self.masker.get_mask(self.seeds_to_python(seeds))

            src_embs = self._graph_embeddings(input_nodes, blocks, masked=masked)
            node_embs_, element_embs_, labels = self.prepare_for_prediction(
                src_embs, seeds, self.target_embedding_fn, negative_factor=negative_factor,
                neg_sampling_strategy=neg_sampling_strategy,
                train_embeddings=False
            )

            if self.measure_scores:
                if count % self.dilate_scores == 0:
                    scores_ = self.target_embedder.score_candidates(self.seeds_to_global(seeds), src_embs,
                                                                 self.link_predictor, at=at,
                                                                 type=self.link_predictor_type, device=self.device)
                    for key, val in scores_.items():
                        scores[key].append(val)

            acc, loss = self.compute_acc_loss(node_embs_, element_embs_, labels)

            scores["Loss"].append(loss.item())
            scores["Accuracy"].append(acc)
            count += 1

        scores = {key: sum_scores(val) for key, val in scores.items()}
        return scores
        # return total_loss / count, total_acc / count, {key: val / ndcg_count for key, val in
        #                                                total_ndcg.items()} if self.measure_scores else None

    # def _evaluate_embedder(self, ee, lp, data_split, neg_sampling_factor=1):
    #
    #     total_loss = 0
    #     total_acc = 0
    #     ndcg_at = [1, 3, 5, 10]
    #     total_ndcg = {f"ndcg@{k}": 0. for k in ndcg_at}
    #     ndcg_count = 0
    #     count = 0
    #
    #     for input_nodes, seeds, blocks in tqdm(
    #             getattr(self, f"{data_split}_loader"), total=getattr(self, f"num_{data_split}_batches")
    #     ):
    #         blocks = [blk.to(self.device) for blk in blocks]
    #
    #         if self.masker is None:
    #             masked = None
    #         else:
    #             masked = self.masker.get_mask(self.seeds_to_python(seeds))
    #
    #         src_embs = self._graph_embeddings(input_nodes, blocks, masked=masked)
    #         # logits, labels = self._logits_embedder(src_embs, ee, lp, seeds, neg_sampling_factor)
    #         node_embs_, element_embs_, labels = self._logits_embedder(src_embs, ee, lp, seeds, neg_sampling_factor)
    #
    #         if self.measure_scores:
    #             if count % self.dilate_scores == 0:
    #                 ndcg = self.target_embedder.score_candidates(self.seeds_to_global(seeds), src_embs, self.link_predictor, at=ndcg_at, type=self.link_predictor_type, device=self.device)
    #                 for key, val in ndcg.items():
    #                     total_ndcg[key] = total_ndcg[key] + val
    #                 ndcg_count += 1
    #
    #         # logits = self.link_predictor(node_embs_, element_embs_)
    #         #
    #         # logp = nn.functional.log_softmax(logits, 1)
    #         # loss = nn.functional.cross_entropy(logp, labels)
    #         # acc = _compute_accuracy(logp.argmax(dim=1), labels)
    #
    #         acc, loss = self.compute_acc_loss(node_embs_, element_embs_, labels)
    #
    #         total_loss += loss.item()
    #         total_acc += acc
    #         count += 1
    #     return total_loss / count, total_acc / count, {key: val / ndcg_count for key, val in total_ndcg.items()} if self.measure_scores else None
    #
    # def _evaluate_nodes(self, ee, lp, create_api_call_loader, data_split, neg_sampling_factor=1):
    #
    #     total_loss = 0
    #     total_acc = 0
    #     ndcg_at = [1, 3, 5, 10]
    #     total_ndcg = {f"ndcg@{k}": 0. for k in ndcg_at}
    #     ndcg_count = 0
    #     count = 0
    #
    #     for input_nodes, seeds, blocks in tqdm(
    #             getattr(self, f"{data_split}_loader"), total=getattr(self, f"num_{data_split}_batches")
    #     ):
    #         blocks = [blk.to(self.device) for blk in blocks]
    #
    #         if self.masker is None:
    #             masked = None
    #         else:
    #             masked = self.masker.get_mask(self.seeds_to_python(seeds))
    #
    #         src_embs = self._graph_embeddings(input_nodes, blocks, masked=masked)
    #         # logits, labels = self._logits_nodes(src_embs, ee, lp, create_api_call_loader, seeds, neg_sampling_factor)
    #         # node_embs_, element_embs_, labels = self._logits_nodes(src_embs, ee, lp, create_api_call_loader, seeds, neg_sampling_factor)
    #
    #         node_embs_, element_embs_, labels = self.prepare_for_prediction(
    #                 src_embs, seeds, self.target_embedding_fn, negative_factor=1,
    #                 neg_sampling_strategy=None, train_embeddings=True,
    #                 update_embeddings_for_queries=False
    #         )
    #
    #         if self.measure_scores:
    #             if count % self.dilate_scores == 0:
    #                 ndcg = self.target_embedder.score_candidates(self.seeds_to_global(seeds), src_embs, self.link_predictor, at=ndcg_at, type=self.link_predictor_type, device=self.device)
    #                 for key, val in ndcg.items():
    #                     total_ndcg[key] = total_ndcg[key] + val
    #                 ndcg_count += 1
    #
    #         # logits = self.link_predictor(node_embs_, element_embs_)
    #         #
    #         # logp = nn.functional.log_softmax(logits, 1)
    #         # loss = nn.functional.cross_entropy(logp, labels)
    #         # acc = _compute_accuracy(logp.argmax(dim=1), labels)
    #
    #         acc, loss = self.compute_acc_loss(node_embs_, element_embs_, labels)
    #
    #         total_loss += loss.item()
    #         total_acc += acc
    #         count += 1
    #     return total_loss / count, total_acc / count, {key: val / ndcg_count for key, val in total_ndcg.items()} if self.measure_scores else None

    def check_early_stopping(self, metric):
        """
        Checks the metric value and raises Early Stopping when the metric stops increasing.
            Assumes that the metric grows. Uses accuracy as a metric by default. Check implementation of child classes.
        :param metric: metric value
        :return: Nothing
        """
        if self.early_stopping_tracker is not None:
            self.early_stopping_trigger = self.early_stopping_tracker.should_stop(metric)

    def evaluate(self, data_split, *, neg_sampling_strategy=None, early_stopping=False, early_stopping_tolerance=20):
        # negative factor is 1 for evaluation
        scores = self.evaluate_objective(data_split, neg_sampling_strategy=None, negative_factor=1)
        if data_split == "val":
            self.check_early_stopping(scores["Accuracy"])
        return scores

    @abstractmethod
    def parameters(self, recurse: bool = True):
        raise NotImplementedError()

    @abstractmethod
    def custom_state_dict(self):
        raise NotImplementedError()

    @abstractmethod
    def custom_load_state_dict(self, state_dicts):
        raise NotImplementedError()

    def get_prefix(self, prefix, state_dict):
        return {key.replace(f"{prefix}.", ""): val for key, val in state_dict.items() if key.startswith(prefix)}
