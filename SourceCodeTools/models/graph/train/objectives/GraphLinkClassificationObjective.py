import logging
import sys
from collections import defaultdict

import numpy as np
import random as rnd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from SourceCodeTools.code.data.dataset.SubwordMasker import SubwordMasker
from SourceCodeTools.mltools.torch import compute_accuracy
from SourceCodeTools.models.graph.ElementEmbedder import GraphLinkSampler
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.LinkPredictor import BilinearLinkClassifier, TransRLinkScorer, LinkClassifier
from SourceCodeTools.models.graph.train.Scorer import Scorer
from SourceCodeTools.models.graph.train.objectives.GraphLinkObjective import GraphLinkObjective
from SourceCodeTools.tabular.common import compact_property


class GraphLinkClassificationObjective(GraphLinkObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # raise Exception('why measure score is set true')
        self.measure_scores = False
        self.update_embeddings_for_queries = False

    def create_graph_link_sampler(self, data_loading_func, *args, **kwargs):
        raise NotImplementedError()
        # self.target_embedder = TargetLinkMapper(
        #     elements=data_loading_func(), nodes=nodes, emb_size=self.target_emb_size, ns_groups=self.ns_groups
        # )

    def _create_link_scorer(self):
        self.link_scorer = LinkClassifier(
            self.graph_model.emb_size, self.target_emb_size, self.dataloader.train_loader.num_classes
        ).to(self.device)
        self._loss_op = nn.CrossEntropyLoss()

        def compute_average_score(scores, labels=None):
            assert len(scores.shape) > 1 and scores.shape[1] > 1
            sm = nn.Softmax(dim=-1)
            scores = scores.cpu()
            labels = labels.cpu()
            return sm(scores)[torch.full(scores.shape, False).scatter_(1, labels.reshape(-1, 1), True)].mean().item()
            # return compute_accuracy(scores.argmax(dim=-1), labels)

        self._compute_average_score = compute_average_score

    # def _create_positive_labels(self, ids):
    #     return torch.LongTensor(self.target_embedder.get_labels(ids))

    def _compute_scores_loss(self, node_embs, positive_embs, negative_embs, positive_labels, negative_labels):
        pos_scores = self.link_scorer(node_embs, positive_embs)
        loss = self._loss_op(pos_scores, positive_labels)
        with torch.no_grad():
            scores = {
                f"positive_score/{self.link_scorer_type.name}": self._compute_average_score(pos_scores, positive_labels),
            }
        return (pos_scores, None), scores, loss

    def forward(
            self, input_nodes, input_mask, blocks, positive_indices, negative_indices,
            update_ns_callback=None, subgraph=None, **kwargs
    ):
        unique_embeddings = self._graph_embeddings(input_nodes, blocks, mask=input_mask)

        all_embeddings = unique_embeddings[kwargs["slice_map"]]

        graph_embeddings = all_embeddings[kwargs["src_nodes_mask"]]
        positive_embeddings = all_embeddings[kwargs["positive_nodes_mask"]]
        negative_embeddings = all_embeddings[kwargs["negative_nodes_mask"]]

        # non_src_nodes_mask = ~kwargs["src_nodes_mask"]
        # non_src_ids = kwargs["compute_embeddings_for"][non_src_nodes_mask]
        # non_src_embeddings = all_embeddings[non_src_nodes_mask].cpu().detach().numpy()

        pos_labels = kwargs["positive_labels"]  #  self._create_positive_labels(positive_indices).to(self.device)
        neg_labels = kwargs["negative_labels"]  # self._create_negative_labels(negative_embeddings).to(self.device)

        pos_neg_scores, avg_scores, loss = self._compute_scores_loss(
            graph_embeddings, positive_embeddings, negative_embeddings, pos_labels, neg_labels
        )

        return graph_embeddings, pos_neg_scores, (pos_labels, neg_labels), loss, avg_scores


class GraphLinkMisuseObjective(GraphLinkClassificationObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def _create_link_scorer(self):
        super(GraphLinkMisuseObjective, self)._create_link_scorer()

        def compute_binary_precision(scores, labels=None):
            scores = scores.cpu().argmax(dim=-1)
            if scores.sum() == 0.:
                return 0.
            return ((labels * scores).sum() / scores.sum()).item()

        def compute_binary_recall(scores, labels=None):
            labels_sum = labels.sum()
            if labels_sum == 0:
                logging.warning("Trying to compute recall for batch without positive labels. Skipping.")
                labels_sum = 1.0
            scores = scores.cpu().argmax(dim=-1)
            return ((labels * scores).sum() / labels_sum).item()

        self._compute_precision = compute_binary_precision
        self._compute_recall = compute_binary_recall

    def _compute_scores_loss(self, node_embs, positive_embs, negative_embs, positive_labels, negative_labels):

        pos_scores = self.link_scorer(node_embs[:len(positive_labels)], positive_embs)
        # neg_scores = self.link_scorer(node_embs[len(positive_labels):], negative_embs)
        loss = self._loss_op(
            # torch.cat([pos_scores, neg_scores]),
            # torch.cat([positive_labels, negative_labels])
            pos_scores,
            positive_labels
        )
        with torch.no_grad():
            misuse_mask = positive_labels == 1
            scores = {
                f"positive_score/{self.link_scorer_type.name}": self._compute_average_score(pos_scores, positive_labels),
                f"misuse_score/{self.link_scorer_type.name}": self._compute_average_score(pos_scores[misuse_mask], positive_labels[misuse_mask]),
                f"precision/{self.link_scorer_type.name}": self._compute_precision(pos_scores, positive_labels),
                f"recall/{self.link_scorer_type.name}": self._compute_recall(pos_scores, positive_labels),
                # f"negative_score/{self.link_scorer_type.name}": self._compute_average_score(neg_scores, negative_labels)
            }
        return (pos_scores, None), scores, loss

    def forward(
            self, input_nodes, input_mask, blocks, src_slice_map, dst_slice_map, labels,
            update_ns_callback=None, subgraph=None, **kwargs
    ):
        unique_embeddings = self._graph_embeddings(input_nodes, blocks, mask=input_mask)

        src_embeddings = unique_embeddings[src_slice_map]
        dst_embeddings = unique_embeddings[dst_slice_map]

        pos_neg_scores, avg_scores, loss = self._compute_scores_loss(
            src_embeddings, dst_embeddings, None, labels, None
        )

        return src_embeddings, pos_neg_scores, (labels, None), loss, avg_scores



class TransRObjective(GraphLinkClassificationObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_link_scorer(self):
        self.link_scorer = TransRLinkScorer(
            self.target_emb_size, self.target_emb_size, h_dim=30,
            num_classes=self.target_embedder.num_classes
        ).to(self.device)
        self._loss_op = nn.CrossEntropyLoss()

        def compute_average_score(scores, labels=None):
            assert len(scores.shape) > 1 and scores.shape[1] > 1
            sm = nn.Softmax(dim=-1)
            scores = scores.cpu()
            labels = labels.cpu()
            return sm(scores)[torch.full(scores.shape, False).scatter_(1, labels.reshape(-1, 1), True)].mean().item()
            # return compute_accuracy(scores.argmax(dim=-1), labels)

        self._compute_average_score = compute_average_score

    # def _compute_scores_loss(self, node_embs_, element_embs_, labels):
    #
    #     num_examples = len(labels) // 2
    #     anchor = node_embs_[:num_examples, :]
    #     positive = element_embs_[:num_examples, :]
    #     negative = element_embs_[num_examples:, :]
    #     labels_ = labels[:num_examples]
    #
    #     loss, sim = self.link_predictor(anchor, positive, negative, labels_)
    #     acc = compute_accuracy(sim, labels >= 0)
    #
    #     return acc, loss


# class TargetLinkMapper(GraphLinkSampler):
#     def __init__(self, elements, emb_size=1, ns_groups=None):
#         super(TargetLinkMapper, self).__init__(
#             elements, compact_dst=False, emb_size=emb_size, ns_groups=ns_groups
#         )
#
#     def init(self, elements, compact_dst):
#         if len(elements) == 0:
#             logging.error(f"Not enough data for the embedder: {len(elements)}. Exiting...")
#             sys.exit()
#
#         compacted = self.compact_dst(elements, compact_dst)
#
#         self.link_type2id, self.inverse_link_type_map = compact_property(elements['type'], return_order=True, index_from_one=True)
#         links_type = list(map(lambda x: self.link_type2id[x], elements["type"].tolist()))
#
#         self.element_lookup = defaultdict(list)
#         for src, dst, link_type in zip(elements["id"], compacted, links_type):
#             self.element_lookup[src].append((dst, link_type))
#
#         self.init_neg_sample(elements)
#         self.num_classes = len(self.inverse_link_type_map)
#         self.null_class = 0
#
#     def sample_positive(self, ids):
#         self.cached_ids = ids
#         node_ids, labels = zip(*(rnd.choice(self.element_lookup[id]) for id in ids))
#         self.cached_labels = list(labels)
#         return np.array(node_ids, dtype=np.int32)#, torch.LongTensor(np.array(labels, dtype=np.int32))
#
#     def get_labels(self, ids):
#         if self.cached_ids == ids:
#             return self.cached_labels
#         else:
#             node_ids, labels = zip(*(rnd.choice(self.element_lookup[id]) for id in ids))
#             return list(labels)
#
#     def sample_negative(self, size, ids=None, strategy="w2v"):  # TODO switch to w2v?
#         if strategy == "w2v":
#             negative = ElementEmbedderBase.sample_negative(self, size)
#         else:
#             negative = Scorer.sample_closest_negative(self, ids, k=size // len(ids))
#             assert len(negative) == size
#         return negative
