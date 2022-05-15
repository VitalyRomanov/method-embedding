import logging
import sys
from collections import defaultdict

import numpy as np
import random as rnd
import torch
from torch.nn import CrossEntropyLoss

from SourceCodeTools.code.data.dataset.SubwordMasker import SubwordMasker
from SourceCodeTools.mltools.torch import compute_accuracy
from SourceCodeTools.models.graph.ElementEmbedder import GraphLinkSampler
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.LinkPredictor import BilinearLinkPedictor, TransRLinkPredictor
from SourceCodeTools.models.graph.train.Scorer import Scorer
from SourceCodeTools.models.graph.train.objectives.GraphLinkObjective import GraphLinkObjective
from SourceCodeTools.tabular.common import compact_property


class GraphLinkClassificationObjective(GraphLinkObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.measure_scores = True
        self.update_embeddings_for_queries = True

    def create_graph_link_sampler(self, data_loading_func, *args, **kwargs):
        raise NotImplementedError()
        # self.target_embedder = TargetLinkMapper(
        #     elements=data_loading_func(), nodes=nodes, emb_size=self.target_emb_size, ns_groups=self.ns_groups
        # )

    def _create_link_predictor(self):
        self.link_predictor = BilinearLinkPedictor(
            self.target_emb_size, self.graph_model.emb_size, self.target_embedder.num_classes
        ).to(self.device)
        # self.positive_label = 1
        self.negative_label = self.target_embedder.null_class
        self.label_dtype = torch.long

    def _create_positive_labels(self, ids):
        return torch.LongTensor(self.target_embedder.get_labels(ids))

    def _compute_acc_loss(self, node_embs_, element_embs_, labels):
        logits = self.link_predictor(node_embs_, element_embs_)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1))

        acc = compute_accuracy(logits.argmax(dim=1), labels)

        return acc, loss


class TransRObjective(GraphLinkClassificationObjective):
    def __init__(self, **kwargs):
        super().__init__(name="TransR", **kwargs)

    def _create_link_predictor(self):
        self.link_predictor = TransRLinkPredictor(
            input_dim=self.target_emb_size, rel_dim=30,
            num_relations=self.target_embedder.num_classes
        ).to(self.device)
        # self.positive_label = 1
        self.negative_label = -1
        self.label_dtype = torch.long

    def _compute_acc_loss(self, node_embs_, element_embs_, labels):

        num_examples = len(labels) // 2
        anchor = node_embs_[:num_examples, :]
        positive = element_embs_[:num_examples, :]
        negative = element_embs_[num_examples:, :]
        labels_ = labels[:num_examples]

        loss, sim = self.link_predictor(anchor, positive, negative, labels_)
        acc = compute_accuracy(sim, labels >= 0)

        return acc, loss


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
