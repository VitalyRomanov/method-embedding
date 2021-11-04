import numpy as np
import random as rnd
import torch
from torch.nn import CrossEntropyLoss

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.mltools.torch import compute_accuracy
from SourceCodeTools.models.graph.ElementEmbedder import GraphLinkSampler
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.LinkPredictor import BilinearLinkPedictor, TransRLinkPredictor
from SourceCodeTools.models.graph.train.Scorer import Scorer
from SourceCodeTools.models.graph.train.objectives.GraphLinkObjective import GraphLinkObjective
from SourceCodeTools.tabular.common import compact_property


class GraphLinkClassificationObjective(GraphLinkObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, ns_groups=None
    ):
        super().__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, ns_groups=ns_groups
        )
        self.measure_scores = True
        self.update_embeddings_for_queries = False

    def create_graph_link_sampler(self, data_loading_func, nodes):
        self.target_embedder = TargetLinkMapper(
            elements=data_loading_func(), nodes=nodes
        )

    def create_link_predictor(self):
        self.link_predictor = BilinearLinkPedictor(
            self.target_emb_size, self.graph_model.emb_size, self.target_embedder.num_classes
        ).to(self.device)
        # self.positive_label = 1
        self.negative_label = self.target_embedder.null_class
        self.label_dtype = torch.long

    def create_positive_labels(self, ids):
        return torch.LongTensor(self.target_embedder.get_labels(ids))

    def compute_acc_loss(self, node_embs_, element_embs_, labels):
        logits = self.link_predictor(node_embs_, element_embs_)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1))

        acc = compute_accuracy(logits.argmax(dim=1), labels)

        return acc, loss


class TransRObjective(GraphLinkClassificationObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, ns_groups=None
    ):
        super().__init__(
            "TransR", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, ns_groups=ns_groups
        )

    def create_link_predictor(self):
        self.link_predictor = TransRLinkPredictor(
            input_dim=self.target_emb_size, rel_dim=30,
            num_relations=self.target_embedder.num_classes
        ).to(self.device)
        # self.positive_label = 1
        self.negative_label = -1
        self.label_dtype = torch.long

    def compute_acc_loss(self, node_embs_, element_embs_, labels):

        num_examples = len(labels) // 2
        anchor = node_embs_[:num_examples, :]
        positive = element_embs_[:num_examples, :]
        negative = element_embs_[num_examples:, :]
        labels_ = labels[:num_examples]

        loss, sim = self.link_predictor(anchor, positive, negative, labels_)
        acc = compute_accuracy(sim, labels >= 0)

        return acc, loss

class TargetLinkMapper(GraphLinkSampler):
    def __init__(self, elements, nodes):
        super(TargetLinkMapper, self).__init__(elements, nodes, compact_dst=False, dst_to_global=True, emb_size=1)

    def init(self, compact_dst):
        if compact_dst:
            elem2id, self.inverse_dst_map = compact_property(self.elements['dst'], return_order=True)
            self.elements['emb_id'] = self.elements['dst'].apply(lambda x: elem2id[x])
        else:
            self.elements['emb_id'] = self.elements['dst']

        self.link_type2id, self.inverse_link_type_map = compact_property(self.elements['type'], return_order=True, index_from_one=True)
        self.elements["link_type"] = list(map(lambda x: self.link_type2id[x], self.elements["type"].tolist()))

        self.element_lookup = {}
        for id_, emb_id, link_type in self.elements[["id", "emb_id", "link_type"]].values:
            if id_ in self.element_lookup:
                self.element_lookup[id_].append((emb_id, link_type))
            else:
                self.element_lookup[id_] = [(emb_id, link_type)]

        self.init_neg_sample()
        self.num_classes = len(self.inverse_link_type_map)
        self.null_class = 0

    def __getitem__(self, ids):
        self.cached_ids = ids
        node_ids, labels = zip(*(rnd.choice(self.element_lookup[id]) for id in ids))
        self.cached_labels = list(labels)
        return np.array(node_ids, dtype=np.int32)#, torch.LongTensor(np.array(labels, dtype=np.int32))

    def get_labels(self, ids):
        if self.cached_ids == ids:
            return self.cached_labels
        else:
            node_ids, labels = zip(*(rnd.choice(self.element_lookup[id]) for id in ids))
            return list(labels)

    def sample_negative(self, size, ids=None, strategy="w2v"):  # TODO switch to w2v?
        if strategy == "w2v":
            negative = ElementEmbedderBase.sample_negative(self, size)
        else:
            negative = Scorer.sample_closest_negative(self, ids, k=size // len(ids))
            assert len(negative) == size
        return negative
