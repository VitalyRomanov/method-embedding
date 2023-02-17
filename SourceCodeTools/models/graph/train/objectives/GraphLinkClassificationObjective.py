import logging
from typing import Tuple, Optional, Dict

import torch
from torch import nn

from SourceCodeTools.models.graph.LinkPredictor import BilinearLinkClassifier, TransRLinkScorer, LinkClassifier, \
    DistMultLinkScorer
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import ObjectiveOutput
from SourceCodeTools.models.graph.train.objectives.GraphLinkObjective import GraphLinkObjective


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

    def _compute_scores_loss(
            self, node_embs, positive_embs, negative_embs, positive_labels, negative_labels
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict, torch.Tensor]:
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
        gnn_output = self._graph_embeddings(input_nodes, blocks, mask=input_mask)
        unique_embeddings = gnn_output.output

        all_embeddings = unique_embeddings[kwargs["slice_map"]]

        graph_embeddings = all_embeddings[kwargs["src_nodes_mask"]]
        positive_embeddings = all_embeddings[kwargs["positive_nodes_mask"]]
        negative_embeddings = all_embeddings[kwargs["negative_nodes_mask"]]

        # non_src_nodes_mask = ~kwargs["src_nodes_mask"]
        # non_src_ids = kwargs["compute_embeddings_for"][non_src_nodes_mask]
        # non_src_embeddings = all_embeddings[non_src_nodes_mask].cpu().detach().numpy()

        labels_pos = kwargs["positive_labels"]  #  self._create_positive_labels(positive_indices).to(self.device)
        labels_neg = kwargs["negative_labels"]  # self._create_negative_labels(negative_embeddings).to(self.device)

        pos_neg_scores, avg_scores, loss = self._compute_scores_loss(
            graph_embeddings, positive_embeddings, negative_embeddings, labels_pos, labels_neg
        )

        return ObjectiveOutput(
            gnn_output=gnn_output,
            logits=pos_neg_scores,
            labels=(labels_pos, labels_neg),
            loss=loss,
            scores=avg_scores,
            prediction=(
                torch.softmax(pos_neg_scores[0], dim=-1),
                torch.softmax(pos_neg_scores[1], dim=-1) if pos_neg_scores[1] is not None else None
            )
        )


class GraphLinkMisuseObjective(GraphLinkClassificationObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def _create_link_scorer(self):
        super(GraphLinkMisuseObjective, self)._create_link_scorer()

        def compute_binary_precision(scores, labels=None):
            scores = scores.cpu().argmax(dim=-1)
            if labels is not None:
                labels = labels.cpu()
            if scores.sum() == 0.:
                return 0.
            return ((labels * scores).sum() / scores.sum()).item()

        def compute_binary_recall(scores, labels=None):
            labels_sum = labels.sum()
            if labels is not None:
                labels = labels.cpu()
            if labels_sum == 0:
                logging.warning("Trying to compute recall for batch without positive labels. Skipping.")
                labels_sum = 1.0
            scores = scores.cpu().argmax(dim=-1)
            return ((labels * scores).sum() / labels_sum).item()

        self._compute_precision = compute_binary_precision
        self._compute_recall = compute_binary_recall

    def _compute_scores_loss(
            self, node_embs, positive_embs, negative_embs, positive_labels, negative_labels
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict, torch.Tensor]:

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
            self, input_nodes, input_mask, blocks, src_slice_map, dst_slice_map, labels=None,
            update_ns_callback=None, subgraph=None, **kwargs
    ):
        gnn_output = self._graph_embeddings(input_nodes, blocks, mask=input_mask)
        unique_embeddings = gnn_output.output

        src_embeddings = unique_embeddings[src_slice_map]
        dst_embeddings = unique_embeddings[dst_slice_map]

        pos_neg_scores, avg_scores, loss = self._compute_scores_loss(
            src_embeddings, dst_embeddings, None, labels, None
        )

        return ObjectiveOutput(
            gnn_output=gnn_output,
            logits=pos_neg_scores,
            labels=(labels, None),
            loss=loss,
            scores=avg_scores,
            prediction=(
                torch.softmax(pos_neg_scores[0], dim=-1),
                torch.softmax(pos_neg_scores[1], dim=-1) if pos_neg_scores[1] is not None else None
            )
        )


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


class RelationalDistMult(GraphLinkClassificationObjective):
    def __init__(self, *args, **kwargs):
        self.edge_type_params = kwargs.pop("edge_type_params")
        super(RelationalDistMult, self).__init__(*args, **kwargs)

    def _create_link_scorer(self):
        edge_type_params = self.edge_type_params[1]
        edge_type_mapping = self.edge_type_params[0]
        self.link_scorer = DistMultLinkScorer(
            edge_type_params.shape[1], edge_type_params.shape[0], edge_type_params, finetune=False, edge_type_mapping=edge_type_mapping
        ).to(self.device)
        self._loss_op = nn.BCELoss()

        def compute_average_score(scores, labels=None):
            scores = scores.cpu()
            labels = labels.cpu()
            return scores.mean().item()
            # return (((scores > 0.) == labels).sum() / len(scores)).item()
            # return compute_accuracy(scores.argmax(dim=-1), labels)

        self._compute_average_score = compute_average_score

    def _compute_scores_loss(
            self, node_embs, positive_embs, negative_node_embs, negative_embs, positive_labels, negative_labels
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict, torch.Tensor]:

        pos_scores = self.link_scorer(node_embs, positive_embs, positive_labels)
        neg_scores = self.link_scorer(negative_node_embs, negative_embs, negative_labels)

        pos_labels = torch.ones(len(positive_labels), dtype=torch.float32)
        neg_labels = torch.zeros(len(negative_labels), dtype=torch.float32)

        edge_scores = torch.cat([torch.sigmoid(pos_labels), torch.sigmoid(neg_labels)], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        pos_loss = self._loss_op(
            torch.sigmoid(pos_scores),
            pos_labels
        )
        neg_loss = self._loss_op(
            torch.sigmoid(neg_scores),
            neg_labels
        ) / len(neg_labels)
        # loss = self._loss_op(
        #     edge_scores,
        #     labels
        # )
        loss = pos_loss.mean() + neg_loss.mean()
        with torch.no_grad():
            scores = {
                f"positive_score/{self.link_scorer_type.name}": self._compute_average_score(pos_scores, pos_labels),
                f"negative_score/{self.link_scorer_type.name}": self._compute_average_score(neg_scores, neg_labels),
            }
        return (pos_scores, neg_scores), scores, loss

    def forward(
            self, input_nodes, input_mask, blocks, src_slice_map, dst_slice_map, labels,
            neg_src_slice_map, neg_dst_slice_map, neg_labels,
            update_ns_callback=None, subgraph=None, **kwargs
    ):
        unique_embeddings = self._extract_embed(blocks[0].dstdata["original_id"])

        src_embeddings = unique_embeddings[src_slice_map]
        dst_embeddings = unique_embeddings[dst_slice_map]
        neg_src_embeddings = unique_embeddings[neg_src_slice_map]
        neg_dst_embeddings = unique_embeddings[neg_dst_slice_map]

        # assert (src_slice_map == neg_src_slice_map).all()

        pos_neg_scores, avg_scores, loss = self._compute_scores_loss(
            src_embeddings, dst_embeddings, neg_src_embeddings, neg_dst_embeddings, kwargs["original_edge_types"], kwargs["original_neg_edge_types"]
        )

        return ObjectiveOutput(
            gnn_output=None,
            logits=pos_neg_scores,
            labels=(labels, neg_labels),
            loss=loss,
            scores=avg_scores,
            prediction=(
                torch.softmax(pos_neg_scores[0], dim=-1),
                torch.softmax(pos_neg_scores[1], dim=-1) if pos_neg_scores[1] is not None else None
            )
        )