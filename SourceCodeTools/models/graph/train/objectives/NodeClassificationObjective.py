import logging
from collections import OrderedDict
from itertools import chain
from typing import Tuple, Optional, Dict

import torch
from sklearn.metrics import ndcg_score, top_k_accuracy_score, f1_score
from torch import nn

from SourceCodeTools.models.graph.TargetLoader import TargetLoader
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective, ScoringMethods, \
    ObjectiveOutput

import numpy as np


class NodeClassifierObjective(AbstractObjective):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.update_embeddings_for_queries = True

    def _verify_parameters(self):
        pass

    def _create_target_embedder(self, target_emb_size, tokenizer_path):
        self.target_embedder = None

    def _create_scorers(self):
        for partition in ["train", "test", "val"]:
            setattr(self, f"{partition}_scorer", None)

    def _create_link_scorer(self):
        self.classifier = NodeClassifier(
            input_dims=self.target_emb_size, num_classes=self.dataloader.get_num_classes()
        ).to(self.device)
        self.link_scorer_type = ScoringMethods.nn
        self._loss_op = nn.CrossEntropyLoss()

        def compute_average_score(scores, labels=None):
            assert len(scores.shape) > 1 and scores.shape[1] > 1
            sm = nn.Softmax(dim=-1)
            scores = scores.cpu()
            labels = labels.cpu()
            return sm(scores)[torch.full(scores.shape, False).scatter_(1, labels.reshape(-1, 1), True)].mean().item()
            # return compute_accuracy(scores.argmax(dim=-1), labels)

        def compute_micro_f1(scores, labels=None):
            assert len(scores.shape) > 1 and scores.shape[1] > 1
            scores = scores.cpu()
            labels = labels.cpu()
            return f1_score(labels.numpy(), scores.argmax(-1).numpy(), average="micro")

        def compute_macro_f1(scores, labels=None):
            assert len(scores.shape) > 1 and scores.shape[1] > 1
            scores = scores.cpu()
            labels = labels.cpu()
            return f1_score(labels.numpy(), scores.argmax(-1).numpy(), average="macro")

        def compute_ndcg(scores, labels=None):
            assert len(scores.shape) > 1 and scores.shape[1] > 1
            scores = scores.cpu()
            labels = labels.cpu()
            return ndcg_score(torch.nn.functional.one_hot(labels, num_classes=scores.size(1)).numpy(), scores.numpy())

        self._compute_average_score = compute_average_score
        self._compute_micro_f1 = compute_micro_f1
        self._compute_macro_f1 = compute_macro_f1
        self._compute_ndcg = compute_ndcg

    def _compute_scores_loss(
            self, graph_emb, positive_emb, negative_emb, labels_pos, labels_neg
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict, torch.Tensor]:
        pos_scores = self.classifier(graph_emb)
        loss = self._loss_op(pos_scores, labels_pos)
        with torch.no_grad():
            scores = {
                f"positive_score/{self.link_scorer_type.name}": self._compute_average_score(pos_scores, labels_pos),
                f"micro_f1/{self.link_scorer_type.name}": self._compute_micro_f1(pos_scores, labels_pos),
                f"macro_f1/{self.link_scorer_type.name}": self._compute_macro_f1(pos_scores, labels_pos),
                f"ndcg/{self.link_scorer_type.name}": self._compute_ndcg(pos_scores, labels_pos),
            }
        return (pos_scores, None), scores, loss

    def _prepare_for_prediction(
            self, node_embeddings, positive_indices, negative_indices, target_embedding_fn, update_ns_callback
    ):
        return node_embeddings, None, None, positive_indices, None

    def _do_score_measurement(self, batch, graph_emb, longterm_metrics, scorer, **kwargs):
        at = [1, 3, 5, 10]

        logits = kwargs.pop("logits")[0]
        sm = nn.Softmax(dim=-1)
        y_pred = sm(logits).to("cpu").numpy()

        labels = kwargs.pop("y_true")[0].to("cpu").numpy()
        y_true = np.zeros(y_pred.shape)
        y_true[np.arange(0, y_true.shape[0]), labels] = 1.

        scores_ = {}
        y_true_onehot = np.array(y_true)
        labels = list(range(y_true_onehot.shape[1]))

        for k in at:
            if k >= y_pred.shape[1]:  # do not measure for binary classification
                if not hasattr(self, f"meaning_scores_warning_{k}"):
                    logging.warning(f"Disabling @{k} scores for task with {y_pred.shape[1]} classes")
                    setattr(self, f"meaning_scores_warning_{k}", True)
                continue  # scores do not have much sense in this situation
            scores_[f"ndcg@{k}"] = ndcg_score(y_true, y_pred, k=k)
            if y_pred.shape[1] == 2:
                accuracy = top_k_accuracy_score(y_true_onehot.argmax(-1), y_pred.argmax(-1), k=k, labels=labels)
            else:
                accuracy = top_k_accuracy_score(y_true_onehot.argmax(-1), y_pred, k=k, labels=labels)
            scores_[f"acc@{k}"] = accuracy
        for key, val in scores_.items():
            longterm_metrics[key].append(val)

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.classifier.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        return state_dict

    def custom_load_state_dict(self, state_dicts):
        self.classifier.load_state_dict(
            self.get_prefix("classifier", state_dicts)
        )


class NodeNameClassifier(NodeClassifierObjective):
    def __init__(
            self, **kwargs
    ):
        super().__init__(name="NodeNameClassifier", **kwargs)


class MisuseNodeClassifierObjective(NodeClassifierObjective):
    def __init__(self, *args, **kwargs):
        super(MisuseNodeClassifierObjective, self).__init__(*args, **kwargs)

    def _create_link_scorer(self):
        super(MisuseNodeClassifierObjective, self)._create_link_scorer()

        def compute_precision(scores, labels=None):
            scores = scores.cpu().argmax(dim=-1)
            if scores.sum() == 0.:
                return 0.
            if labels is not None:
                labels = labels.cpu()
            return ((labels * scores).sum() / scores.sum()).item()

        def compute_recall(scores, labels=None):
            assert labels.sum() > 0.
            scores = scores.cpu().argmax(dim=-1)
            if labels is not None:
                labels = labels.cpu()
            return ((labels * scores).sum() / labels.sum()).item()

        self._compute_precision = compute_precision
        self._compute_recall = compute_recall

    def _compute_scores_loss(
            self, graph_emb, positive_emb, negative_emb, labels_pos, labels_neg
    ) -> Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Dict, torch.Tensor]:
        pos_scores = self.classifier(graph_emb)
        loss = self._loss_op(pos_scores, labels_pos)
        with torch.no_grad():
            scores = {
                f"positive_score/{self.link_scorer_type.name}": self._compute_average_score(pos_scores, labels_pos),
                f"precision/{self.link_scorer_type.name}": self._compute_precision(pos_scores, labels_pos),
                f"recall/{self.link_scorer_type.name}": self._compute_recall(pos_scores, labels_pos),
            }
        return (pos_scores, None), scores, loss

    def forward(
            self, input_nodes, input_mask, blocks, positive_indices, negative_indices,
            update_ns_callback=None, misuse_node_mask=None, **kwargs
    ):
        gnn_output = self._graph_embeddings(input_nodes, blocks, mask=input_mask)
        graph_emb = gnn_output.output

        _, positive_emb, negative_emb, labels_pos, labels_neg = self._prepare_for_prediction(
            graph_emb, positive_indices, negative_indices, self.target_embedding_fn, update_ns_callback  # , subgraph
        )

        if misuse_node_mask is not None:
            graph_emb = graph_emb[misuse_node_mask, :]
            labels_pos = labels_pos[misuse_node_mask]

        pos_neg_scores, avg_scores, loss  = self._compute_scores_loss(
            graph_emb, positive_emb, negative_emb, labels_pos, labels_neg
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


class ClassifierTargetMapper(TargetLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def num_classes(self):
        return len(self._label_encoder._inverse_target_map)

    def set_embed(self, *args, **kwargs):
        pass

    def prepare_index(self, *args):
        pass


class NodeClassifier(nn.Module):
    def __init__(self, input_dims, num_classes, hidden=100):
        super().__init__()

        self.l1 = nn.Linear(input_dims, hidden)
        self.l1_a = nn.LeakyReLU()

        self.logits = nn.Linear(hidden, num_classes)

    def forward(self, x, **kwargs):
        x = self.l1_a(self.l1(x))
        return self.logits(x)