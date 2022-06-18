import logging
from collections import OrderedDict, defaultdict
from itertools import chain

import torch
from sklearn.metrics import ndcg_score, top_k_accuracy_score
from torch import nn
from torch.nn import CrossEntropyLoss

from SourceCodeTools.models.graph.TargetLoader import TargetLoader
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective, compute_accuracy
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.train.Scorer import Scorer

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

    def _create_link_predictor(self):
        self.classifier = NodeClassifier(
            input_dims=self.target_emb_size, num_classes=self.dataloader.train_loader.num_classes
        ).to(self.device)

    def _compute_acc_loss(self, graph_emb, element_emb, labels):
        logits = self.classifier(graph_emb)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1))

        acc = compute_accuracy(logits.argmax(dim=1), labels)

        return logits, acc, loss

    def _prepare_for_prediction(
            self, node_embeddings, positive_indices, negative_indices, target_embedding_fn, update_ns_callback, graph
    ):
        return node_embeddings, None, positive_indices

    def _do_score_measurement(self, batch, graph_emb, longterm_metrics, scorer, **kwargs):
        at = [1, 3, 5, 10]

        logits = kwargs.pop("logits")
        y_pred = nn.functional.softmax(logits, dim=-1).to("cpu").numpy()

        labels = kwargs.pop("y_true").to("cpu").numpy()
        y_true = np.zeros(y_pred.shape)
        y_true[np.arange(0, y_true.shape[0]), labels] = 1.

        # scores_ = scorer.score_candidates(
        #     batch["indices"], graph_emb, self.link_predictor, at=,
        #     type=self.link_predictor_type, device=self.device
        # )
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
            scores_[f"acc@{k}"] = top_k_accuracy_score(y_true_onehot.argmax(-1), y_pred, k=k, labels=labels)
        for key, val in scores_.items():
            longterm_metrics[key].append(val)

    # def _evaluate_objective(self, data_split, neg_sampling_strategy=None, negative_factor=1):
    #     at = [1, 3, 5, 10]
    #     count = 0
    #     scores = defaultdict(list)
    #
    #     for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
    #         blocks = [blk.to(self.device) for blk in blocks]
    #
    #         if self.masker is None:
    #             masked = None
    #         else:
    #             masked = self.masker.get_mask(self.seeds_to_python(seeds))
    #
    #         src_embs = self._graph_embeddings(input_nodes, blocks, masked=masked)
    #         node_embs_, element_embs_, labels = self.prepare_for_prediction(
    #             src_embs, seeds, self.target_embedding_fn, negative_factor=negative_factor,
    #             neg_sampling_strategy=neg_sampling_strategy,
    #             train_embeddings=False
    #         )
    #         # indices = self.seeds_to_global(seeds).tolist()
    #         # labels = self.target_embedder[indices]
    #         # labels = torch.LongTensor(labels).to(self.device)
    #         acc, loss, logits = self._compute_acc_loss(node_embs_, element_embs_, labels)
    #
    #         y_pred = nn.functional.softmax(logits, dim=-1).to("cpu").numpy()
    #         y_true = np.zeros(y_pred.shape)
    #         y_true[np.arange(0, y_true.shape[0]), labels.to("cpu").numpy()] = 1.
    #
    #         if self.measure_scores:
    #             if y_pred.shape[1] == 2:
    #                 logging.warning("Scores are meaningless for binary classification. Disabling.")
    #                 self.measure_scores = False
    #             else:
    #                 if count % self.dilate_scores == 0:
    #                     y_true_onehot = np.array(y_true)
    #                     labels = list(range(y_true_onehot.shape[1]))
    #
    #                     for k in at:
    #                         if k >= y_pred.shape[1]:  # do not measure for binary classification
    #                             if not hasattr(self, f"meaning_scores_warning_{k}"):
    #                                 logging.warning(f"Disabling @{k} scores for task with {y_pred.shape[1]} classes")
    #                                 setattr(self, f"meaning_scores_warning_{k}", True)
    #                             continue  # scores do not have much sense in this situation
    #                         scores[f"ndcg@{k}"].append(ndcg_score(y_true, y_pred, k=k))
    #                         scores[f"acc@{k}"].append(
    #                             top_k_accuracy_score(y_true_onehot.argmax(-1), y_pred, k=k, labels=labels)
    #                         )
    #
    #         scores["Loss"].append(loss.item())
    #         scores["Accuracy"].append(acc)
    #         count += 1
    #
    #     if count == 0:
    #         count += 1
    #
    #     scores = {key: self._sum_scores(val) for key, val in scores.items()}
    #     return scores

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.classifier.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        return state_dict


class NodeNameClassifier(NodeClassifierObjective):
    def __init__(
            self, **kwargs
    ):
        super().__init__(name="NodeNameClassifier", **kwargs)


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