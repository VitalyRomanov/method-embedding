import logging
from collections import OrderedDict, defaultdict
from itertools import chain

import torch
from sklearn.metrics import ndcg_score, top_k_accuracy_score
from torch import nn
from torch.nn import CrossEntropyLoss

from SourceCodeTools.code.data.dataset import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective, compute_accuracy, \
    sum_scores
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.train.Scorer import Scorer

import numpy as np


class NodeClassifierObjective(AbstractObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type=None, masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, early_stopping=False, early_stopping_tolerance=20
    ):
        super().__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, early_stopping=early_stopping, early_stopping_tolerance=early_stopping_tolerance
        )

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = ClassifierTargetMapper(
            elements=data_loading_func(), nodes=nodes
        )

    def create_link_predictor(self):
        self.classifier = NodeClassifier(
            self.target_emb_size, self.target_embedder.num_classes).to(self.device)

    def compute_acc_loss(self, graph_emb, element_emb, labels, return_logits=False):
        logits = self.classifier(graph_emb)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1))

        acc = compute_accuracy(logits.argmax(dim=1), labels)

        if return_logits:
            return acc, loss, logits
        return acc, loss

    def prepare_for_prediction(
            self, node_embeddings, seeds, target_embedding_fn, negative_factor=1,
            neg_sampling_strategy=None, train_embeddings=True,
    ):
        indices = self.seeds_to_global(seeds).tolist()
        labels = torch.LongTensor(
            self.target_embedder[indices]).to(self.device)

        return node_embeddings, None, labels

    def evaluate_objective(self, data_split, neg_sampling_strategy=None, negative_factor=1):
        at = [1, 3, 5, 10]
        count = 0
        scores = defaultdict(list)

        for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
            blocks = [blk.to(self.device) for blk in blocks]

            if self.masker is None:
                masked = None
            else:
                masked = self.masker.get_mask(self.seeds_to_python(seeds))

            src_embs = self._graph_embeddings(
                input_nodes, blocks, masked=masked)
            node_embs_, element_embs_, labels = self.prepare_for_prediction(
                src_embs, seeds, self.target_embedding_fn, negative_factor=negative_factor,
                neg_sampling_strategy=neg_sampling_strategy,
                train_embeddings=False
            )
            # indices = self.seeds_to_global(seeds).tolist()
            # labels = self.target_embedder[indices]
            # labels = torch.LongTensor(labels).to(self.device)
            acc, loss, logits = self.compute_acc_loss(
                node_embs_, element_embs_, labels, return_logits=True)

            y_pred = nn.functional.softmax(logits, dim=-1).to("cpu").numpy()
            y_true = np.zeros(y_pred.shape)
            y_true[np.arange(0, y_true.shape[0]),
                   labels.to("cpu").numpy()] = 1.

            if self.measure_scores:
                if y_pred.shape[1] == 2:
                    logging.warning(
                        "Scores are meaningless for binary classification. Disabling.")
                    self.measure_scores = False
                else:
                    if count % self.dilate_scores == 0:
                        y_true_onehot = np.array(y_true)
                        labels = list(range(y_true_onehot.shape[1]))

                        for k in at:
                            # do not measure for binary classification
                            if k >= y_pred.shape[1]:
                                if not hasattr(self, f"meaning_scores_warning_{k}"):
                                    logging.warning(
                                        f"Disabling @{k} scores for task with {y_pred.shape[1]} classes")
                                    setattr(
                                        self, f"meaning_scores_warning_{k}", True)
                                continue  # scores do not have much sense in this situation
                            scores[f"ndcg@{k}"].append(
                                ndcg_score(y_true, y_pred, k=k))
                            scores[f"acc@{k}"].append(
                                top_k_accuracy_score(
                                    y_true_onehot.argmax(-1), y_pred, k=k, labels=labels)
                            )

            scores["Loss"].append(loss.item())
            scores["Accuracy"].append(acc)
            count += 1

        if count == 0:
            count += 1

        scores = {key: sum_scores(val) for key, val in scores.items()}
        return scores

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.classifier.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        return state_dict


class NodeNameClassifier(NodeClassifierObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type=None, masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, early_stopping=False, early_stopping_tolerance=20
    ):
        super().__init__(
            "NodeNameClassifier", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, early_stopping=early_stopping, early_stopping_tolerance=early_stopping_tolerance
        )


class ClassifierTargetMapper(ElementEmbedderBase, Scorer):
    def __init__(self, elements, nodes):
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes,)
        self.num_classes = len(self.inverse_dst_map)

    def set_embed(self, *args, **kwargs):
        pass

    def prepare_index(self, *args):
        pass


class NodeClassifier(nn.Module):
    def __init__(self, input_dims, num_classes, hidden=50):
        super().__init__()
        print('id', input_dims, 'nc', num_classes)
        self.l1 = nn.Linear(input_dims, hidden)
        self.l1_a = nn.LeakyReLU()

        self.logits = nn.Linear(hidden, num_classes)

    def forward(self, x, **kwargs):
        x = self.l1_a(self.l1(x))
        return self.logits(x)
