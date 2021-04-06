from collections import OrderedDict
from itertools import chain

import torch
from sklearn.metrics import ndcg_score
from torch import nn
from torch.nn import CrossEntropyLoss

from SourceCodeTools.code.data.sourcetrail import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective, _compute_accuracy
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.train.Scorer import Scorer

import numpy as np


class NodeNameClassifier(AbstractObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type=None, masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super().__init__(
            "NodeNameClassifier", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = ClassifierTargetMapper(
            elements=data_loading_func(), nodes=nodes
        )

    def create_link_predictor(self):
        self.classifier = NodeClassifier(self.target_emb_size, self.target_embedder.num_classes).to(self.device)

    def compute_acc_loss(self, graph_emb, labels, return_logits=False):
        logits = self.classifier(graph_emb)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1))

        acc = _compute_accuracy(logits.argmax(dim=1), labels)

        if return_logits:
            return acc, loss, logits
        return acc, loss

    def forward(self, input_nodes, seeds, blocks, train_embeddings=True):
        masked = self.masker.get_mask(self.seeds_to_python(seeds)) if self.masker is not None else None
        graph_emb = self._logits_batch(input_nodes, blocks, train_embeddings, masked=masked)
        indices = self.seeds_to_global(seeds).tolist()
        labels = torch.LongTensor(self.target_embedder[indices]).to(self.device)
        acc, loss = self.compute_acc_loss(graph_emb, labels)

        return loss, acc

    def evaluate_generation(self, data_split):
        total_loss = 0
        total_acc = 0
        ndcg_at = [1, 3, 5, 10]
        total_ndcg = {f"ndcg@{k}": 0. for k in ndcg_at}
        ndcg_count = 0
        count = 0

        for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
            blocks = [blk.to(self.device) for blk in blocks]

            src_embs = self._logits_batch(input_nodes, blocks)
            indices = self.seeds_to_global(seeds).tolist()
            labels = self.target_embedder[indices]
            labels = torch.LongTensor(labels).to(self.device)
            acc, loss, logits = self.compute_acc_loss(src_embs, labels, return_logits=True)

            y_pred = nn.functional.softmax(logits, dim=-1).to("cpu").numpy()
            y_true = np.zeros(y_pred.shape)
            y_true[np.arange(0, y_true.shape[0]), labels.to("cpu").numpy()] = 1.

            if self.measure_ndcg:
                if count % self.dilate_ndcg == 0:
                    for k in ndcg_at:
                        total_ndcg[f"ndcg@{k}"] += ndcg_score(y_true, y_pred, k=k)
                    ndcg_count += 1

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count, {key: val / ndcg_count for key, val in total_ndcg.items()} if self.measure_ndcg else None

    def evaluate(self, data_split, neg_sampling_factor=1):
        loss, acc, bleu = self.evaluate_generation(data_split)
        return loss, acc, bleu

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.classifier.state_dict().items():
            state_dict[f"target_embedder.{k}"] = v
        return state_dict


class ClassifierTargetMapper(ElementEmbedderBase, Scorer):
    def __init__(self, elements, nodes):
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes,)
        self.num_classes = len(self.inverse_dst_map)

    def set_embed(self, *args, **kwargs):
        pass

    def prepare_index(self):
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