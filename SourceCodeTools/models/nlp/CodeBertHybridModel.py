import logging
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
from SourceCodeTools.mltools.torch import to_numpy

import torch.nn as nn


@dataclass
class HybridModelOutput:
    token_embs: torch.Tensor
    graph_embs: Optional[torch.Tensor]
    logits: torch.Tensor
    prediction: torch.Tensor
    loss: Optional[torch.Tensor]
    scores: Optional[Dict]


class CodeBertHybridModel(nn.Module):
    def __init__(
            self, codebert_model, graph_model, node_embedder, graph_padding_idx, num_classes, dense_hidden=100, dropout=0.1,
            bert_emb_size=768, no_graph=False
    ):
        super(CodeBertHybridModel, self).__init__()

        self.codebert_model = codebert_model
        self.graph_model = graph_model
        self.node_embedder = node_embedder

        graph_emb_dim = graph_model.out_dim
        self.zero_pad = torch.zeros((1, graph_emb_dim))

        self.fc1 = nn.Linear(
            bert_emb_size + graph_emb_dim,
            dense_hidden
        )
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden, num_classes)

        self.loss_f = nn.CrossEntropyLoss(reduction="mean")

    # def get_tensors_for_saliency(self, token_ids, graph_ids, mask):
    #     token_embs = self.codebert_model.embeddings.word_embeddings(token_ids)
    #     x = self.codebert_model(input_embeds=token_embs, attention_mask=mask).last_hidden_state
    #
    #     if self.use_graph:
    #         graph_emb = self.graph_emb(graph_ids)
    #         x = torch.cat([x, graph_emb], dim=-1)
    #
    #     x = torch.relu(self.fc1(x))
    #     x = self.drop(x)
    #     logits = self.fc2(x)
    #
    #     if self.use_graph:
    #         return token_embs, graph_emb, logits
    #     else:
    #         return token_embs, logits

    def forward(self, token_ids, graph_ids, graph, mask, graph_mask, finetune=False):

        if not hasattr(self, "graph_mask_warning"):
            logging.warning("`forward` is different from CodeBertSemiHybridModel. As a consequence, `graph_mask` is not used.")
            self.graph_mask_warning = True

        with torch.set_grad_enabled(finetune):
            x_ = self.codebert_model(input_ids=token_ids, attention_mask=mask).last_hidden_state

        node_emb = self.node_embedder(graph.ndata["embedding_id"], mask=None)
        graph_emb = self.graph_model({"_N": node_emb}, graph=graph)
        graph_emb_shifted = torch.cat([self.zero_pad, graph_emb["_N"]], dim=0)

        graph_emb_aligned = graph_emb_shifted[graph_ids, :]

        x = torch.cat([x_, graph_emb_aligned], dim=-1)

        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return HybridModelOutput(
            token_embs=x_,
            graph_embs=graph_emb,
            logits=x,
            prediction=x.argmax(-1),
            loss=None, scores=None
        )

    def loss(self, logits, labels, mask, class_weights=None, extra_mask=None):
        if extra_mask is not None:
            mask = torch.logical_and(mask, extra_mask)
        logits = logits[mask, :]
        labels = labels[mask]
        loss = self.loss_f(logits, labels)
        # if class_weights is None:
        #     loss = tf.reduce_mean(tf.boolean_mask(losses, seq_mask))
        # else:
        #     loss = tf.reduce_mean(tf.boolean_mask(losses * class_weights, seq_mask))

        return loss

    def score(self, logits, labels, mask, scorer=None, extra_mask=None):
        if extra_mask is not None:
            mask = torch.logical_and(mask, extra_mask)
        true_labels = labels[mask]
        argmax = logits.argmax(-1)
        estimated_labels = argmax[mask]

        p, r, f1 = scorer(to_numpy(estimated_labels), to_numpy(true_labels))

        scores = {}
        scores["Precision"] = p
        scores["Recall"] = r
        scores["F1"] = f1

        return scores