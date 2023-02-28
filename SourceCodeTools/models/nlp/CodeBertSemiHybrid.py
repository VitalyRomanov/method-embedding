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
    text_encoder_output: torch.Tensor
    logits: torch.Tensor
    prediction: torch.Tensor
    loss: Optional[torch.Tensor]
    scores: Optional[Dict]


class CodeBertSemiHybridModel(nn.Module):
    def __init__(
            self, codebert_model, graph_emb, graph_padding_idx, num_classes, dense_hidden=100, dropout=0.1,
            bert_emb_size=768, no_graph=False
    ):
        super(CodeBertSemiHybridModel, self).__init__()

        self.codebert_model = codebert_model
        self.buffered_toke_type_ids = self.codebert_model.embeddings.token_type_ids.tile(1, 2)
        assert self.buffered_toke_type_ids.sum().item() == 0
        self.use_graph = not no_graph

        if self.use_graph:
            num_emb = (graph_padding_idx if isinstance(graph_padding_idx, int) else 1) + 1  # padding id is usually not a real embedding
            assert graph_emb is not None, "Graph embeddings are not provided, but model requires them"
            graph_emb_dim = graph_emb.n_dims
            self.graph_emb = nn.Embedding(
                num_embeddings=num_emb, embedding_dim=graph_emb_dim, padding_idx=graph_padding_idx if isinstance(graph_padding_idx, int) else 1
            )

            pretrained_embeddings = torch.from_numpy(
                np.concatenate([graph_emb.get_embedding_table(), np.zeros((1, graph_emb_dim))], axis=0)
            ).float()
            new_param = torch.nn.Parameter(pretrained_embeddings)
            assert self.graph_emb.weight.shape == new_param.shape
            self.graph_emb.weight = new_param
            # self.graph_emb.weight.requires_grad = False
            # logging.warning("Graph embeddings are not finetuned")
            self.graph_adapter = nn.Linear(pretrained_embeddings.shape[1], bert_emb_size, bias=False)
        else:
            graph_emb_dim = 0

        self.fc1 = nn.Linear(
            bert_emb_size,  # + (graph_emb_dim if self.use_graph else 0),
            dense_hidden
        )
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden, num_classes)

        self.loss_f = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, token_ids, graph_ids, mask, finetune=False, graph_embs=None):
        with torch.set_grad_enabled(finetune):
            token_embs_ = self.codebert_model.embeddings.word_embeddings(token_ids)
            position_ids = None
            if self.use_graph:
                graph_emb = self.graph_emb(graph_ids)
                position_ids = torch.arange(2, token_embs_.shape[1] + 2).reshape(1, -1).to(token_ids.device)
                position_ids = torch.cat([position_ids, position_ids], dim=1)
                token_embs = torch.cat([token_embs_, self.graph_adapter(graph_emb)], dim=1)
                mask = torch.cat([mask, mask], dim=1)
            else:
                token_embs = token_embs_
                graph_emb = None

            token_type_ids = self.buffered_toke_type_ids[:, :token_embs.size(1)].to(token_ids.device)
            codebert_output = self.codebert_model(
                inputs_embeds=token_embs,
                attention_mask=mask,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                output_attentions=True
            )

        x = codebert_output.last_hidden_state
        x = torch.relu(self.fc1(x[:, :token_ids.size(1), :]))
        x = self.drop(x)
        x = self.fc2(x)

        return HybridModelOutput(
            token_embs=token_embs_,
            graph_embs=graph_emb,
            text_encoder_output=codebert_output,
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