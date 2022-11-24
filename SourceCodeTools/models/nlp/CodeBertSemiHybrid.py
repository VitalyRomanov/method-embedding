import numpy as np
import torch
from SourceCodeTools.mltools.torch import to_numpy

import torch.nn as nn


class CodeBertSemiHybridModel(nn.Module):
    def __init__(
            self, codebert_model, graph_emb, graph_padding_idx, num_classes, dense_hidden=100, dropout=0.1,
            bert_emb_size=768, no_graph=False
    ):
        super(CodeBertSemiHybridModel, self).__init__()

        self.codebert_model = codebert_model
        self.use_graph = not no_graph

        if self.use_graph:
            num_emb = graph_padding_idx + 1  # padding id is usually not a real embedding
            assert graph_emb is not None, "Graph embeddings are not provided, but model requires them"
            graph_emb_dim = graph_emb.e.shape[1]
            self.graph_emb = nn.Embedding(
                num_embeddings=num_emb, embedding_dim=graph_emb_dim, padding_idx=graph_padding_idx
            )

            pretrained_embeddings = torch.from_numpy(
                np.concatenate([graph_emb.e, np.zeros((1, graph_emb_dim))], axis=0)
            ).float()
            new_param = torch.nn.Parameter(pretrained_embeddings)
            assert self.graph_emb.weight.shape == new_param.shape
            self.graph_emb.weight = new_param
            self.graph_emb.weight.requires_grad = False
        else:
            graph_emb_dim = 0

        self.fc1 = nn.Linear(
            bert_emb_size + (graph_emb_dim if self.use_graph else 0),
            dense_hidden
        )
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden, num_classes)

        self.loss_f = nn.CrossEntropyLoss(reduction="mean")

    def get_tensors_for_saliency(self, token_ids, graph_ids, mask):
        token_embs = self.codebert_model.embeddings.word_embeddings(token_ids)
        x = self.codebert_model(input_embeds=token_embs, attention_mask=mask).last_hidden_state

        if self.use_graph:
            graph_emb = self.graph_emb(graph_ids)
            x = torch.cat([x, graph_emb], dim=-1)

        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x)

        if self.use_graph:
            return token_embs, graph_emb, logits
        else:
            return token_embs, logits

    def forward(self, token_ids, graph_ids, mask, finetune=False):
        if finetune:
            x = self.codebert_model(input_ids=token_ids, attention_mask=mask).last_hidden_state
        else:
            with torch.no_grad():
                x = self.codebert_model(input_ids=token_ids, attention_mask=mask).last_hidden_state

        if self.use_graph:
            graph_emb = self.graph_emb(graph_ids)
            x = torch.cat([x, graph_emb], dim=-1)

        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x

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