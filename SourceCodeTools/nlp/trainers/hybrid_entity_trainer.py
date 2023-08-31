import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dgl
import torch
from tqdm import tqdm
from transformers import RobertaModel

from SourceCodeTools.mltools.torch import get_length_mask
from SourceCodeTools.models.graph import RGCN
from SourceCodeTools.models.graph.NodeEmbedder import SimplestNodeEmbedder
from SourceCodeTools.nlp.batchers.HybridBatcher import HybridBatcher
from SourceCodeTools.nlp.trainers.codebert_entity_trainer import CodeBertModelTrainer


class HybridModelTrainer(CodeBertModelTrainer):
    def __init__(self, *args, **kwargs):
        self._graph_config = kwargs.pop("graph_config")
        super().__init__(*args, **kwargs)

    def set_batcher_class(self):
        self.batcher = HybridBatcher

    def set_model_class(self):
        from SourceCodeTools.models.nlp.CodeBertHybridModel import CodeBertHybridModel
        self.model = CodeBertHybridModel

    def get_training_dir(self):
        if not hasattr(self, "_timestamp"):
            self._timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_")
        return Path(self.trainer_params["model_output"]).joinpath("HybridModel_" + self._timestamp)

    def get_batcher(self, *args, **kwargs):
        kwargs.update({"tokenizer": "codebert"})
        kwargs["graph_config"] = self._graph_config
        return self.batcher(*args, **kwargs)

    def get_model(self, *args, **kwargs):
        codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")

        graph_model = RGCN(
            ["_N"], ["_E"], **self._graph_config["MODEL"]
        )
        node_embedder = SimplestNodeEmbedder(
            emb_size=self._graph_config["MODEL"]["node_emb_size"],
            dtype=torch.float64,
            n_buckets=self._graph_config["TRAINING"]["embedding_table_size"],
            sparse=False
        )

        model = self.model(
            codebert_model, graph_model, node_embedder,
            graph_padding_idx=kwargs["graph_padding_idx"],
            num_classes=kwargs["num_classes"],
            no_graph=self.no_graph
        )
        if self.use_cuda:
            model.cuda()

        if self.ckpt_path is not None:
            ckpt_path = os.path.join(self.ckpt_path, "checkpoint")
            model = self.load_checkpoint(model, ckpt_path)
        return model

    def iterate_batches(self, model, batches, epoch, num_train_batches, train_scores, scorer, train=True):
        scores_for_averaging = defaultdict(list)

        batch_count = 0

        if train is True:
            self.set_model_training(model)
        else:
            self.set_model_evaluation(model)

        for ind, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch}")):
            self._format_batch(batch, self.device)
            scores = self.make_step(
                model=model, optimizer=self.optimizer, token_ids=batch['tok_ids'],
                prefix=batch['prefix'], suffix=batch['suffix'],
                graph_ids=batch['graph_ids'] if 'graph_ids' in batch else None,
                graph=batch['graph'] if 'graph' in batch else None,
                labels=batch['tags'], lengths=batch['lens'],
                extra_mask=batch['no_loc_mask'] if self.no_localization else batch['hide_mask'],
                # class_weights=batch['class_weights'],
                scorer=scorer, finetune=self.finetune and (epoch >= self.pretraining_epochs),
                vocab_mapping=self.vocab_mapping,
                train=train
            )

            batch_count += 1

            scores["batch_size"] = batch['tok_ids'].shape[0]
            for score, value in scores.items():
                self._write_to_summary(f"{score}/{'Train' if train else 'Test'}", value,
                                       epoch * num_train_batches + ind)
                scores_for_averaging[score].append(value)
            train_scores.append(scores_for_averaging)

        return num_train_batches

    @classmethod
    def _format_batch(cls, batch, device):
        key_types = {
            'tok_ids': torch.LongTensor,
            'tags': torch.LongTensor,
            'hide_mask': torch.BoolTensor,
            'no_loc_mask': torch.BoolTensor,
            'lens': torch.LongTensor,
            'graph_ids': torch.LongTensor,
            'graph_embs': torch.FloatTensor,
            'graph_mask': torch.BoolTensor,
            'graph': None
        }
        for key, tf in key_types.items():
            if key in batch:
                if key == "graph":
                    batch[key] = dgl.from_networkx(
                        batch[key],
                        node_attrs=["original_id", "embedding_id"],
                        edge_attrs=["original_id"]
                    )
                    # batch[key] = dgl.to_heterogeneous(batch[key], ntypes=["node_"], etypes=["edge_"])
                else:
                    batch[key] = tf(batch[key]).to(device)

    @classmethod
    def make_step(
            cls, model, optimizer, token_ids, prefix, suffix, graph_ids, labels, lengths, graph=None,
            graph_mask=None,
            extra_mask=None, class_weights=None, scorer=None, finetune=False, vocab_mapping=None, train=False
    ):

        token_ids[token_ids == 50265] = 1

        with torch.set_grad_enabled(train):
            model_output, scores = cls.compute_loss_and_scores(
                model, token_ids, prefix, suffix, graph_ids, labels, lengths, graph=graph,
                graph_mask=graph_mask,
                extra_mask=extra_mask, class_weights=class_weights, scorer=scorer, finetune=finetune,
                vocab_mapping=vocab_mapping, training=train
            )

        if train is True:
            optimizer.zero_grad()
            model_output.loss.backward()
            optimizer.step()

        return scores

    @classmethod
    def compute_loss_and_scores(
            cls, model, token_ids, prefix, suffix, graph_ids, labels, lengths, graph=None, graph_mask=None,
            extra_mask=None, class_weights=None, scorer=None, finetune=False, vocab_mapping=None,
            training=False
    ):
        token_ids[token_ids == len(vocab_mapping)] = vocab_mapping["<unk>"]
        seq_mask = get_length_mask(token_ids, lengths)
        if graph_mask is not None:
            graph_mask = graph_mask * seq_mask
        else:
            graph_mask = seq_mask
        model_output = model(
            token_ids, graph_ids, graph=graph, mask=seq_mask, graph_mask=graph_mask,
            finetune=finetune
        )
        loss = model.loss(model_output.logits, labels, mask=seq_mask, class_weights=class_weights,
                          extra_mask=extra_mask)
        if scorer is not None:
            scores = model.score(model_output.logits, labels, mask=seq_mask, scorer=scorer, extra_mask=extra_mask)
        else:
            scores = {}

        scores["loss"] = loss.cpu().item()
        model_output.loss = loss
        return model_output, scores
