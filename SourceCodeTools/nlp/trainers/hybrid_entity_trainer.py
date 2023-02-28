import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from transformers import RobertaModel

from SourceCodeTools.models.graph import RGCN
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
            ["node_"], ["edge_"], **self._graph_config["MODEL"]
        )
        model = self.model(
            codebert_model, graph_model, graph_emb=kwargs["graph_embedder"],
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
                graph_embs=batch['graph_embs'] if 'graph_embs' in batch else None,
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
