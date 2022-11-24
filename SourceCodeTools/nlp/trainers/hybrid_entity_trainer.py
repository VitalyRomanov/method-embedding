import os
import pickle
from datetime import datetime
from pathlib import Path

from transformers import RobertaModel

from SourceCodeTools.models.graph import RGCN
from SourceCodeTools.models.training_config import get_config
from SourceCodeTools.nlp.batchers.HybridBatcher import HybridBatcher
from SourceCodeTools.nlp.trainers.codebert_entity_trainer import CodeBertModelTrainer


class HybridModelTrainer(CodeBertModelTrainer):
    def __init__(self, *args, **kwargs):
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

    def get_model(self, *args, **kwargs):
        codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")

        graph_model_params = get_config()
        graph_model = RGCN(
            ["node_"], ["edge_"], **graph_model_params
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
