from tensorboard import program
from random import random

from SourceCodeTools.models.training_config import get_config, save_config, load_config
from SourceCodeTools.code.data.dataset.Dataset import SourceGraphDataset, filter_dst_by_freq
from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure, SamplingMultitaskTrainer
from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import NodeClassifierObjective
from SourceCodeTools.models.graph.train.objectives.SubgraphClassifierObjective import SubgraphAbstractObjective, \
    SubgraphClassifierObjective, SubgraphEmbeddingObjective
from SourceCodeTools.models.graph.train.objectives.SubgraphEmbedderObjective import SubgraphEmbeddingObjective, \
    SubgraphMatchingObjective
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from SourceCodeTools.models.graph import RGGAN
from SourceCodeTools.tabular.common import compact_property
from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.models.graph.train.objectives.SCAAClassifierObjetive import SCAAClassifierObjective


import dgl
import torch
import numpy as np
from argparse import Namespace
from torch import nn
from datetime import datetime
from os.path import join
from functools import partial

tokenizer_path = "sentencepiece_bpe.model"

data_path = "./examples/one_vs_10/with_ast"
subgraph_partition = join(data_path, "subgraph_partition.json.bz2")
filecontent_path = join(data_path, "common_filecontent.json.bz2")

config = get_config(
    subgraph_id_column="file_id",
    data_path=data_path,
    model_output_dir=data_path,
    subgraph_partition=subgraph_partition,
    tokenizer_path=tokenizer_path,
    objectives="subgraph_clf",
    measure_scores=True,
    dilate_scores=1,
    epochs=100,
)
save_config(config, "var_misuse_tiny.yaml")

dataset = SourceGraphDataset(
    **{**config["DATASET"], **config["TOKENIZER"]}
)


def load_labels():
    filecontent = unpersist(filecontent_path)
    return filecontent[["id", "user"]].rename({"id": "src", "user": "dst"}, axis=1)


load_labels()


class Trainer(SamplingMultitaskTrainer):
    def create_objectives(self, dataset, tokenizer_path):
        self.objectives = nn.ModuleList()

        self.objectives.append(
            self._create_subgraph_objective(
                objective_name="SCAAMatching",
                objective_class=SCAAClassifierObjective,
                dataset=dataset,
                tokenizer_path=tokenizer_path,
                labels_fn=load_labels,
            )
        )


# %tensorboard --logdir data

# the path of your log file.
tracking_address = './examples/one_vs_10/old_pooling'

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tracking_address])
url = tb.launch()
print(f"Tensorflow listening on {url}")

training_procedure(
    dataset,
    model_name=RGGAN,
    model_params=config["MODEL"],
    trainer_params=config["TRAINING"],
    model_base_path=get_model_base(
        config["TRAINING"], get_name(RGGAN, str(datetime.now()))),
    trainer=Trainer
)
