from random import random

from SourceCodeTools.models.training_config import get_config, save_config, load_config
from SourceCodeTools.code.data.dataset.Dataset import SourceGraphDataset, filter_dst_by_freq
from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure, SamplingMultitaskTrainer
from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import NodeClassifierObjective
from SourceCodeTools.models.graph.train.objectives.SubgraphClassifierObjective import SubgraphAbstractObjective, \
    SubgraphClassifierObjective, SubgraphEmbeddingObjective
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from SourceCodeTools.models.graph import RGGAN
from SourceCodeTools.tabular.common import compact_property
from SourceCodeTools.code.data.file_utils import unpersist

import dgl
import torch
import numpy as np
from argparse import Namespace
from torch import nn
from datetime import datetime
from os.path import join
from functools import partial


tokenizer_path = "sentencepiece_bpe.model"

data_path = "10_percent_v1"
subgraph_partition = join(data_path, "partition.json.bz2")
filecontent_path = join(data_path, "common_filecontent.json.bz2")


labels = unpersist(filecontent_path)
for part in labels['partition'].unique():
    print(part)
    print(labels.query('partition == @part')['label'].value_counts())
    print()
    
    
config = get_config(
    subgraph_id_column="file_id",
    data_path=data_path,
    model_output_dir=data_path,
    subgraph_partition=subgraph_partition,
    tokenizer_path=tokenizer_path,
    objectives="subgraph_clf",
    #use_edge_types=True,
    gpu=0,
    epochs=10,
    learning_rate=0.8,
    
    train_frac=0.8,
    random_seed=42, 
    
    # model parameters
    elem_emb_size=150,
    node_emb_size=150,                  # *** dimensionality of node embeddings
    h_dim=150,                           # *** should match to node dimensionality
    n_layers=3,
    dropout=0.1,
    activation="relu"
)

save_config(config, "max_pooling_main_var_misuse_subgraph.yaml")

dataset = SourceGraphDataset(
    **{**config["DATASET"], **config["TOKENIZER"]}
)

def load_labels():
    filecontent = unpersist(filecontent_path)
    return filecontent[["id", "label"]].rename({"id": "src", "label": "dst"}, axis=1)

from SourceCodeTools.models.graph.train.objectives.SubgraphClassifierObjective import SubgraphAbstractObjective, \
    SubgraphClassifierObjective, SubgraphEmbeddingObjective, SubgraphClassifierObjectiveWithMaxPooling

class Trainer(SamplingMultitaskTrainer):
    def create_objectives(self, dataset, tokenizer_path):
        self.objectives = nn.ModuleList()
        
        self.objectives.append(
            self._create_subgraph_objective(
                objective_name="VariableMisuseClfMaxPooling",
                objective_class=SubgraphClassifierObjectiveWithMaxPooling,
                dataset=dataset,
                tokenizer_path=tokenizer_path,
                labels_fn=load_labels,
            )
        )
        
        
training_procedure(
    dataset, 
    model_name=RGGAN, 
    model_params=config["MODEL"],
    trainer_params=config["TRAINING"],
    model_base_path=get_model_base(config["TRAINING"], get_name(RGGAN, str(datetime.now()))),
    trainer=Trainer
)