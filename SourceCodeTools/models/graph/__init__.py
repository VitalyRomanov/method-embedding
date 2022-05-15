# import torch
# import torch.nn as nn
# import dgl.function as fn
# from dgl.nn.pytorch import edge_softmax, GATConv
# import numpy as np

# https://docs.dgl.ai/tutorials/hetero/1_basics.html#working-with-heterogeneous-graphs-in-dgl

from SourceCodeTools.models.graph.gat import GAT
from SourceCodeTools.models.graph.rgcn_hetero import RGCN
from SourceCodeTools.models.graph.rgcn_sampling import RGCNSampling
# from SourceCodeTools.models.graph.ggnn import GGNN
from SourceCodeTools.models.graph.rggan import RGAN, RGGAN