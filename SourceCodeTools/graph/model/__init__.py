# import torch
# import torch.nn as nn
# import dgl.function as fn
# from dgl.nn.pytorch import edge_softmax, GATConv
# import numpy as np

# https://docs.dgl.ai/tutorials/hetero/1_basics.html#working-with-heterogeneous-graphs-in-dgl

from SourceCodeTools.graph.model.gat import GAT
from SourceCodeTools.graph.model.rgcn_hetero import RGCN
from SourceCodeTools.graph.model.rgcn_sampling import RGCNSampling
from SourceCodeTools.graph.model.ggnn import GGNN
from SourceCodeTools.graph.model.NodeSampler import GCNSampling, GATSampler
