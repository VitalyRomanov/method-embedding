import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import numpy as np

# https://docs.dgl.ai/tutorials/hetero/1_basics.html#working-with-heterogeneous-graphs-in-dgl

from gat import GAT
from rgcn_hetero import RGCN
from rgcn_sampling import RGCNSampling
from ggnn import GGNN
from NodeSampler import GCNSampling, GATSampler
