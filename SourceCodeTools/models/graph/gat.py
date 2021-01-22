"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv
# from graphtools import Embedder
from SourceCodeTools.models.Embedder import Embedder

# import torch
# torch.manual_seed(42)

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # node embeddings
        # embed_dict = {ntype: nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), in_dim))
        #               for ntype in g.ntypes}
        self.embed = nn.Parameter(torch.Tensor(g.number_of_nodes(), in_dim))
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

        # self.norm = nn.BatchNorm1d(num_classes)

        self.emb_size = num_classes

    def forward(self, inputs=None):
        h = self.embed
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

    def get_layers(self):
        """
        Retrieve tensor values on the layers for further use as node embeddings.
        :return:
        """
        h = self.embed
        l_out = [h.detach().numpy()]
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
            l_out.append(h.detach().numpy())
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        l_out.append(logits.detach().numpy())

        return l_out

    def get_embeddings(self, id_maps):
        return [Embedder(id_maps, e) for e in self.get_layers()]

