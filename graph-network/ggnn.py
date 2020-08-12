import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GatedGraphConv
from Dataset import compact_property
from Embedder import Embedder

class GGNN(nn.Module):
    def __init__(self, g, n_steps,
                 in_dim,
                 num_hidden,
                 num_classes, **kwargs):
        super(GGNN, self).__init__()
        self.etypes = g.edata['etypes']
        # self.g = dgl.to_homo(g)
        self.g = g

        self.embed = nn.Parameter(torch.Tensor(self.g.number_of_nodes(), in_dim))

        self.emb_size = num_classes

        self.ggnn = GatedGraphConv(in_feats=in_dim,
                                   out_feats=num_classes,
                                   n_steps=n_steps,
                                   n_etypes=len(compact_property(self.etypes)))

    def forward(self, inputs=None):
        return self.ggnn(self.g, self.embed, self.etypes)


    def get_layers(self):
        """
        Retrieve tensor values on the layers for further use as node embeddings.
        :return:
        """
        h = self.embed
        l_out = [h.detach().numpy()]
        h = self.ggnn(self.g, self.embed, self.etypes)
        l_out.append(h.detach().numpy())

        return l_out

    def get_embeddings(self, id_maps):
        return [Embedder(id_maps, e) for e in self.get_layers()]