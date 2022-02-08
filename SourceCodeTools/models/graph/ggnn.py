import torch
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
from dgl.nn.pytorch.conv import GatedGraphConv

from SourceCodeTools.code.data.dataset.Dataset import compact_property
from SourceCodeTools.models.Embedder import Embedder


class GatedGraphConv(nn.Module):
    r"""Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
        h_{i}^{0} & = [ x_i \| \mathbf{0} ]

        a_{i}^{t} & = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} & = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    n_steps : int
        Number of recurrent steps.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 n_etypes,
                 bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes

        self.encoding = nn.Linear(in_feats, out_feats)

        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def forward(self, graph, feat, etypes):
        """Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        assert graph.is_homograph(), \
            "not a homograph; convert it with to_homo and pass in the edge type as argument"
        graph = graph.local_var()

        encoded = self.encoding(feat)
        feat = nn.functional.relu(encoded)
        # zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
        # feat = th.cat([feat, zero_pad], -1)

        for _ in range(self._n_steps):
            graph.ndata['h'] = feat
            for i in range(self._n_etypes):
                eids = (etypes == i).nonzero().view(-1)
                if len(eids) > 0:
                    graph.apply_edges(
                        lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                        eids
                    )
            graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
            a = graph.ndata.pop('a') # (N, D)
            feat = self.gru(a, feat)
        return feat



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