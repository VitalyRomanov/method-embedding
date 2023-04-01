import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from torch import ones, float32, Tensor, split, matmul, no_grad, zeros, arange
from torch.utils import checkpoint


class CkptGATConv(dglnn.GATConv):
    def __init__(
            self, in_feats, out_feats, num_heads, feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False,
            activation=None, allow_zero_in_degree=False, use_checkpoint=False
    ):
        super(CkptGATConv, self).__init__(
            in_feats, out_feats, num_heads,
            feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope,
            residual=residual, activation=activation, allow_zero_in_degree=allow_zero_in_degree
        )
        self.dummy_tensor = ones(1, dtype=float32, requires_grad=True)
        self.use_checkpoint = use_checkpoint

    def custom(self, graph):
        def custom_forward(*inputs):
            feat0, feat1, dummy = inputs
            return super(CkptGATConv, self).forward(graph, (feat0, feat1))
        return custom_forward

    def forward(self, graph, feat):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self.custom(graph), feat[0], feat[1], self.dummy_tensor)  # .squeeze(1)
        else:
            return super(CkptGATConv, self).forward(graph, feat)  # .squeeze(1)


def get_tensor_metrics(tensor, path):
    output_metrics = {}
    with torch.no_grad():
        output_metrics[f"{path}/mean"] = tensor.mean()
        output_metrics[f"{path}/max"] = tensor.max()
        output_metrics[f"{path}/quantile_0.75"] = torch.quantile(tensor, 0.75)
        output_metrics[f"{path}/quantile_0.5"] = torch.quantile(tensor, 0.5)

    return output_metrics


def write_metrics(tensor_metrics, tensor, metric_key):
    try:
        tensor_metrics.update(get_tensor_metrics(tensor, metric_key))
    except RuntimeError as e:
        # logging.warning(f"Encountered error when computing metrics: {e}")
        pass


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(
            self, in_feat, out_feat, rel_names, ntype_names, num_bases, *, weight=True, bias=True, activation=None,
            self_loop=False, dropout=0.0, use_gcn_checkpoint=False, **kwargs
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.ntype_names = ntype_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.use_gcn_checkpoint = use_gcn_checkpoint
        self.glu = nn.GLU()

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(Tensor(len(self.rel_names), in_feat, out_feat))
                # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(self.weight)

        self.create_conv(in_feat, out_feat, rel_names)

        # bias
        if bias:
            self.bias_dict = nn.ParameterDict()
            for ntype_name in self.ntype_names:
                self.bias_dict[ntype_name] = nn.Parameter(Tensor(1, out_feat))
                nn.init.normal_(self.bias_dict[ntype_name])
            # self.h_bias = nn.Parameter(th.Tensor(1, out_feat))
            # nn.init.normal_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(Tensor(in_feat, out_feat))
            # nn.init.xavier_uniform_(self.loop_weight,
            #                         gain=nn.init.calculate_gain('tanh'))
            nn.init.xavier_normal_(self.loop_weight)

        self.dropout = nn.Dropout(dropout)

    def create_conv(self, in_feat, out_feat, rel_names):
        # rel : dglnn.GATConv(in_feat, out_feat, num_heads=4)
        # rel : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False, allow_zero_in_degree=True)
        # self.conv = dglnn.HeteroGraphConv({
        #     rel: CkptGATConv((in_feat, in_feat), out_feat, num_heads=1, use_checkpoint=self.use_gcn_checkpoint)
        #     for rel in rel_names
        # }, aggregate="mean")
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(
                in_feat, out_feat, norm='right', weight=False, bias=True, allow_zero_in_degree=True,
                activation=self.activation
            )
            for rel in rel_names
        }, aggregate="mean")

    def do_convolution(self, g, inputs_src, wdict):
        hs = self.conv(g, inputs_src, mod_kwargs=wdict)
        return hs

    def forward(self, g, inputs, layer_id=None, tensor_metrics=None):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(split(weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            # the beginning of src and dst indexes match, that is why we can simply slice the first
            # nodes to get dst embeddings
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.do_convolution(g, inputs_src, wdict)

        def _apply(ntype, h):
            h = self.glu(h.tile(1, 2))
            if self.self_loop:
                h = h + matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.bias_dict[ntype]
                # h = h + self.h_bias
            if tensor_metrics is not None:
                write_metrics(tensor_metrics, h, f"layer_{layer_id}/{ntype}/logits")
            if self.activation:
                h = self.activation(h)
            if tensor_metrics is not None:
                write_metrics(tensor_metrics, h, f"layer_{layer_id}/{ntype}/activation")
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class RGCN(nn.Module):
    def __init__(
            self, ntypes, etypes, h_dim, node_emb_size, num_bases, n_layers=1, dropout=0, use_self_loop=False,
            activation=F.relu, use_gcn_checkpoint=False, collect_tensor_metrics=False, **kwargs
    ):
        super(RGCN, self).__init__()
        self.ntypes = list(set(ntypes))
        self.ntypes.sort()
        self.etypes = list(set(etypes))
        self.etypes.sort()
        self.h_dim = h_dim
        self.out_dim = node_emb_size
        self.activation = activation
        self.num_bases = num_bases
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_gcn_checkpoint = use_gcn_checkpoint
        self.tensor_metrics = {} if collect_tensor_metrics else None

        self._initialize()

    def _initialize(self):
        if self.num_bases < 0 or self.num_bases > len(self.etypes):
            self.num_bases = len(self.etypes)
        else:
            self.num_bases = self.num_bases
        self.dropout = self.dropout
        self.use_self_loop = self.use_self_loop

        # self.layers = nn.ModuleList()
        # self.layer_norm = nn.ModuleList()
        # for i in range(self.n_layers):
        #     self.layers.append(RelGraphConvLayer(
        #         self.h_dim, self.h_dim, self.etypes, self.ntypes,
        #         self.num_bases, activation=self.activation, self_loop=self.use_self_loop,
        #         dropout=self.dropout, weight=False,
        #         use_gcn_checkpoint=self.use_gcn_checkpoint))  # changed weight for GATConv
        #     self.layer_norm.append(nn.LayerNorm([self.h_dim]))

        self.layer = RelGraphConvLayer(
            self.h_dim, self.h_dim, self.etypes, self.ntypes,
            self.num_bases, activation=self.activation, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=True,
            use_gcn_checkpoint=self.use_gcn_checkpoint
        )
        self.layers = [self.layer] * self.n_layers
        self.layer_norm = nn.ModuleList([nn.LayerNorm([self.h_dim]) for _ in range(self.num_layers)])

    @property
    def emb_size(self):
        return self.out_dim

    @property
    def num_layers(self):
        return len(self.layers)

    def apply_norm_layer(self, h, layer, layer_id):
        out = {}
        for key, val in h.items():
            norm_ = layer(val)
            if self.tensor_metrics is not None:
                write_metrics(self.tensor_metrics, norm_, f"layer_{layer_id}/{key}/norm")
            out[key] = norm_
        return out
        # return {key: layer(val) for key, val in h.items()}

    def forward(self, h, blocks=None, graph=None):

        if blocks is None:
            # full graph training
            for ind, (layer, norm) in enumerate(zip(self.layers, self.layer_norm)):
                h = layer(graph, h, layer_id=ind, tensor_metrics=self.tensor_metrics)
                # h = self.apply_norm_layer(h, norm, ind)
        else:
            # minibatch training
            for ind, (layer, norm, block) in enumerate(zip(self.layers, self.layer_norm, blocks)):
                h = layer(block, h, layer_id=ind, tensor_metrics=self.tensor_metrics)
                # h = self.apply_norm_layer(h, norm, ind)
        return h
