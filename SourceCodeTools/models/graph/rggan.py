import torch
from torch.utils import checkpoint

from SourceCodeTools.models.graph.rgcn import RGCN, RelGraphConvLayer, CkptGATConv, get_tensor_metrics, write_metrics

import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from SourceCodeTools.nlp import token_hasher


# class AttentiveAggregator(nn.Module):
#     def __init__(self, emb_dim, num_dst_embeddings=3000, dropout=0., use_checkpoint=False):
#         super(AttentiveAggregator, self).__init__()
#         # number of nodes is too much for attention of this style
#         self.att = nn.MultiheadAttention(emb_dim, num_heads=4, dropout=dropout)
#         self.query_emb = nn.Embedding(num_dst_embeddings, emb_dim)
#         self.num_query_buckets = num_dst_embeddings
#         self.use_checkpoint = use_checkpoint
#         # self.proj = nn.Linear(emb_dim, emb_dim)
#
#         self.dummy_tensor = th.ones(1, dtype=th.float32, requires_grad=True)
#
#     def do_stuff(self, query, key, value, dummy=None):
#         att_out, att_w = self.att(query, key, value)
#         return att_out, att_w
#
#     # def custom(self):
#     #     def custom_forward(*inputs):
#     #         query, key, value = inputs
#     #         return self.do_stuff(query, key, value)
#     #     return custom_forward
#
#     def forward(self, list_inputs, dsttype):  # pylint: disable=unused-argument
#         if len(list_inputs) == 1:
#             return list_inputs[0]
#         device = self.att.in_proj_bias.device
#
#         key = value = th.stack(list_inputs)#.squeeze(dim=1)
#         # key = self.proj(value)
#         query = self.query_emb(
#             th.LongTensor([token_hasher(dsttype, self.num_query_buckets)]).to(device)
#         ).unsqueeze(0).repeat(1, key.shape[1], 1)
#         if self.use_checkpoint:
#             att_out, att_w = checkpoint.checkpoint(self.do_stuff, query, key, value, self.dummy_tensor)
#         else:
#             att_out, att_w = self.do_stuff(query, key, value)
#         # att_out, att_w = self.att(query, key, value)
#         # return att_out.mean(0)#.unsqueeze(1)
#         return att_out.mean(0).unsqueeze(1)


class RGANLayer(RelGraphConvLayer):
    def __init__(
            self, in_feat, out_feat, rel_names, ntype_names, num_bases, *,
            weight=True, bias=True, activation=None, self_loop=False, dropout=0.0,
            use_gcn_checkpoint=False, use_att_checkpoint=False
    ):
        self.use_att_checkpoint = use_att_checkpoint
        super(RGANLayer, self).__init__(
            in_feat, out_feat, rel_names, ntype_names, num_bases,
            weight=weight, bias=bias, activation=activation, self_loop=self_loop, dropout=dropout,
            use_gcn_checkpoint=use_gcn_checkpoint
        )

    def create_conv(self, in_feat, out_feat, rel_names):
        # self.attentive_aggregator = AttentiveAggregator(out_feat, use_checkpoint=self.use_att_checkpoint)
        # self.conv = dglnn.HeteroGraphConv({
        #     rel: dglnn.GraphConv(
        #         in_feat, out_feat, norm='right', weight=False, bias=True, allow_zero_in_degree=True,
        #         activation=self.activation
        #     )
        #     for rel in rel_names
        # }, aggregate=self.attentive_aggregator)
        self.conv = dglnn.HeteroGraphConv({
            rel: CkptGATConv(
                (in_feat, in_feat), out_feat, num_heads=1, activation=self.activation,
                use_checkpoint=self.use_gcn_checkpoint
            )
            for rel in rel_names
        }, aggregate="mean")

    def do_convolution(self, g, inputs_src, wdict):
        hs = self.conv(g, inputs_src, mod_kwargs=wdict)
        hs = {key: val.reshape(val.size(0), -1) for key, val in hs.items()}
        return hs


class RGAN(RGCN):
    def __init__(
            self, ntypes, etypes, h_dim, node_emb_size, num_bases, n_layers=1, dropout=0, use_self_loop=False,
            activation=F.relu, use_gcn_checkpoint=False, use_att_checkpoint=False, **kwargs
    ):
        self.use_att_checkpoint = use_att_checkpoint

        super(RGAN, self).__init__(
            ntypes, etypes, h_dim, node_emb_size, num_bases, n_layers=n_layers, dropout=dropout,
            use_self_loop=use_self_loop, activation=activation, use_gcn_checkpoint=use_gcn_checkpoint, **kwargs
        )

    def _initialize(self):
        self.rel_names = list(set(self.etypes))
        self.rel_names.sort()
        self.ntype_names = list(set(self.ntypes))
        self.ntype_names.sort()
        if self.num_bases < 0 or self.num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = self.num_bases
        self.dropout = self.dropout
        self.use_self_loop = self.use_self_loop

        # self.layers = nn.ModuleList()
        # for i in range(self.n_layers):
        #     self.layers.append(RGANLayer(
        #         self.h_dim, self.h_dim, self.rel_names, self.ntype_names,
        #         self.num_bases, activation=self.activation, self_loop=self.use_self_loop,
        #         dropout=self.dropout, weight=False, use_gcn_checkpoint=self.use_gcn_checkpoint,
        #     use_att_checkpoint=self.use_att_checkpoint))

        self.layer = RGANLayer(
            self.h_dim, self.h_dim, self.rel_names, self.ntype_names,
            self.num_bases, activation=self.activation, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False, use_gcn_checkpoint=self.use_gcn_checkpoint,
            use_att_checkpoint=self.use_att_checkpoint
        )
        self.layers = [self.layer] * self.n_layers
        self.layer_norm = nn.ModuleList([nn.LayerNorm([self.h_dim]) for _ in range(self.num_layers)])


# class OneStepGRU(nn.Module):
#     """A gated recurrent unit (GRU) cell
#
#     .. math::
#
#         \begin{array}{ll}
#         r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
#         z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
#         n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
#         h' = (1 - z) * n + z * h
#         \end{array}
#
#     where :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product."""
#     def __init__(self, dim, use_checkpoint=False):
#         super(OneStepGRU, self).__init__()
#         self.gru_rx = nn.Linear(dim, dim)
#         self.gru_rh = nn.Linear(dim, dim)
#         self.gru_zx = nn.Linear(dim, dim)
#         self.gru_zh = nn.Linear(dim, dim)
#         self.gru_nx = nn.Linear(dim, dim)
#         self.gru_nh = nn.Linear(dim, dim)
#         self.act_r = nn.Sigmoid()
#         self.act_z = nn.Sigmoid()
#         self.act_n = nn.Tanh()
#         self.use_checkpoint = use_checkpoint
#         self.dummy_tensor = th.ones(1, dtype=th.float32, requires_grad=True)
#
#     def do_stuff(self, x, h, dummy_tensor=None):
#         # x = x.unsqueeze(1)
#         r = self.act_r(self.gru_rx(x) + self.gru_rh(h))
#         z = self.act_z(self.gru_zx(x) + self.gru_zh(h))
#         n = self.act_n(self.gru_nx(x) + self.gru_nh(r * h))
#         return (1 - z) * n + z * h
#
#     def forward(self, x, h):
#         if self.use_checkpoint:
#             h = checkpoint.checkpoint(self.do_stuff, x, h, self.dummy_tensor)
#         else:
#             h = self.do_stuff(x, h)
#         return h


class RGGANLayer(RGANLayer):
    def __init__(
            self, in_feat, out_feat, rel_names, ntype_names, num_bases, *,
            weight=True, bias=True, activation=None, self_loop=False,
            dropout=0.0, use_gcn_checkpoint=False, use_att_checkpoint=False, use_gru_checkpoint=False
    ):
        super(RGGANLayer, self).__init__(
            in_feat, out_feat, rel_names, ntype_names, num_bases, weight=weight, bias=bias, activation=activation,
            self_loop=self_loop, dropout=dropout, use_gcn_checkpoint=use_gcn_checkpoint,
            use_att_checkpoint=use_att_checkpoint
        )
        self.gru = nn.GRUCell(input_size=out_feat, hidden_size=out_feat, bias=True)

    def forward(self, g, inputs, layer_id=None, tensor_metrics=None):
        if g.is_block:
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_dst = inputs

        hs = super().forward(g, inputs, layer_id=layer_id, tensor_metrics=tensor_metrics)

        out = {}
        for dsttype, h_dst in hs.items():
            gru_out = self.gru(
                h_dst, inputs_dst[dsttype]
            )
            out[dsttype] = gru_out
            if tensor_metrics is not None:
                write_metrics(tensor_metrics, gru_out, f"layer_{layer_id}/{dsttype}/norm")

        return out

        # return {
        #     dsttype: self.gru(
        #         h_dst, inputs_dst[dsttype]
        #     )
        # }


class RGGAN(RGAN):
    def __init__(
            self, ntypes, etypes, h_dim, node_emb_size, num_bases, n_layers=1, dropout=0, use_self_loop=False,
            activation=F.relu, use_gcn_checkpoint=False, use_att_checkpoint=False, use_gru_checkpoint=False, **kwargs
    ):
        self.use_gru_checkpoint = use_gru_checkpoint

        super(RGGAN, self).__init__(
            ntypes, etypes, h_dim, node_emb_size, num_bases, n_layers=n_layers, dropout=dropout,
            use_self_loop=use_self_loop, activation=activation, use_gcn_checkpoint=use_gcn_checkpoint,
            use_att_checkpoint=use_att_checkpoint, **kwargs
        )

    def _initialize(self):
        assert self.h_dim == self.out_dim, f"Parameter h_dim and num_classes should be equal in {self.__class__.__name__}"

        self.rel_names = list(set(self.etypes))
        self.ntype_names = list(set(self.ntypes))
        self.rel_names.sort()
        self.ntype_names.sort()
        if self.num_bases < 0 or self.num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = self.num_bases

        self.dropout = self.dropout
        self.use_self_loop = self.use_self_loop

        self.layer = RGGANLayer(
            self.h_dim, self.h_dim, self.rel_names, self.ntype_names,
            self.num_bases, activation=self.activation, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False, use_gcn_checkpoint=self.use_gcn_checkpoint, # : )
            use_att_checkpoint=self.use_att_checkpoint, use_gru_checkpoint=self.use_gru_checkpoint
        )

        self.layers = [self.layer] * self.n_layers
        self.layer_norm = nn.ModuleList([nn.LayerNorm([self.h_dim]) for _ in range(self.num_layers)])
