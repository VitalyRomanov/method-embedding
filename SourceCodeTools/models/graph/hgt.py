import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from SourceCodeTools.models.graph.rggan import RGGAN


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, node_types, relation_types, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_types = node_types
        self.relation_types = relation_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.a_linears = nn.ModuleDict()
        self.norms = nn.ModuleDict()
        self.skip = nn.ParameterDict()
        self.use_norm = use_norm

        for ntype in node_types:
            self.k_linears[ntype] = nn.Linear(in_dim, out_dim)
            self.q_linears[ntype] = nn.Linear(in_dim, out_dim)
            self.v_linears[ntype] = nn.Linear(in_dim, out_dim)
            self.a_linears[ntype] = nn.Linear(out_dim, out_dim)
            if use_norm:
                self.norms[ntype] = nn.LayerNorm(out_dim)
            self.skip[ntype] = nn.Parameter(torch.tensor(1.))

        self.relation_pri = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in relation_types:
            self.relation_pri[etype] = nn.Parameter(torch.ones(self.n_heads))
            self.relation_att[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            self.relation_msg[etype] = nn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
            nn.init.xavier_uniform_(self.relation_att[etype])
            nn.init.xavier_uniform_(self.relation_msg[etype])
        self.drop = nn.Dropout(dropout)

    def edge_attention(self, edges):
        etype = edges.canonical_etype[1]
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        relation_msg = self.relation_msg[etype]
        key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk
        val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0)
        return {'a': att, 'v': val}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, input):
        k_linear = self.k_linears["node_"]
        v_linear = self.v_linears["node_"]
        q_linear = self.q_linears["node_"]

        if G.is_block:
            input_src = input
            input_dst = {k: v[:G.number_of_dst_nodes(k)] for k, v in input.items()}
            G.srcnodes["node_"].data['k'] = k_linear(input_src["node_"].view(-1, self.n_heads, self.d_k))
            G.srcnodes["node_"].data['v'] = v_linear(input_src["node_"].view(-1, self.n_heads, self.d_k))
            G.dstnodes["node_"].data['q'] = q_linear(input_dst["node_"].view(-1, self.n_heads, self.d_k))
        else:
            input_src = input_dst = input
            G.srcnodes["node_"].data['k'] = k_linear(input["node_"].view(-1, self.n_heads, self.d_k))
            G.srcnodes["node_"].data['v'] = v_linear(input["node_"].view(-1, self.n_heads, self.d_k))
            G.dstnodes["node_"].data['q'] = q_linear(input["node_"].view(-1, self.n_heads, self.d_k))

        for srctype, etype, dsttype in G.canonical_etypes:
            # k_linear = self.k_linears[srctype]
            # v_linear = self.v_linears[srctype]
            # q_linear = self.q_linears[dsttype]

            assert srctype == dsttype, "Multiple node types are not supported"

            # G.nodes[srctype].data['k'] = k_linear(input[srctype].view(-1, self.n_heads, self.d_k))
            # G.nodes[srctype].data['v'] = v_linear(input[srctype].view(-1, self.n_heads, self.d_k))
            # G.nodes[dsttype].data['q'] = q_linear(input[dsttype].view(-1, self.n_heads, self.d_k))

            if G.number_of_edges(etype) == 0:
                continue

            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all(
            {
                etype: (self.message_func, self.reduce_func)
                for etype in self.relation_types if G.number_of_edges(etype) > 0
            }, cross_reducer='mean'
        )

        h = dict()
        for ntype in self.node_types:
            alpha = torch.sigmoid(self.skip[ntype])
            trans_out = self.a_linears[ntype](G.dstnodes[ntype].data['t'])
            trans_out = trans_out * alpha + input_dst[ntype] * (1 - alpha)
            if self.use_norm:
                h[ntype] = self.drop(self.norms[ntype](trans_out))
            else:
                h[ntype] = self.drop(trans_out)
        return h

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class HGT(RGGAN):
    def __init__(
            self, ntypes, etypes, h_dim, node_emb_size, num_bases, n_layers=1, n_heads=1, use_norm=True,
            dropout=0, use_self_loop=False, activation=F.relu, use_gcn_checkpoint=False,
            use_att_checkpoint=False, use_gru_checkpoint=False, **kwargs
    ):
        self.n_heads = n_heads
        self.use_norm = use_norm

        super(HGT, self).__init__(
            ntypes, etypes, h_dim, node_emb_size, num_bases=1, n_layers=n_layers,
            use_gcn_checkpoint=False, use_att_checkpoint=False, use_gru_checkpoint=False
        )

    def _initialize(self):
        # self.n_inp = self.h_dim
        # self.n_hid = self.h_dim
        # self.n_out = self.out_dim
        # self.n_layers = self.n_layers
        self.adapt_ws = nn.ModuleDict()

        for n_type in self.ntypes:
            self.adapt_ws[n_type] = nn.Linear(self.h_dim, self.h_dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(HGTLayer(self.h_dim, self.h_dim, self.ntypes, self.etypes, self.n_heads,
                                     use_norm=self.use_norm))
        self.out = nn.Linear(self.h_dim, self.out_dim)

    # def forward(self, h, blocks=None, graph=None,
    #             return_all=False): # added this as an experimental feature for intermediate supervision
    #     all_layers = [] # added this as an experimental feature for intermediate supervision
    #
    #     if blocks is None:
    #         # full graph training
    #         h0 = h
    #         for layer in self.layers:
    #             h = layer(graph, h, h0)
    #             all_layers.append(h) # added this as an experimental feature for intermediate supervision
    #     else:
    #         # minibatch training
    #         h0 = h
    #         for layer, block in zip(self.layers, blocks):
    #             # h = checkpoint.checkpoint(self.custom(layer), block, h)
    #             h = layer(block, h, h0)
    #             all_layers.append(h) # added this as an experimental feature for intermediate supervision
    #
    #     if return_all: # added this as an experimental feature for intermediate supervision
    #         return all_layers
    #     else:
    #         return h

    def forward(self, input, blocks=None, graph=None, **kwargs):
        h = dict()
        for ntype in input:
            h[ntype] = torch.tanh(self.adapt_ws[ntype](input[ntype]))

        if blocks is not None:
            for block, layer in zip(blocks, self.layers):
                h = layer(block, h)
        else:
            for layer in self.layers:
                h = layer(graph, h)
        h = {key: self.out(val) for key, val in h.items()}
        return h
    #
    # def __repr__(self):
    #     return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
    #         self.__class__.__name__, self.n_inp, self.n_hid,
    #         self.n_out, self.n_layers)
