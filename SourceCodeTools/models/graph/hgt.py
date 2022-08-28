import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from SourceCodeTools.models.graph import RGGAN


class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['id'][0]
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

    def forward(self, G, inp_key, out_key):
        node_dict, edge_dict = G.node_dict, G.edge_dict
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]]
            q_linear = self.q_linears[node_dict[dsttype]]

            G.nodes[srctype].data['k'] = k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] = v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] = q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)

            G.apply_edges(func=self.edge_attention, etype=etype)
        G.multi_update_all({etype: (self.message_func, self.reduce_func) \
                            for etype in edge_dict}, cross_reducer='mean')
        for ntype in G.ntypes:
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = trans_out * alpha + G.nodes[ntype].data[inp_key] * (1 - alpha)
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(self.norms[n_id](trans_out))
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)


class HGT(RGGAN):
    def __init__(
            self, ntypes, etypes, h_dim, node_emb_size, num_bases, num_hidden_layers, n_heads=1, use_norm=True,
            dropout=0, use_self_loop=False, activation=F.relu, use_gcn_checkpoint=False,
            use_att_checkpoint=False, use_gru_checkpoint=False, **kwargs
    ):
        super(HGT, self).__init__(
            ntypes, etypes, h_dim, node_emb_size, num_bases=1, num_hidden_layers=num_hidden_layers,
            use_gcn_checkpoint=False, use_att_checkpoint=False, use_gru_checkpoint=False
        )

        self.ntypes = ntypes
        self.etypes = etypes
        self.h_dim = h_dim
        self.out_dim = node_emb_size
        self.activation = activation
        self.num_bases = num_bases
        self.n_heads = n_heads
        self.use_norm = use_norm
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_gcn_checkpoint = use_gcn_checkpoint
        self.use_att_checkpoint = use_att_checkpoint
        self.use_gru_checkpoint = use_gru_checkpoint

        self._initialize()

    def _initialize(self):
        self.gcs = nn.ModuleList()
        self.n_inp = self.h_dim
        self.n_hid = self.h_dim
        self.n_out = self.node_emb_size
        self.n_layers = self.n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(self.ntypes)):
            self.adapt_ws.append(nn.Linear(self.h_dim, self.h_dim))
        for _ in range(self.n_layers):
            self.gcs.append(HGTLayer(self.h_dim, self.h_dim, len(self.ntypes), len(self.etypes), self.n_heads,
                                     use_norm=self.use_norm))
        self.out = nn.Linear(self.h_dim, self.node_emb_size)

    # def forward(self, h, blocks=None, graph=None, return_all=False):
    #     for ntype in G.ntypes:
    #         n_id = G.node_dict[ntype]
    #         G.nodes[ntype].data['h'] = torch.tanh(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
    #     for i in range(self.n_layers):
    #         self.gcs[i](G, 'h', 'h')
    #     return self.out(G.nodes[out_key].data['h'])
    #
    # def __repr__(self):
    #     return '{}(n_inp={}, n_hid={}, n_out={}, n_layers={})'.format(
    #         self.__class__.__name__, self.n_inp, self.n_hid,
    #         self.n_out, self.n_layers)
