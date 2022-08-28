"""RGCN layer implementation"""
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl import dataloading
from torch import ones, float32, Tensor, split, matmul, no_grad, zeros, arange
from torch.utils import checkpoint
from tqdm import tqdm


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

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(Tensor(len(self.rel_names), in_feat, out_feat))
                # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(self.weight)

        # TODO
        # think of possibility switching to GAT
        # rel : dglnn.GATConv(in_feat, out_feat, num_heads=4)
        # rel : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False, allow_zero_in_degree=True)
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
            # if self.use_basis:
            #     self.loop_weight_basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.ntype_names))
            # else:
            #     self.loop_weight = nn.Parameter(th.Tensor(len(self.ntype_names), in_feat, out_feat))
            #     # nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            #     nn.init.xavier_normal_(self.loop_weight)
            self.loop_weight = nn.Parameter(Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('tanh'))
            # # nn.init.xavier_normal_(self.loop_weight)

        self.dropout = nn.Dropout(dropout)

    def create_conv(self, in_feat, out_feat, rel_names):
        self.conv = dglnn.HeteroGraphConv({
            rel: CkptGATConv((in_feat, in_feat), out_feat, num_heads=1, use_checkpoint=self.use_gcn_checkpoint)
            for rel in rel_names
        })

    def forward(self, g, inputs):
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
            # if self.self_loop:
            #     self_loop_weight = self.loop_weight_basis() if self.use_basis else self.loop_weight
            #     self_loop_wdict = {self.ntype_names[i]: w.squeeze(0)
            #              for i, w in enumerate(th.split(self_loop_weight, 1, dim=0))}
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            # the begginning of src and dst indexes match, that is why we can simply slice the first
            # nodes to get dst embeddings
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                # h = h + th.matmul(inputs_dst[ntype], self_loop_wdict[ntype])
                h = h + matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.bias_dict[ntype]
                # h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        # TODO
        # think of possibility switching to GAT
        # return {ntype: _apply(ntype, h) for ntype, h in hs.items()}
        return {ntype: _apply(ntype, h).mean(1) for ntype, h in hs.items()}

# class RelGraphEmbed(nn.Module):
#     r"""Embedding layer for featureless heterograph."""
#     def __init__(self,
#                  g,
#                  embed_size,
#                  embed_name='embed',
#                  activation=None,
#                  dropout=0.0):
#         super(RelGraphEmbed, self).__init__()
#         self.g = g
#         self.embed_size = embed_size
#         self.embed_name = embed_name
#         # self.activation = activation
#         # self.dropout = nn.Dropout(dropout)
#
#         # create weight embeddings for each node for each relation
#         self.embeds = nn.ParameterDict()
#         for ntype in g.ntypes:
#             embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
#             # TODO
#             # watch for activation in init
#             # nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
#             nn.init.xavier_normal_(embed)
#             self.embeds[ntype] = embed
#
#     def forward(self, block=None):
#         """Forward computation
#
#         Parameters
#         ----------
#         block : DGLHeteroGraph, optional
#             If not specified, directly return the full graph with embeddings stored in
#             :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
#             graph and return.
#
#         Returns
#         -------
#         DGLHeteroGraph
#             The block graph fed with embeddings.
#         """
#         return self.embeds


class RGCNSampling(nn.Module):
    def __init__(
            self, ntypes, etypes, h_dim, node_emb_size, num_bases, num_hidden_layers=1, dropout=0, use_self_loop=False,
            activation=F.relu, use_gcn_checkpoint=False, **kwargs
    ):
        super(RGCNSampling, self).__init__()
        self.ntypes = ntypes
        self.etypes = etypes
        self.h_dim = h_dim
        self.out_dim = node_emb_size
        self.activation = activation
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_gcn_checkpoint = use_gcn_checkpoint

        self._initialize()

    def _initialize(self):
        self.rel_names = list(set(self.etypes))
        self.rel_names.sort()
        self.ntype_names = list(set(self.ntypes))
        self.ntype_names.sort()
        if self.num_bases < 0 or self.num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = self.num_bases
        self.num_hidden_layers = self.num_hidden_layers
        self.dropout = self.dropout
        self.use_self_loop = self.use_self_loop

        # self.embed_layer = RelGraphEmbed(g, self.h_dim)
        self.layers = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.h_dim, self.rel_names, self.ntype_names,
            self.num_bases, activation=self.activation, self_loop=self.use_self_loop,
            dropout=self.dropout, weight=False, use_gcn_checkpoint=self.use_gcn_checkpoint))
        self.layer_norm.append(nn.LayerNorm([self.h_dim]))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, self.rel_names, self.ntype_names,
                self.num_bases, activation=self.activation, self_loop=self.use_self_loop,
                dropout=self.dropout, weight=False,
                use_gcn_checkpoint=self.use_gcn_checkpoint))  # changed weight for GATConv
            self.layer_norm.append(nn.LayerNorm([self.h_dim]))
            # TODO
            # think of possibility switching to GAT
            # weight=False
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, self.rel_names, self.ntype_names,
            self.num_bases, activation=None,
            self_loop=self.use_self_loop, weight=False,
            use_gcn_checkpoint=self.use_gcn_checkpoint))  # changed weight for GATConv
        self.layer_norm.append(nn.LayerNorm([self.out_dim]))
        # TODO
        # think of possibility switching to GAT
        # weight=False

        self.emb_size = self.out_dim
        self.num_layers = len(self.layers)

    # def node_embed(self):
    #     return self.embed_layer()

    # def forward(self, h=None, blocks=None):
    #     if h is None:
    #         # full graph training
    #         h = self.embed_layer()
    #     if blocks is None:
    #         # full graph training
    #         for layer in self.layers:
    #             h = layer(self.g, h)
    #     else:
    #         # minibatch training
    #         for layer, block in zip(self.layers, blocks):
    #             h = layer(block, h)
    #     return h

    def custom(self, layer):
        def custom_forward(*inputs):
            block, h = inputs
            h = layer(block, h)
            return h
        return custom_forward

    def normalize(self, h, layer):
        return {key: layer(val) for key, val in h.items()}
        # return {key: val / torch.linalg.norm(val, dim=1, keepdim=True) for key, val in h.items()}

    def forward(self, h, blocks=None, graph=None,
                return_all=False): # added this as an experimental feature for intermediate supervision
        # if h is None:
        #     # full graph training
        #     h = self.embed_layer()

        all_layers = [] # added this as an experimental feature for intermediate supervision

        if blocks is None:
            # full graph training
            h0 = h
            for layer, norm in zip(self.layers, self.layer_norm):
                h = layer(graph, h, h0)
                h = self.normalize(h, norm)
                all_layers.append(h) # added this as an experimental feature for intermediate supervision
        else:
            # minibatch training
            h0 = h
            for layer, norm, block in zip(self.layers, self.layer_norm, blocks):
                # h = checkpoint.checkpoint(self.custom(layer), block, h)
                h = layer(block, h, h0)
                h = self.normalize(h, norm)
                all_layers.append(h) # added this as an experimental feature for intermediate supervision

        if return_all: # added this as an experimental feature for intermediate supervision
            return all_layers
        else:
            return h

    def inference(self, batch_size, device, num_workers, x=None):
        """Minibatch inference of final representation over all node types.

        ***NOTE***
        For node classification, the model is trained to predict on only one node type's
        label.  Therefore, only that type's final representation is meaningful.
        """
        h0 = x

        with no_grad():

            # if x is None:
            #     x = self.embed_layer()

            for l_ind, (layer, norm) in enumerate(zip(self.layers, self.layer_norm)):
                y = {
                    k: zeros(
                        self.g.number_of_nodes(k),
                        self.h_dim if l_ind != len(self.layers) - 1 else self.out_dim)
                    for k in self.g.ntypes}

                sampler = dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dataloading.NodeDataLoader(
                    self.g,
                    {k: arange(self.g.number_of_nodes(k)) for k in self.g.ntypes},
                    sampler,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=num_workers)

                for input_nodes, output_nodes, blocks in tqdm(dataloader, desc=f"Layer {l_ind}"):
                    block = blocks[0].to(device)

                    if not isinstance(input_nodes, dict):
                        key = next(iter(self.g.ntypes))
                        input_nodes = {key: input_nodes}
                        output_nodes = {key: output_nodes}

                    _h0 = {k: h0[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                    h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                    h = layer(block, h, _h0)
                    h = self.normalize(h, norm)

                    for k in h.keys():
                        y[k][output_nodes[k]] = h[k].cpu()

                x = y
            return y

    # def get_layers(self):
    #     """
    #     Retrieve tensor values on the layers for further use as node embeddings.
    #     :return:
    #     """
    #     # h = self.embed_layer()
    #     # l_out = [th.cat([h[ntype] for ntype in self.g.ntypes], dim=0).detach().numpy()]
    #     # for layer in self.layers:
    #     #     h = layer(self.g, h)
    #     #     l_out.append(th.cat([h[ntype] for ntype in self.g.ntypes], dim=0).detach().numpy())
    #
    #     h = self.inference(batch_size=256, device='cpu', num_workers=0)
    #     l_out = [th.cat([h[ntype] for ntype in self.g.ntypes], dim=0).detach().numpy()]
    #
    #     return l_out
    #
    # def get_embeddings(self, id_maps):
    #     return [Embedder(id_maps, e) for e in self.get_layers()]
