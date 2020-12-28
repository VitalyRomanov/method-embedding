import torch
import torch.nn as nn
import dgl
import dgl.function as fn

from dgl.nn.pytorch import edge_softmax

from SourceCodeTools.graph.model.Embedder import Embedder

class NodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdate, self).__init__()
        self.dense = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = node.data['h']
        h = self.dense(h)
        if self.activation:
            h = self.activation(h)
        return {'activation': h}

    def inference(self, x):
        h = x
        h = self.dense(h)
        if self.activation:
            h = self.activation(h)
        return h


class GCNSampling(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 n_hidden,
                 num_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.g = g
        self.dropout = dropout
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(NodeUpdate(in_dim, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers-1):
            self.layers.append(NodeUpdate(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(NodeUpdate(n_hidden, num_classes))

    def forward(self, nf=None):
        if nf is None:
            return self.inference()

        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = nn.Dropout(self.dropout)(h)
            nf.layers[i].data['h'] = h
            # block_compute() computes the feature of layer i given layer
            # i-1, with the given message, reduce, and apply functions.
            # Here, you essentially aggregate the neighbor node features in
            # the previous layer, and update it with the `layer` function.
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             lambda node : {'h': node.mailbox['m'].mean(axis=1)},
                             layer)
        h = nf.layers[-1].data.pop('activation')
        return h

    def inference(self):
        self.g.ndata['activation'] = self.g.ndata['features']
        for i, layer in enumerate(self.layers):
            h = self.g.ndata.pop('activation')
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_src(src='h', out='m'),
                             lambda node: {'h': node.mailbox['m'].mean(axis=1)},
                             layer)

        h = self.g.ndata.pop('activation')
        return h


    def get_layers(self):
        """
        Retrieve tensor values on the layers for further use as node embeddings.
        :return:
        """
        self.g.ndata['activation'] = self.g.ndata['features']

        l_out = [self.g.ndata['activation'].detach().numpy()]

        for i, layer in enumerate(self.layers):
            h = self.g.ndata.pop('activation')
            self.g.ndata['h'] = h
            self.g.update_all(fn.copy_src(src='h', out='m'),
                              lambda node: {'h': node.mailbox['m'].mean(axis=1)},
                              layer)
            l_out.append(self.g.ndata['activation'].detach().numpy())

        return l_out

    def get_embeddings(self, id_maps):
        return [Embedder(id_maps, e) for e in self.get_layers()]






class GATConvSampler(dgl.nn.pytorch.GATConv):
    def __init__(self, *args, **kwargs):
        super(GATConvSampler, self).__init__(*args, **kwargs)

    def forward(self, g, h=None, nf=None, layer=None, flatten=False):

        if layer is None:
            return self.inference(g, h)

        for block_ind in range(nf.num_blocks):
            nodespace = nf.layers[block_ind]
            next_nodespace = nf.layers[block_ind + 1]

            featl = nodespace.data['activations']
            featr = next_nodespace.data['activations']

            h_src = self.feat_drop(featl)
            h_dst = self.feat_drop(featr)

            feat_src = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
            feat_dst = self.fc(h_dst).view(
                -1, self._num_heads, self._out_feats)

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

            nodespace.data.update({'ft': feat_src, 'el': el})
            next_nodespace.data.update({'er': er})

            edgespace = nf.blocks[block_ind]
            nf.apply_block(block_ind, func=fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(edgespace.data.pop('e'))
            edgespace.data['a'] = self.attn_drop(edge_softmax(g, e, nf.block_parent_eid(block_ind)))
            nf.block_compute(block_ind, message_func=fn.u_mul_e('ft', 'a', 'm'),
                             reduce_func=fn.sum('m', 'ft'))

            rst = next_nodespace.data['ft']

            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            next_nodespace.data['new_activations'] = rst.flatten(1) if flatten else rst

        for l in range(1, nf.num_layers):
            nf.layers[l].data.update({'activations': nf.layers[l].data.pop('new_activations')})

        return nf.layers[-1].data['activations']

        # nodespace = nf.layers[layer]
        # next_nodespace = nf.layers[layer + 1]
        # featl = nodespace.data['features']
        # featr = next_nodespace.data['features']
        #
        # edgespace = nf.blocks[layer]
        #
        # h_src = self.feat_drop(featl)
        # h_dst = self.feat_drop(featr)
        #
        # feat_src = self.fc(h_src).view(
        #         -1, self._num_heads, self._out_feats)
        # feat_dst = self.fc(h_dst).view(
        #         -1, self._num_heads, self._out_feats)
        #
        # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        # nodespace.data.update({'ft': feat_src, 'el': el})
        # next_nodespace.data.update({'er': er})
        #
        # nf.apply_block(layer, func=fn.u_add_v('el', 'er', 'e'))
        # e = self.leaky_relu(edgespace.data.pop('e'))
        # # compute softmax
        # # https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/softmax.html
        # edgespace.data['a'] = self.attn_drop(edge_softmax(g, e, nf.block_parent_eid(layer)))
        # # message passing
        # nf.block_compute(layer, message_func=fn.u_mul_e('ft', 'a', 'm'),
        #                  reduce_func=fn.sum('m', 'ft'))
        # rst = next_nodespace.data['ft']
        # # residual
        # if self.res_fc is not None:
        #     resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
        #     rst = rst + resval
        # # activation
        # if self.activation:
        #     rst = self.activation(rst)
        # return rst

    def inference(self, graph, h):

        feat = h

        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            -1, self._num_heads, self._out_feats)

        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

class GATSampler(nn.Module):
    # https://docs.dgl.ai/en/0.4.x/api/python/nodeflow.html
    def __init__(self,
                 g,
                 num_layers, #n_layers
                 in_dim, #in_feats
                 num_hidden, #n_hidden,
                 num_classes, #n_classes,
                 heads,
                 activation,
                 feat_drop, #dropout
                 attn_drop,
                 negative_slope,
                 residual,
                 **kwargs):
        super(GATSampler, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        # node embeddings
        # embed_dict = {ntype: nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), in_dim))
        #               for ntype in g.ntypes}
        self.embed = nn.Parameter(torch.Tensor(g.number_of_nodes(), in_dim))
        # input projection (no residual)
        self.layers.append(GATConvSampler(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.layers.append(GATConvSampler(
                num_hidden * heads[l - 1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.layers.append(GATConvSampler(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

        self.emb_size = num_classes

    def forward(self, nf=None):

        if nf is None:
            return self.inference()

        for l in range(nf.num_layers):
            nf.layers[l].data['activations'] = nf.layers[l].data['features']

        for l in range(self.num_layers):
            h = self.layers[l](self.g, nf=nf, layer=l, flatten=True)
        # output projection
        logits = self.layers[-1](self.g, nf=nf, layer=self.num_layers).mean(1)
        return logits

    def inference(self):
        h = self.g.ndata['features']
        for l in range(self.num_layers):
            h = self.layers[l](self.g, h=h).flatten(1)
        # output projection
        logits = self.layers[-1](self.g, h=h).mean(1)
        return logits

    def get_layers(self):
        """
        Retrieve tensor values on the layers for further use as node embeddings.
        :return:
        """

        h = self.g.ndata['features']
        l_out = [h.detach().numpy()]
        for l in range(self.num_layers):
            h = self.layers[l](self.g, h).flatten(1)
            l_out.append(h.detach().numpy())
        # output projection
        logits = self.layers[-1](self.g, h).mean(1)
        l_out.append(logits.detach().numpy())

        return l_out

    def get_embeddings(self, id_maps):
        return [Embedder(id_maps, e) for e in self.get_layers()]

    def node_embed(self):
        return {'_U':self.embed}