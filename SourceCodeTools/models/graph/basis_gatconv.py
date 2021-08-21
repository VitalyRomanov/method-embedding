from dgl import DGLError
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
import dgl.function as fn
from torch import nn, softmax
import torch as th
from torch.utils import checkpoint


class BasisGATConv(GATConv):
    """
    Does not seem to improve memory requirements
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 basis,
                 attn_basis,
                 basis_coef,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 use_checkpoint=False):
        super(GATConv, self).__init__()
        self._basis = basis
        self._basis_coef = basis_coef
        self._attn_basis = attn_basis

        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        # if isinstance(in_feats, tuple):
        #     self.fc_src = nn.Linear(
        #         self._in_src_feats, out_feats * num_heads, bias=False)
        #     self.fc_dst = nn.Linear(
        #         self._in_dst_feats, out_feats * num_heads, bias=False)
        # else:
        #     self.fc = nn.Linear(
        #         self._in_src_feats, out_feats * num_heads, bias=False)
        # self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

        self.dummy_tensor = th.ones(1, dtype=th.float32, requires_grad=True)
        self.use_checkpoint = use_checkpoint

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        # if hasattr(self, 'fc'):
        #     nn.init.xavier_normal_(self.fc.weight, gain=gain)
        # else:
        #     nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        #     nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_l, gain=gain)
        # nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value


    def _forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                basis_coef = softmax(self._basis_coef, dim=-1).reshape(-1, 1, 1)
                # if not hasattr(self, 'fc_src'):
                params_src = (self._basis[0] * basis_coef).sum(dim=0)
                params_dst = (self._basis[1] * basis_coef).sum(dim=0)
                feat_src = (params_src @ h_src.T).view(-1, self._num_heads, self._out_feats)
                feat_dst = (params_dst @ h_dst.T).view(-1, self._num_heads, self._out_feats)
                #     # feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                #     # feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                # else:
                #     params = self._basis * basis_coef
                #     feat_src = (params @ h_src.T).view(-1, self._num_heads, self._out_feats)
                #     feat_dst = (params @ h_dst.T).view(-1, self._num_heads, self._out_feats)
                #     # feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                #     # feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                basis_coef = softmax(self._basis_coef, dim=-1).reshape(-1, 1, 1)
                params = (self._basis * basis_coef).sum(dim=0)
                feat_src = feat_dst = (params @ h_src.T).view(-1, self._num_heads, self._out_feats)
                # feat_src = feat_dst = self.fc(h_src).view(
                #     -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            attn_l_param = (self._attn_basis[0] * basis_coef).sum(dim=0)
            attn_r_param = (self._attn_basis[1] * basis_coef).sum(dim=0)
            el = (feat_src * attn_l_param).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * attn_r_param).sum(dim=-1).unsqueeze(-1)
            # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
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
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

    def custom(self, graph, get_attention):
        def custom_forward(*inputs):
            feat0, feat1, dummy = inputs
            return self._forward(graph, (feat0, feat1), get_attention=get_attention)
        return custom_forward

    def forward(self, graph, feat, get_attention=False):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self.custom(graph, get_attention), feat[0], feat[1], self.dummy_tensor).squeeze(1)
        else:
            return self._forward(graph, feat, get_attention=get_attention).squeeze(1)