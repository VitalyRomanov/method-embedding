import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, encoder_out_dim, decoder_dim, out_dim, vocab_size, nheads=1, layers=1):
        super(Decoder, self).__init__()
        self.encoder_adapter = nn.Linear(encoder_out_dim, decoder_dim)
        self.embed = nn.Embedding(vocab_size, decoder_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(decoder_dim, nheads, dim_feedforward=decoder_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layers)
        self.mask = self.generate_square_subsequent_mask(1)

        self.fc = nn.Linear(decoder_dim, out_dim)

    def forward(self, encoder_out, target):
        encoder_out = self.encoder_adapter(encoder_out).permute(1, 0, 2)
        target = self.embed(target).permute(1, 0, 2)
        if self.mask.size(0) != target.size(0):  # for self-attention
            self.mask = self.generate_square_subsequent_mask(target.size(0)).to(encoder_out.device)
        out = self.decoder(target, encoder_out, tgt_mask=self.mask)

        out = self.fc(out)

        return out.permute(1, 0, 2)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, encoder_dim, output_embed_dim):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.encoder_proj = Linear(encoder_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(2*output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)
        source_hids = self.encoder_proj(source_hids)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        attn_scores = F.softmax(attn_scores.t(), dim=1).t()  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(nn.Module):
    def __init__(self, num_buckets, padding=0, encoder_embed_dim=100, embed_dim=100,
                 out_embed_dim=100, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True):
        embed_dim = out_embed_dim = encoder_embed_dim
        super(LSTMDecoder, self).__init__()
        self.use_cuda = use_cuda
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        num_embeddings = num_buckets
        padding_idx = padding
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.create_layers(encoder_embed_dim, embed_dim, num_layers)

        self.attention = AttentionLayer(out_embed_dim, encoder_embed_dim, embed_dim)
        if embed_dim != out_embed_dim:
            self.additional_fc = Linear(embed_dim, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def create_layers(self, encoder_embed_dim, embed_dim, num_layers):
        self.layers = nn.ModuleList([
            LSTMCell(encoder_embed_dim + embed_dim if layer == 0 else encoder_embed_dim, encoder_embed_dim if layer == 0 else embed_dim)
            for layer in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(encoder_embed_dim + embed_dim if layer == 0 else encoder_embed_dim) for layer in range(num_layers)
        ])

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, inference=False):
        if incremental_state is not None:  # TODO what is this?
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out.unsqueeze(0), None, None

        srclen = encoder_outs.size(0)

        x = self.embed_tokens(prev_output_tokens) # (bze, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        x = x.transpose(0, 1) # (seqlen, bsz, embed_dim)

        # initialize previous states (or get from cache during incremental generation)
        # cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        # initialize previous states (or get from cache during incremental generation)
        cached_state = None  # utils.get_incremental_state(self, incremental_state, 'cached_state')

        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            # _, encoder_hiddens, encoder_cells = encoder_out
            num_layers = len(self.layers)
            prev_hiddens = [Variable(x.data.new(bsz, embed_dim).zero_()) if i != 0 else encoder_outs[0] for i in range(num_layers)]
            prev_cells = [Variable(x.data.new(bsz, embed_dim if i!=0 else encoder_outs.size(-1)).zero_()) for i in range(num_layers)]
            input_feed = Variable(x.data.new(bsz, encoder_outs.size(-1)).zero_())

        attn_scores = Variable(x.data.new(srclen, seqlen, bsz).zero_())
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, (rnn, lnorm) in enumerate(zip(self.layers, self.norms)):
                # recurrent cell
                input = lnorm(input)
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs)
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        # utils.set_incremental_state(
        #     self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        attn_scores = attn_scores.transpose(0, 2)

        x = self.fc_out(x)

        return x#, attn_scores


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    # def reorder_incremental_state(self, incremental_state, new_order):
    #     cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
    #     if cached_state is None:
    #         return
    #
    #     def reorder_state(state):
    #         if isinstance(state, list):
    #             return [reorder_state(state_i) for state_i in state]
    #         return state.index_select(0, new_order)
    #
    #     if not isinstance(new_order, Variable):
    #         new_order = Variable(new_order)
    #     new_state = tuple(map(reorder_state, cached_state))
    #     utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

# class LSTMDecoder(nn.Module):
#     def __init__(self, num_buckets, padding=0, encoder_embed_dim=100, embed_dim=100,
#                  out_embed_dim=100, num_layers=1, dropout_in=0.1,
#                  dropout_out=0.1, use_cuda=True):
#         super(LSTMDecoder, self).__init__()
#         self.use_cuda = use_cuda
#         self.dropout_in = dropout_in
#         self.dropout_out = dropout_out
#         self.encoder_embed_dim = encoder_embed_dim
#         self.embed_dim = embed_dim
#         self.out_embed_dim = out_embed_dim
#
#         num_embeddings = num_buckets
#         padding_idx = padding
#         self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
#
#         self.create_layers(encoder_embed_dim, embed_dim, num_layers)
#
#         if embed_dim != out_embed_dim:
#             self.additional_fc = Linear(embed_dim, out_embed_dim)
#         self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)
#
#     def create_layers(self, encoder_embed_dim, embed_dim, num_layers):
#         self.layers = nn.ModuleList([
#             LSTMCell(encoder_embed_dim + embed_dim if layer == 0 else embed_dim, embed_dim)
#             for layer in range(num_layers)
#         ])
#
#     def forward(self, prev_output_tokens, encoder_out):
#         bsz, seqlen = prev_output_tokens.size()
#
#         x = self.embed_tokens(prev_output_tokens) # (bze, seqlen, embed_dim)
#         x = F.dropout(x, p=self.dropout_in, training=self.training)
#         embed_dim = x.size(2)
#
#         x = x.transpose(0, 1) # (seqlen, bsz, embed_dim)
#
#         num_layers = len(self.layers)
#         prev_hiddens = [Variable(x.data.new(bsz, embed_dim).zero_()) for i in range(num_layers)]
#         prev_cells = [Variable(x.data.new(bsz, embed_dim).zero_()) for i in range(num_layers)]
#
#         outs = []
#         for j in range(seqlen):
#             # input feeding: concatenate context vector from previous time step
#             input = torch.cat((x[j, :, :], encoder_out), dim=1)
#
#             for i, rnn in enumerate(self.layers):
#                 # recurrent cell
#                 hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
#
#                 input = F.dropout(hidden, p=self.dropout_out, training=self.training)
#
#                 # save state for next time step
#                 prev_hiddens[i] = hidden
#                 prev_cells[i] = cell
#
#             out = hidden
#             out = F.dropout(out, p=self.dropout_out, training=self.training)
#
#             # save final output
#             outs.append(out)
#
#         # collect outputs across time steps
#         x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
#         # T x B x C -> B x T x C
#         x = x.transpose(1, 0)
#
#         x = self.fc_out(x)
#
#         return x


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    # for name, param in m.named_parameters():
    #     if 'weight' in name or 'bias' in name:
    #         param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    # m.weight.data.uniform_(-0.1, 0.1)
    # if bias:
    #     m.bias.data.uniform_(-0.1, 0.1)
    return m