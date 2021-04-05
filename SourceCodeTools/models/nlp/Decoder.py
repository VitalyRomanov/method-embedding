import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMDecoder(nn.Module):
    def __init__(self, num_buckets, padding=0, encoder_embed_dim=100, embed_dim=100,
                 out_embed_dim=100, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1, use_cuda=True):
        super(LSTMDecoder, self).__init__()
        self.use_cuda = use_cuda
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.encoder_embed_dim = encoder_embed_dim
        self.embed_dim = embed_dim
        self.out_embed_dim = out_embed_dim

        num_embeddings = num_buckets
        padding_idx = padding
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.create_layers(encoder_embed_dim, embed_dim, num_layers)

        if embed_dim != out_embed_dim:
            self.additional_fc = Linear(embed_dim, out_embed_dim)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def create_layers(self, encoder_embed_dim, embed_dim, num_layers):
        self.layers = nn.ModuleList([
            LSTMCell(encoder_embed_dim + embed_dim if layer == 0 else embed_dim, embed_dim)
            for layer in range(num_layers)
        ])

    def forward(self, prev_output_tokens, encoder_out):
        bsz, seqlen = prev_output_tokens.size()

        x = self.embed_tokens(prev_output_tokens) # (bze, seqlen, embed_dim)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        x = x.transpose(0, 1) # (seqlen, bsz, embed_dim)

        num_layers = len(self.layers)
        prev_hiddens = [Variable(x.data.new(bsz, embed_dim).zero_()) for i in range(num_layers)]
        prev_cells = [Variable(x.data.new(bsz, embed_dim).zero_()) for i in range(num_layers)]

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], encoder_out), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # save final output
            outs.append(out)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, embed_dim)
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        x = self.fc_out(x)

        return x


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