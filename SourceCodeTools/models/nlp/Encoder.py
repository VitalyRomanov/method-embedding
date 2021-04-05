from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMEncoder(nn.Module):
    """LSTM encoder."""
    def __init__(self, embed_dim=100, num_layers=1, dropout_in=0.1, dropout_out=0.1):
        super(LSTMEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=self.dropout_out,
            bidirectional=False,
        )

    def forward(self, x):

        bsz, seqlen = x.size()[:-1]

        # embed tokens
        # x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        embed_dim = x.size(2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # # pack embedded source tokens into a PackedSequence
        # packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        h0 = Variable(x.data.new(self.num_layers, bsz, embed_dim).zero_())
        c0 = Variable(x.data.new(self.num_layers, bsz, embed_dim).zero_())
        x, (final_hiddens, final_cells) = self.lstm(
            x,
            (h0, c0),
        )

        # unpack outputs and apply dropout
        # x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0.)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, embed_dim]

        return x.permute(1,0,2)#, final_hiddens, final_cells

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    # for name, param in m.named_parameters():
    #     if 'weight' in name or 'bias' in name:
    #         param.data.uniform_(-0.1, 0.1)
    return m