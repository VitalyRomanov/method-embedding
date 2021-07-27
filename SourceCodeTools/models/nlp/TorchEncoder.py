import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, encoder_dim, out_dim, nheads=1, layers=1):
        super(Encoder, self).__init__()
        # self.embed = nn.Embedding(vocab_size, encoder_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(encoder_dim, nheads, dim_feedforward=encoder_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_lauer, num_layers=layers)
        self.out_adapter = nn.Linear(encoder_dim, out_dim)

    def get_mask(self, max_seq_len, lengths):
        length_mask = torch.arange(max_seq_len).to(self.device).expand(len(lengths), max_seq_len) < lengths.unsqueeze(1)
        mask = torch.zeros_like(length_mask)
        mask.float().masked_fill(mask == False, float('-inf')).masked_fill(mask == True, float(0.0))
        return mask

    def forward(self, input, lengths=None):
        # input = self.embed(input_seq).permute(1, 0, 2)
        input = input.permute(1, 0, 2)
        # mask = self.get_mask(input.size(0), lengths)
        out = self.encoder(input) #, src_key_padding_mask=mask)

        out = self.out_adapter(out)
        return out.permute(1, 0, 2)


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