import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Dense, Dropout
from tensorflow_addons.layers import MultiHeadAttention

from SourceCodeTools.models.nlp.common import positional_encoding


class ConditionalDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(ConditionalDecoderLayer, self).__init__()

        def point_wise_feed_forward_network(d_model, dff):
            return tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
                tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
            ])

        self.mha1 = MultiHeadAttention(d_model, num_heads, return_attn_coef=True)
        self.mha2 = MultiHeadAttention(d_model, num_heads, return_attn_coef=True)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, look_ahead_mask=None, mask=None, training=None):
        encoder_out, x = inputs
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1((x, x, x), mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x) # skip connection

        attn2, attn_weights_block2 = self.mha2(
            (out1, encoder_out, encoder_out), mask=tf.tile(tf.expand_dims(encoder_out._keras_mask, axis=1), (1,out1.shape[1],1)))  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # skip connection (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # skip connection (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class ConditionalAttentionDecoder(tf.keras.layers.Layer):
    def __init__(self, input_dim, out_dim, num_layers, num_heads, ff_hidden, target_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(ConditionalAttentionDecoder, self).__init__()

        self.d_model = out_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, input_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, input_dim)

        self.dec_layers = [ConditionalDecoderLayer(input_dim, num_heads, ff_hidden, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.look_ahead_mask = self.create_look_ahead_mask(1)
        self.fc_out = Dense(out_dim)

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def compute_mask(self, inputs, mask=None):
        # encoder_out, target = inputs
        return mask

    def call(self, inputs, training=None, mask=None):
        encoder_out, target = inputs

        seq_len = tf.shape(target)[1]
        if self.look_ahead_mask.shape[0] != seq_len:
            self.look_ahead_mask = self.create_look_ahead_mask(seq_len)

        attention_weights = {}

        x = self.embedding(target)  # (batch_size, target_seq_len, d_model)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                (encoder_out, x), look_ahead_mask=self.look_ahead_mask, mask=mask, training=training
            )

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return self.fc_out(x), attention_weights


class FlatDecoder(Layer):
    def __init__(self, out_dims, hidden=100, dropout=0.1):
        super(FlatDecoder, self).__init__()
        self.fc1 = Dense(hidden, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.HeNormal())
        self.drop = Dropout(rate=dropout)
        self.fc2 = Dense(out_dims)

    def call(self, inputs, training=None, mask=None):
        encoder_out, target = inputs
        return self.fc2(self.drop(self.fc1(encoder_out, training=training), training=training), training=training), None