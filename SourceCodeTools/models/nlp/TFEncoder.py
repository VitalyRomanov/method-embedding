from copy import copy

from scipy.linalg import toeplitz
from tensorflow.keras.layers import Layer, Dense, Conv2D, Flatten, Input, Embedding, concatenate, GRU
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.python.keras.layers import Dropout


class DefaultEmbedding(Layer):
    """
    Creates an embedder that provides the default value for the index -1. The default value is a zero-vector
    """
    def __init__(self, init_vectors=None, shape=None, trainable=True):
        super(DefaultEmbedding, self).__init__()

        if init_vectors is not None:
            self.embs = tf.Variable(init_vectors, dtype=tf.float32,
                           trainable=trainable, name="default_embedder_var")
            shape = init_vectors.shape
        else:
            # TODO
            # the default value is no longer constant. need to replace this with a standard embedder
            self.embs = tf.Variable(tf.random.uniform(shape=(shape[0], shape[1]), dtype=tf.float32),
                               name="default_embedder_pad")
        # self.pad = tf.zeros(shape=(1, init_vectors.shape[1]), name="default_embedder_pad")
        # self.pad = tf.random.uniform(shape=(1, init_vectors.shape[1]), name="default_embedder_pad")
        self.pad = tf.Variable(tf.random.uniform(shape=(1, shape[1]), dtype=tf.float32),
                               name="default_embedder_pad")

    # def compute_mask(self, inputs, mask=None):
    #     ids, lengths = inputs
    #     # position with value -1 is a pad
    #     return tf.sequence_mask(lengths, ids.shape[1])
    #     # return inputs != self.embs.shape[0]

    def call(self, ids):
        emb_matr = tf.concat([self.embs, self.pad], axis=0)
        return tf.nn.embedding_lookup(params=emb_matr, ids=ids)
        # return tf.expand_dims(tf.nn.embedding_lookup(params=self.emb_matr, ids=ids), axis=3)


class PositionalEncoding(Model):
    def __init__(self, seq_len, pos_emb_size):
        """
        Create positional embedding with a trainable embedding matrix. Currently not using because it results
         in N^2 computational complexity. Should move this functionality to batch preparation.
        :param seq_len: maximum sequence length
        :param pos_emb_size: the dimensionality of positional embeddings
        """
        super(PositionalEncoding, self).__init__()

        positions = list(range(seq_len * 2))
        position_splt = positions[:seq_len]
        position_splt.reverse()
        self.position_encoding = tf.constant(toeplitz(position_splt, positions[seq_len:]),
                                        dtype=tf.int32,
                                        name="position_encoding")
        # self.position_embedding = tf.random.uniform(name="position_embedding", shape=(seq_len * 2, pos_emb_size), dtype=tf.float32)
        self.position_embedding = tf.Variable(tf.random.uniform(shape=(seq_len * 2, pos_emb_size), dtype=tf.float32),
                               name="position_embedding")
        # self.position_embedding = tf.Variable(name="position_embedding", shape=(seq_len * 2, pos_emb_size), dtype=tf.float32)

    def call(self):
        # return tf.nn.embedding_lookup(self.position_embedding, self.position_encoding, name="position_lookup")
        return tf.nn.embedding_lookup(self.position_embedding, self.position_encoding, name="position_lookup")


class TextCnnLayer(Model):
    def __init__(self, out_dim, kernel_shape, activation=None):
        super(TextCnnLayer, self).__init__()

        self.kernel_shape = kernel_shape
        self.out_dim = out_dim

        self.textConv = Conv2D(filters=out_dim, kernel_size=kernel_shape,
                                  activation=activation, data_format='channels_last')

        padding_size = (self.kernel_shape[0] - 1) // 2
        assert padding_size * 2 + 1 == self.kernel_shape[0]
        self.pad_constant = tf.constant([[0, 0], [padding_size, padding_size], [0, 0], [0, 0]])

        self.supports_masking = True

    def call(self, x, training=None, mask=None):
        padded = tf.pad(x, self.pad_constant)
        # emb_sent_exp = tf.expand_dims(input, axis=3)
        convolve = self.textConv(padded)
        return tf.squeeze(convolve, axis=-2)


class TextCnnEncoder(Model):
    """
    TextCnnEncoder model for classifying tokens in a sequence. The model uses following pipeline:

    token_embeddings (provided from outside) ->
    several convolutional layers, get representations for all tokens ->
    pass representation for all tokens through a dense network ->
    classify each token
    """
    def __init__(self, input_size, h_sizes, seq_len,
                 pos_emb_size, cnn_win_size, dense_size, out_dim,
                 activation=None, dense_activation=None, drop_rate=0.2):
        """

        :param input_size: dimensionality of input embeddings
        :param h_sizes: sizes of hidden CNN layers, internal dimensionality of token embeddings
        :param seq_len: maximum sequence length
        :param pos_emb_size: dimensionality of positional embeddings
        :param cnn_win_size: width of cnn window
        :param dense_size: number of unius in dense network
        :param num_classes: number of output units
        :param activation: activation for cnn
        :param dense_activation: activation for dense layers
        :param drop_rate: dropout rate for dense network
        """
        super(TextCnnEncoder, self).__init__()

        self.seq_len = seq_len
        self.h_sizes = h_sizes
        self.dense_size = dense_size
        self.out_dim = out_dim

        def infer_kernel_sizes(h_sizes):
            """
            Compute kernel sizes from the desired dimensionality of hidden layers
            :param h_sizes:
            :return:
            """
            kernel_sizes = copy(h_sizes)
            kernel_sizes.pop(-1) # pop last because it is the output of the last CNN layer
            kernel_sizes.insert(0, input_size) # the first kernel size should be (cnn_win_size, input_size)
            kernel_sizes = [(cnn_win_size, ks) for ks in kernel_sizes]
            return kernel_sizes

        kernel_sizes = infer_kernel_sizes(h_sizes)

        self.layers_tok = [TextCnnLayer(out_dim=h_size, kernel_shape=kernel_size, activation=activation)
            for h_size, kernel_size in zip(h_sizes, kernel_sizes)]

        # self.layers_pos = [TextCnnLayer(out_dim=h_size, kernel_shape=(cnn_win_size, pos_emb_size), activation=activation)
        #                for h_size, _ in zip(h_sizes, kernel_sizes)]

        # self.positional = PositionalEncoding(seq_len=seq_len, pos_emb_size=pos_emb_size)

        if dense_activation is None:
            dense_activation = activation

        # self.attention = tfa.layers.MultiHeadAttention(head_size=200, num_heads=1)

        self.dense_1 = Dense(dense_size, activation=dense_activation)
        self.dropout_1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dense_2 = Dense(out_dim, activation=None) # logits
        self.dropout_2 = tf.keras.layers.Dropout(rate=drop_rate)

        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, embs, training=True, mask=None):

        temp_cnn_emb = embs # shape (?, seq_len, input_size)

        # pass embeddings through several CNN layers
        for l in self.layers_tok:
            temp_cnn_emb = l(tf.expand_dims(temp_cnn_emb, axis=3)) # shape (?, seq_len, h_size)

        # TODO
        # simplify to one CNN and one attention

        # pos_cnn = self.positional()
        # for l in self.layers_pos:
        #     pos_cnn = l(tf.expand_dims(pos_cnn, axis=3))
        #
        # cnn_pool_feat = []
        # for i in range(self.seq_len):
        #     # slice tensor for the line that corresponds to the current position in the sentence
        #     position_features = tf.expand_dims(pos_cnn[i, ...], axis=0, name="exp_dim_%d" % i)
        #     # convolution without activation can be combined later, hence: temp_cnn_emb + position_features
        #     cnn_pool_feat.append(
        #         tf.expand_dims(tf.nn.tanh(tf.reduce_max(temp_cnn_emb + position_features, axis=1)), axis=1))
        #     # cnn_pool_feat.append(
        #     #     tf.expand_dims(tf.nn.tanh(tf.reduce_max(tf.concat([temp_cnn_emb, position_features], axis=-1), axis=1)), axis=1))
        #
        # cnn_pool_features = tf.concat(cnn_pool_feat, axis=1)
        cnn_pool_features = temp_cnn_emb

        # cnn_pool_features = self.attention([cnn_pool_features, cnn_pool_features])

        # token_features = self.dropout_1(
        #     tf.reshape(cnn_pool_features, shape=(-1, self.h_sizes[-1]))
        #     , training=training)

        # reshape before passing through a dense network
        # token_features = tf.reshape(cnn_pool_features, shape=(-1, self.h_sizes[-1])) # shape (? * seq_len, h_size[-1])

        # local_h2 = self.dropout_2(
        #     self.dense_1(token_features)
        #     , training=training)
        local_h2 = self.dense_1(cnn_pool_features) # shape (? * seq_len, dense_size)
        tag_logits = self.dense_2(local_h2) # shape (? * seq_len, num_classes)

        return tag_logits  # tf.reshape(tag_logits, (-1, seq_len, self.out_dim)) # reshape back, shape (?, seq_len, num_classes)


class GRUEncoder(Model):
    def __init__(self, input_dim, out_dim=100, num_layers=1, dropout=0.1):
        super(GRUEncoder, self).__init__()
        self.num_layers = num_layers

        self.gru_layers = [
            tf.keras.layers.Bidirectional(GRU(out_dim, dropout=dropout, return_sequences=True)) for _ in range(num_layers)
        ]

        self.dropout = Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for layer in self.gru_layers:
            x = layer(x, training=training, mask=mask)
            x = self.dropout(x, training=training)

        return x