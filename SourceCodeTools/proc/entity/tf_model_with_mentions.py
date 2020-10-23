# import tensorflow as tf
import sys
# from gensim.models import Word2Vec
import numpy as np
from collections import Counter
from scipy.linalg import toeplitz
# from gensim.models import KeyedVectors
from copy import copy

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Embedding, concatenate
from tensorflow.keras import Model
from tensorflow.keras import regularizers

from spacy.gold import offsets_from_biluo_tags

# alternative models
# https://github.com/flairNLP/flair/tree/master/flair/models
# https://github.com/dhiraa/tener/tree/master/src/tener/models
# https://arxiv.org/pdf/1903.07785v1.pdf
# https://github.com/tensorflow/models/tree/master/research/cvt_text/model

from SourceCodeTools.proc.entity.tf_model import DefaultEmbedding, TextCnnLayer


class TextCnn(Model):
    def __init__(self, input_size, h_sizes, seq_len,
                 pos_emb_size, cnn_win_size, dense_size, num_classes,
                 activation=None, dense_activation=None, drop_rate=0.2):
        super(TextCnn, self).__init__()

        self.seq_len = seq_len
        self.h_sizes = h_sizes
        self.dense_size = dense_size
        self.num_classes = num_classes

        kernel_sizes = copy(h_sizes)
        kernel_sizes.pop(-1)
        kernel_sizes.insert(0, input_size)
        kernel_sizes = [(cnn_win_size, ks) for ks in kernel_sizes]

        # self.layers_tok = [ TextCnnLayer(out_dim=h_size, kernel_shape=kernel_size, activation=activation)
        #     for h_size, kernel_size in zip(h_sizes, kernel_sizes)]

        self.layers_tok = [tfa.layers.MultiHeadAttention(head_size=h_size, num_heads=1, output_size=h_size)
                           for h_size in h_sizes]

        self.layers_pos = [TextCnnLayer(out_dim=h_size, kernel_shape=(cnn_win_size, pos_emb_size), activation=activation)
                       for h_size, _ in zip(h_sizes, kernel_sizes)]

        # self.positional = PositionalEncoding(seq_len=seq_len, pos_emb_size=pos_emb_size)

        if dense_activation is None:
            dense_activation = activation

        self.attention = tfa.layers.MultiHeadAttention(head_size=200, num_heads=1)

        self.dense_1 = Dense(dense_size, activation=dense_activation)
        self.dropout_1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dense_2 = Dense(num_classes, activation=None) # logits
        self.dropout_2 = tf.keras.layers.Dropout(rate=drop_rate)

    def __call__(self, embs, training=True):

        temp_cnn_emb = embs

        # for l in self.layers_tok:
        #     temp_cnn_emb = l(tf.expand_dims(temp_cnn_emb, axis=3))

        for l in self.layers_tok:
            temp_cnn_emb = l([temp_cnn_emb, temp_cnn_emb])

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
        # cnn_pool_features = tf.math.reduce_max(temp_cnn_emb, axis=1)

        # token_features = self.dropout_1(
        #     tf.reshape(cnn_pool_features, shape=(-1, self.h_sizes[-1]))
        #     , training=training)
        token_features = tf.reshape(cnn_pool_features, shape=(-1, self.h_sizes[-1]))

        # local_h2 = self.dropout_2(
        #     self.dense_1(token_features)
        #     , training=training)
        local_h2 = self.dense_1(token_features)
        # local_h2 = self.dense_1(cnn_pool_features)
        tag_logits = self.dense_2(local_h2)

        # return tag_logits
        return tf.reshape(tag_logits, (-1, self.seq_len, self.num_classes))


class TypePredictor(Model):
    def __init__(self, tok_embedder, graph_embedder, train_embeddings=False,
                 h_sizes=[500], dense_size=100, num_classes=None,
                 seq_len=100, pos_emb_size=30, cnn_win_size=3,
                 crf_transitions=None, suffix_prefix_dims=50, suffix_prefix_buckets=1000,
                 target_emb_dim=15, mention_emb_dim=15):
        super(TypePredictor, self).__init__()
        assert num_classes is not None, "set num_classes"

        self.seq_len = seq_len
        self.transition_params = crf_transitions

        with tf.device('/CPU:0'):
            self.tok_emb = DefaultEmbedding(init_vectors=tok_embedder.e, trainable=train_embeddings)
            self.graph_emb = DefaultEmbedding(init_vectors=graph_embedder.e, trainable=train_embeddings)
        self.prefix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))
        self.suffix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))
        self.target_emb = Embedding(2, target_emb_dim)
        self.mention_emb = Embedding(2, mention_emb_dim)

        # self.tok_emb = Embedding(input_dim=tok_embedder.e.shape[0],
        #                          output_dim=tok_embedder.e.shape[1],
        #                          weights=tok_embedder.e, trainable=train_embeddings,
        #                          mask_zero=True)
        #
        # self.graph_emb = Embedding(input_dim=graph_embedder.e.shape[0],
        #                          output_dim=graph_embedder.e.shape[1],
        #                          weights=graph_embedder.e, trainable=train_embeddings,
        #                          mask_zero=True)

        #
        input_dim = tok_embedder.e.shape[1] + suffix_prefix_dims * 2 + graph_embedder.e.shape[1] + target_emb_dim + mention_emb_dim

        self.text_cnn = TextCnn(input_size=input_dim, h_sizes=h_sizes,
                                seq_len=seq_len, pos_emb_size=pos_emb_size,
                                cnn_win_size=cnn_win_size, dense_size=dense_size,
                                num_classes=num_classes, activation=tf.nn.relu,
                                dense_activation=tf.nn.tanh)

        # self.accuracy = tf.keras.metrics.Accuracy()

        self.crf_transition_params = None

    # def compute_mask(self, inputs, mask=None):
    #     mask should come from ids
    #     if mask is None:
    #         return None
    #
    #     return mask


    def __call__(self, token_ids, prefix_ids, suffix_ids, graph_ids, target, mentions, training=True):

        tok_emb = self.tok_emb(token_ids)
        graph_emb = self.graph_emb(graph_ids)
        prefix_emb = self.prefix_emb(prefix_ids)
        suffix_emb = self.suffix_emb(suffix_ids)
        target_emb = self.target_emb(target)
        mention_emb = self.mention_emb(mentions)


        embs = tf.concat([tok_emb,
                          graph_emb,
                          prefix_emb,
                          suffix_emb,
                          target_emb,
                          mention_emb], axis=-1)

        logits = self.text_cnn(embs, training=training)

        return logits


    def loss(self, logits, labels, lengths, class_weights=None):
        losses = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth=logits.shape[-1]), logits, axis=-1)
        if class_weights is None:
            loss = tf.reduce_mean(tf.boolean_mask(losses, tf.sequence_mask(lengths, self.seq_len)))
        else:
            loss = tf.reduce_mean(tf.boolean_mask(losses * class_weights, tf.sequence_mask(lengths, self.seq_len)))

        # log_likelihood, transition_params = tfa.text.crf_log_likelihood(logits, labels, lengths, transition_params=self.crf_transition_params)
        # # log_likelihood, transition_params = tfa.text.crf_log_likelihood(logits, labels, lengths)
        # loss = tf.reduce_mean(-log_likelihood)

        # self.crf_transition_params = transition_params

        return loss

    def score(self, logits, labels, lengths, scorer=None):
        mask = tf.sequence_mask(lengths, self.seq_len)
        true_labels = tf.boolean_mask(labels, mask)
        argmax = tf.math.argmax(logits, axis=-1)
        estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)

        p, r, f1 = scorer(estimated_labels.numpy(), true_labels.numpy())

        return p, r, f1


def estimate_crf_transitions(batches, n_tags):
    transitions = []
    for _, _, labels, lengths in batches:
        _, transition_params = tfa.text.crf_log_likelihood(tf.ones(shape=(labels.shape[0], labels.shape[1], n_tags)), labels, lengths)
        transitions.append(transition_params.numpy())

    return np.stack(transitions, axis=0).mean(axis=0)

# @tf.function
# def train_step(epoch_frac, model, optimizer, token_ids, prefix, suffix, graph_ids, labels, lengths, class_weights=None, scorer=None):
def train_step(model, optimizer, token_ids, prefix, suffix, target, mentions, graph_ids, labels, lengths,
                   class_weights=None, scorer=None, finetune=False):
    with tf.GradientTape() as tape:
        logits = model(token_ids, prefix, suffix, graph_ids, target, mentions, training=True)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth=logits.shape[-1]), logits, axis=-1))
        # p, r, f1 = scorer(tf.math.argmax(logits, axis=-1).numpy(), labels.reshape(-1,))
        loss = model.loss(logits, labels, lengths, class_weights=class_weights)
        p, r, f1 = model.score(logits, labels, lengths, scorer=scorer)
        gradients = tape.gradient(loss, model.trainable_variables)
        if not finetune:
            # do not update embeddings
            optimizer.apply_gradients((g, v) for g, v in zip(gradients, model.trainable_variables) if not v.name.startswith("default_embedder"))
        else:
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, p, r, f1


# @tf.function
def test_step(model, token_ids, prefix, suffix, graph_ids, target, mentions, labels, lengths, class_weights=None, scorer=None):
    logits = model(token_ids, prefix, suffix, graph_ids, target, mentions, training=False)
    # loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth=logits.shape[-1]), logits, axis=-1))
    # p, r, f1 = scorer(tf.math.argmax(logits, axis=-1).numpy(), labels.reshape(-1, ))
    loss = model.loss(logits, labels, lengths, class_weights=class_weights)
    p, r, f1 = model.score(logits, labels, lengths, scorer=scorer)

    return loss, p, r, f1


def train(model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01, learning_rate_decay=1., finetune=False):

    lr = tf.Variable(learning_rate, trainable=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_losses = []
    test_losses = []
    train_f1s = []
    test_f1s = []

    try:

        for e in range(epochs):
            losses = []
            ps = []
            rs = []
            f1s = []

            for ind, batch in enumerate(train_batches):
                # token_ids, graph_ids, labels, class_weights, lengths = b
                loss, p, r, f1 = train_step(model=model, optimizer=optimizer, token_ids=batch['tok_ids'],
                                            prefix=batch['prefix'], suffix=batch['suffix'],
                                            graph_ids=batch['graph_ids'],
                                            labels=batch['tags'],
                                            lengths=batch['lens'],
                                            target=batch['target'],
                                            mentions=batch['mentions'],
                                            # class_weights=batch['class_weights'],
                                            scorer=scorer,
                                            finetune=finetune and e/epochs > 0.6)
                losses.append(loss.numpy())
                ps.append(p)
                rs.append(r)
                f1s.append(f1)

            for ind, batch in enumerate(test_batches):
                # token_ids, graph_ids, labels, class_weights, lengths = b
                test_loss, test_p, test_r, test_f1 = test_step(model=model, token_ids=batch['tok_ids'],
                                                               prefix=batch['prefix'], suffix=batch['suffix'],
                                                               graph_ids=batch['graph_ids'],
                                                               labels=batch['tags'],
                                                               lengths=batch['lens'],
                                                               target=batch['target'],
                                                               mentions=batch['mentions'],
                                                               # class_weights=batch['class_weights'],
                                                               scorer=scorer)

            print(f"Epoch: {e}, Train Loss: {sum(losses) / len(losses)}, Train P: {sum(ps) / len(ps)}, Train R: {sum(rs) / len(rs)}, Train F1: {sum(f1s) / len(f1s)}, "
                  f"Test loss: {test_loss}, Test P: {test_p}, Test R: {test_r}, Test F1: {test_f1}")

            train_losses.append(float(sum(losses) / len(losses)))
            train_f1s.append(float(sum(f1s) / len(f1s)))
            test_losses.append(float(test_loss))
            test_f1s.append(float(test_f1))

            lr.assign(lr * learning_rate_decay)

    except KeyboardInterrupt:
        pass

    return train_losses, train_f1s, test_losses, test_f1s
