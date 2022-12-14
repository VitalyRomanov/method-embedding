import logging

import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from SourceCodeTools.mltools.tensorflow import to_numpy
from SourceCodeTools.models.nlp.TFDecoder import FlatDecoder
from SourceCodeTools.models.nlp.TFEncoder import DefaultEmbedding, TextCnnEncoder


class T5Encoder(Model):
    def __init__(self):
        super(T5Encoder, self).__init__()
        from transformers.models.t5.modeling_tf_t5 import T5Config, TFT5MainLayer
        self.adapter = Dense(40, activation=tf.nn.relu)
        self.t5_config = T5Config(d_model=40, d_ff=40, num_layers=1, num_decoder_layers=1, num_heads=1, is_encoder_decoder=False,d_kv=40)
        self.encoder = TFT5MainLayer(self.t5_config,)

    def call(self, embs, training=True, mask=None):
        from transformers.modeling_tf_utils import input_processing
        inputs = input_processing(
            func=self.call,
            config=self.t5_config,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=self.adapter(embs),
            decoder_inputs_embeds=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            training=training,
            kwargs_call={},
        )
        encoded = self.encoder(inputs["inputs"], inputs_embeds=inputs["inputs_embeds"], training=training)
        return encoded["last_hidden_state"]


class TypePredictor(Model):
    """
    TypePredictor model predicts types for Python functions using the following inputs
    Tokens: FastText embeddings for tokens trained on a large collection of texts
    Graph: Graph embeddings pretrained with GNN model
    Prefix: Embeddings for the first n characters of a token
    Suffix: Embeddings for the last n characters of a token
    """
    def __init__(
            self, tok_embedder, graph_embedder, train_embeddings=False, h_sizes=None, dense_size=100, num_classes=None,
            seq_len=100, pos_emb_size=30, cnn_win_size=3, crf_transitions=None, suffix_prefix_dims=50,
            suffix_prefix_buckets=1000, no_graph=False, **kwargs
    ):
        """
        Initialize TypePredictor. Model initializes embedding layers and then passes embeddings to TextCnnEncoder model
        :param tok_embedder: Embedder for tokens
        :param graph_embedder: Embedder for graph nodes
        :param train_embeddings: whether to finetune embeddings
        :param h_sizes: hiddenlayer sizes
        :param dense_size: sizes of dense layers
        :param num_classes: number of output classes
        :param seq_len: maximum length of sentences
        :param pos_emb_size: dimensionality of positional embeddings
        :param cnn_win_size: width of cnn window
        :param crf_transitions: CRF transition probabilities
        :param suffix_prefix_dims: dimensionality of suffix and prefix embeddings
        :param suffix_prefix_buckets: number of suffix and prefix embeddings
        """
        super(TypePredictor, self).__init__()

        if h_sizes is None:
            h_sizes = [500]

        assert num_classes is not None, "set num_classes"

        self.seq_len = seq_len
        self.transition_params = crf_transitions

        # initialize embeddings
        with tf.device('/CPU:0'):
            self.tok_emb = DefaultEmbedding(init_vectors=tok_embedder.e, trainable=train_embeddings)
            logging.warning("Not finetuning graph embeddings")
            if graph_embedder is not None:
                self.graph_emb = DefaultEmbedding(init_vectors=graph_embedder.e, trainable=False)  #    train_embeddings)
            else:
                self.graph_emb = None
        self.prefix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))
        self.suffix_emb = DefaultEmbedding(shape=(suffix_prefix_buckets, suffix_prefix_dims))

        # compute final embedding size after concatenation
        input_dim = tok_embedder.e.shape[1] + suffix_prefix_dims * 2 #+ graph_embedder.e.shape[1]

        if cnn_win_size % 2 == 0:
            cnn_win_size += 1
            logging.info(f"Window size should be odd. Setting to {cnn_win_size}")

        self.encoder = TextCnnEncoder(
            # input_size=input_dim,
            h_sizes=h_sizes,
            seq_len=seq_len, pos_emb_size=pos_emb_size,
            cnn_win_size=cnn_win_size, dense_size=dense_size,
            out_dim=input_dim, activation=tf.nn.relu,
            dense_activation=tf.nn.tanh)
        # self.encoder = GRUEncoder(input_dim=input_dim, out_dim=input_dim, num_layers=1, dropout=0.1)
        # self.encoder = T5Encoder()

        # self.decoder = ConditionalAttentionDecoder(
        #     input_dim, out_dim=num_classes, num_layers=1, num_heads=1,
        #     ff_hidden=100, target_vocab_size=num_classes, maximum_position_encoding=self.seq_len
        # )
        self.decoder = FlatDecoder(out_dims=num_classes)

        self.crf_transition_params = None

        self.supports_masking = True
        self.use_graph = not no_graph

    # @tf.function
    def __call__(self, token_ids, prefix_ids, suffix_ids, graph_ids, graph_embs=None, target=None, training=False, mask=None):
        """
        # Inference
        :param token_ids: ids for tokens, shape (?, seq_len)
        :param prefix_ids: ids for prefixes, shape (?, seq_len)
        :param suffix_ids: ids for suffixes, shape (?, seq_len)
        :param graph_ids: ids for graph nodes, shape (?, seq_len)
        :param training: whether to finetune embeddings
        :return: logits for token classes, shape (?, seq_len, num_classes)
        """
        assert mask is not None, "Mask is required"

        tok_emb = self.tok_emb(token_ids)
        prefix_emb = self.prefix_emb(prefix_ids)
        suffix_emb = self.suffix_emb(suffix_ids)

        embs = tf.concat([tok_emb,
                          # graph_emb,
                          prefix_emb,
                          suffix_emb], axis=-1)

        encoded = self.encoder(embs, training=training, mask=mask)
        # if target is None:
        #     logits = self.decoder.seq_decode(encoded, training=training, mask=mask)
        # else:
        if self.use_graph:
            assert graph_ids is not None, "Make sure you have provided graph embeddings"
            if graph_embs is None:
                graph_embs = self.graph_emb(graph_ids)
            encoded = tf.concat([encoded, graph_embs], axis=-1)
        logits, _ = self.decoder((encoded, target), training=training, mask=mask) # consider sending input instead of target

        return logits

    def compute_mask(self, inputs, mask=None):
        mask = self.encoder.compute_mask(None, mask=mask)
        return self.decoder.compute_mask(None, mask=mask)

    def loss(self, logits, labels, mask, class_weights=None, extra_mask=None):
        """
        Compute cross-entropy loss for each meaningful tokens. Mask padded tokens.
        :param logits: shape (?, seq_len, num_classes)
        :param labels: ids of labels, shape (?, seq_len)
        :param lengths: actual sequence lenghts, shape (?,)
        :param class_weights: optionally provide weights for each token, shape (?, seq_len)
        :param extra_mask: mask for hiding some of the token labels, not counting them towards the loss, shape (?, seq_len)
        :return: average cross-entropy loss
        """
        losses = tf.nn.softmax_cross_entropy_with_logits(tf.one_hot(labels, depth=logits.shape[-1]), logits, axis=-1)
        seq_mask = mask # logits._keras_mask# tf.sequence_mask(lengths, self.seq_len)
        if extra_mask is not None:
            seq_mask = tf.math.logical_and(seq_mask, extra_mask)
        if class_weights is None:
            loss = tf.reduce_mean(tf.boolean_mask(losses, seq_mask))
        else:
            loss = tf.reduce_mean(tf.boolean_mask(losses * class_weights, seq_mask))

        # log_likelihood, transition_params = tfa.text.crf_log_likelihood(logits, labels, lengths, transition_params=self.crf_transition_params)
        # # log_likelihood, transition_params = tfa.text.crf_log_likelihood(logits, labels, lengths)
        # loss = tf.reduce_mean(-log_likelihood)

        # self.crf_transition_params = transition_params

        return loss

    def score(self, logits, labels, mask, scorer=None, extra_mask=None):
        """
        Compute precision, recall and f1 scores using the provided scorer function
        :param logits: shape (?, seq_len, num_classes)
        :param labels: ids of token labels, shape (?, seq_len)
        :param lengths: tensor of actual sentence lengths, shape (?,)
        :param scorer: scorer function, takes `pred_labels` and `true_labels` as arguments
        :param extra_mask: mask for hiding some of the token labels, not counting them towards the score, shape (?, seq_len)
        :return:
        """
        # mask = logits._keras_mask # tf.sequence_mask(lengths, self.seq_len)
        if extra_mask is not None:
            mask = tf.math.logical_and(mask, extra_mask)
        true_labels = tf.boolean_mask(labels, mask)
        argmax = tf.math.argmax(logits, axis=-1)
        estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)

        p, r, f1 = scorer(to_numpy(estimated_labels), to_numpy(true_labels))

        scores = {}
        scores["Precision"] = p
        scores["Recall"] = r
        scores["F1"] = f1

        return scores
