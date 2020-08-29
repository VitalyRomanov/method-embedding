# import tensorflow as tf
import sys
# from gensim.models import Word2Vec
import numpy as np
from collections import Counter
# from scipy.linalg import toeplitz
# from gensim.models import KeyedVectors


def assemble_model_full(init_vectors, seq_len, n_tags, lr=0.001, train_embeddings=False):
    voc_size = init_vectors.shape[0]
    emb_dim = init_vectors.shape[1]

    d_win = 3
    h1_size = 500
    h2_size = 200
    pos_emb_size = 10

    with tf.variable_scope("positional_encodings") as pe:
        positions = list(range(seq_len * 2))
        position_splt = positions[:seq_len]
        position_splt.reverse()
        position_encoding = tf.constant(toeplitz(position_splt, positions[seq_len:]),
                                        dtype=tf.int32,
                                        name="position_encoding")

        position_embedding = tf.get_variable("position_embedding", shape=(seq_len * 2, pos_emb_size), dtype=tf.float32)

        pos_emb = tf.expand_dims(tf.nn.embedding_lookup(position_embedding, position_encoding, name="position_lookup"), axis=3)




    h1_kernel_shape = (d_win, emb_dim)

    tf_words = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="words")
    tf_labels = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="labels")
    tf_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name="lengths")

    def create_embedding_matrix():
        n_dims = init_vectors.shape[1]
        embs = tf.get_variable("embeddings", initializer=init_vectors, dtype=tf.float32, trainable=train_embeddings)
        pad = tf.zeros(shape=(1, n_dims), name="pad")
        emb_matr = tf.concat([embs, pad], axis=0)
        return emb_matr

    def convolutional_layer(input, units, cnn_kernel_shape, activation=None, name="conv_h1"):
        # padded = tf.pad(input, tf.constant([[0, 0], [1, 1], [0, 0]]))
        # emb_sent_exp = tf.expand_dims(input, axis=3)
        convolve = tf.layers.conv2d(input,
                                    units,
                                    cnn_kernel_shape,
                                    activation=activation,
                                    data_format='channels_last',
                                    name=name)
        return tf.reshape(convolve, shape=(-1, convolve.shape[1], units))

    emv_matr = create_embedding_matrix()

    emb_sent = tf.expand_dims(tf.nn.embedding_lookup(emv_matr, tf_words), axis=3)

    # with tf.variable_scope("input_dicing") as id:
    #     positions = []
    #     for i in range(seq_len):
    #         positions.append(tf.concat([
    #             emb_sent,
    #             tf.tile(tf.expand_dims(pos_emb[i, ...],
    #                            axis=0), [tf.shape(emb_sent)[0], 1, 1, 1])
    #         ], axis=2))

    # def cnn_pool(input_):
    #     local_h1 = convolutional_layer(input_, h1_size, h1_kernel_shape, activation=None)
    #     return tf.reduce_max(local_h1, axis=1)

    with tf.variable_scope("cnn_feature_extraction") as cnn:
        temp_cnn_emb = convolutional_layer(emb_sent, h1_size, h1_kernel_shape, activation=None)

        pos_cnn = convolutional_layer(pos_emb, h1_size, (d_win, pos_emb_size),
                                      activation=None, name="conv_pos")

        cnn_pool_feat = []
        for i in range(seq_len):
            # slice tensor for the line that corresponds to the current position in the sentence
            position_features = tf.expand_dims(pos_cnn[i,...], axis=0, name="exp_dim_%d" % i)
            # convolution without activation can be combined later, hence: temp_cnn_emb + position_features
            cnn_pool_feat.append(tf.expand_dims(tf.nn.tanh(tf.reduce_max(temp_cnn_emb + position_features, axis=1)), axis=1))

        cnn_pool_features = tf.concat(cnn_pool_feat, axis=1)

    with tf.variable_scope('dense_feature_extraction') as dfe:

        token_features = tf.reshape(cnn_pool_features, shape=(-1, h1_size))

        local_h2 = tf.layers.dense(token_features,
                                   h2_size,
                                   activation=tf.nn.tanh,
                                   name="dense_h2")

        tag_logits = tf.layers.dense(local_h2, n_tags, activation=None)
        logits = tf.reshape(tag_logits, (-1, seq_len, n_tags))

    with tf.variable_scope('loss') as l:
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, tf_labels, tf_lengths)
        loss = tf.reduce_mean(-log_likelihood)

    train = tf.train.AdamOptimizer(lr).minimize(loss)

    mask = tf.sequence_mask(tf_lengths, seq_len)
    true_labels = tf.boolean_mask(tf_labels, mask)
    argmax = tf.math.argmax(logits, axis=-1)
    estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)
    accuracy = tf.contrib.metrics.accuracy(estimated_labels, true_labels)

    return {
        'words': tf_words,
        'labels': tf_labels,
        'lengths': tf_lengths,
        'loss': loss,
        'train': train,
        'accuracy': accuracy,
        'argmax': argmax
    }

def assemble_model(init_vectors, seq_len, n_tags, lr=0.001, train_embeddings=False):
    voc_size = init_vectors.shape[0]
    emb_dim = init_vectors.shape[1]

    d_win = 5
    h1_size = 500
    h2_size = 200

    h1_kernel_shape = (d_win, emb_dim)

    tf_words = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="words")
    tf_labels = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="labels")
    tf_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name="lengths")

    def create_embedding_matrix():
        n_dims = init_vectors.shape[1]
        embs = tf.get_variable("embeddings", initializer=init_vectors, dtype=tf.float32, trainable=train_embeddings)
        pad = tf.zeros(shape=(1, n_dims), name="pad")
        emb_matr = tf.concat([embs, pad], axis=0)
        return emb_matr

    def convolutional_layer(input, units, cnn_kernel_shape, activation=None):
        padded = tf.pad(input, tf.constant([[0, 0], [2, 2], [0, 0]]))
        emb_sent_exp = tf.expand_dims(padded, axis=3)
        convolve = tf.layers.conv2d(emb_sent_exp,
                                    units,
                                    cnn_kernel_shape,
                                    activation=activation,
                                    data_format='channels_last',
                                    name="conv_h1")
        return tf.reshape(convolve, shape=(-1, convolve.shape[1], units))

    emv_matr = create_embedding_matrix()

    emb_sent = tf.nn.embedding_lookup(emv_matr, tf_words)

    conv_h1 = convolutional_layer(emb_sent, h1_size, h1_kernel_shape, tf.nn.tanh)

    token_features_1 = tf.reshape(conv_h1, shape=(-1, h1_size))

    local_h2 = tf.layers.dense(token_features_1,
                               h2_size,
                               activation=tf.nn.tanh,
                               name="dense_h2")

    tag_logits = tf.layers.dense(local_h2, n_tags, activation=None)
    logits = tf.reshape(tag_logits, (-1, seq_len, n_tags))

    with tf.variable_scope('loss') as l:
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, tf_labels, tf_lengths)
        loss = tf.reduce_mean(-log_likelihood)

    train = tf.train.AdamOptimizer(lr).minimize(loss)

    mask = tf.sequence_mask(tf_lengths, seq_len)
    true_labels = tf.boolean_mask(tf_labels, mask)
    argmax = tf.math.argmax(logits, axis=-1)
    estimated_labels = tf.cast(tf.boolean_mask(argmax, mask), tf.int32)
    accuracy = tf.contrib.metrics.accuracy(estimated_labels, true_labels)

    return {
        'words': tf_words,
        'labels': tf_labels,
        'lengths': tf_lengths,
        'loss': loss,
        'train': train,
        'accuracy': accuracy,
        'argmax': argmax
    }



# # sent_flattened = tf.maximum(local_sent_enh, axis=1)
# def prepare_data(sents, nlp):
#
#     sents_w = []
#     sents_t = []
#     # sents_w = [];
#     # sent_w = []
#     # sents_t = [];
#     # sent_t = []
#     # sents_c = [];
#     # sent_c = []
#     tags = set()
#     chunk_tags = set()
#
#     for s in sents:
#         doc = nlp(s[0])
#         sents_w.append([t.text for t in doc])
#
#         tags =
#
#     with open(path, "r") as conll:
#         for line in conll.read().strip().split("\n"):
#             if line == '':
#                 sents_w.append(sent_w)
#                 sent_w = []
#                 sents_t.append(sent_t)
#                 sent_t = []
#                 sents_c.append(sent_c)
#                 sent_c = []
#             else:
#                 try:
#                     word, tag, chunk = line.split()
#                 except:
#                     continue
#                 tags.add(tag)
#                 chunk_tags.add(chunk)
#                 sent_w.append(word.lower())
#                 sent_t.append(tag)
#                 sent_c.append(chunk)
#
#     tags = list(tags);
#     tags.sort()
#     chunk_tags = list(chunk_tags);
#     chunk_tags.sort()
#
#     tagmap = dict(zip(tags, range(len(tags))))
#     chunkmap = dict(zip(chunk_tags, range(len(chunk_tags))))
#     return sents_w, sents_t, sents_c, tags, tagmap, chunk_tags, chunkmap


def load_model(model_p):
    # model = Word2Vec.load(model_p)
    model = KeyedVectors.load_word2vec_format(model_p)
    voc_len = len(model.vocab)

    vectors = np.zeros((voc_len, 300), dtype=np.float32)

    w2i = dict()

    for ind, word in enumerate(model.vocab.keys()):
        w2i[word] = ind
        vectors[ind, :] = model[word]

    # w2i["*P*"] = len(w2i)

    return model, w2i, vectors


def create_batches(batch_size, seq_len, sents, repl, tags, graphmap, wordmap, tagmap):
    pad_id = len(wordmap)
    rpad_id = len(graphmap)
    n_sents = len(sents)

    b_sents = []
    b_repls = []
    b_tags = []
    b_lens = []

    for ind, (s, rr, tt)  in enumerate(zip(sents, repl, tags)):
        blank_s = np.ones((seq_len,), dtype=np.int32) * pad_id
        blank_r = np.ones((seq_len,), dtype=np.int32) * rpad_id
        blank_t = np.zeros((seq_len,), dtype=np.int32)

        int_sent = np.array([wordmap.get(w, pad_id) for w in s], dtype=np.int32)
        int_repl = np.array([graphmap.get(r, rpad_id) for r in rr], dtype=np.int32)
        int_tags = np.array([tagmap.get(t, 0) for t in tt], dtype=np.int32)

        blank_s[0:min(int_sent.size, seq_len)] = int_sent[0:min(int_sent.size, seq_len)]
        blank_r[0:min(int_sent.size, seq_len)] = int_repl[0:min(int_sent.size, seq_len)]
        blank_t[0:min(int_sent.size, seq_len)] = int_tags[0:min(int_sent.size, seq_len)]

        # print(int_sent[0:min(int_sent.size, seq_len)].shape)

        b_lens.append(len(s))
        b_sents.append(blank_s)
        b_repls.append(blank_r)
        b_tags.append(blank_t)

    lens = np.array(b_lens, dtype=np.int32)
    sentences = np.stack(b_sents)
    replacements = np.stack(b_repls)
    pos_tags = np.stack(b_tags)

    batch = []
    for i in range(n_sents // batch_size):
        batch.append((sentences[i * batch_size: i * batch_size + batch_size, :],
                      replacements[i * batch_size: i * batch_size + batch_size, :],
                      pos_tags[i * batch_size: i * batch_size + batch_size, :],
                      lens[i * batch_size: i * batch_size + batch_size]))

    return batch


# data_p = sys.argv[1]
# test_p = sys.argv[2]
# model_loc = sys.argv[3]
# target_task = sys.argv[4]
# epochs = int(sys.argv[5])
# gpu_mem = float(sys.argv[6])
# max_len = 40
#
# s_sents, s_tags, s_chunks, tagset, tagmap, chunk_tags, chunkmap = read_data(data_p)
# t_sents, t_tags, t_chunks, _, _, _, _ = read_data(test_p)
#
# if target_task == 'pos':
#     print("Choosing POS")
#     target = s_tags
#     t_map = tagmap
#     test = t_tags
# else:
#     print("Choosing Chunk")
#     target = s_chunks
#     t_map = chunkmap
#     test = t_chunks
#
# i_t_map = dict()
# for t, i in t_map.items():
#     i_t_map[i] = t
#
# print("Loading vectors")
# w2v_model, w2i, init_vectors = load_model(model_loc)
#
# print("Reading data")
# batches = create_batches(128, max_len, s_sents, target, w2i, t_map)
# test_batch = create_batches(len(t_sents), max_len, t_sents, test, w2i, t_map)[0]
#
# hold_out = test_batch
#
# print("Assembling model")
# terminals = assemble_model(init_vectors, max_len, len(t_map))
#
#
#
#
#
#
#
# print("Starting training")
# from tensorflow import GPUOptions
# gpu_options = GPUOptions(per_process_gpu_memory_fraction=gpu_mem)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     summary_writer = tf.summary.FileWriter("model/", graph=sess.graph)
#     for e in range(epochs):
#         for ind, batch in enumerate(batches):
#
#             sentences, pos_tags, lens = batch
#
#             sess.run(terminals['train'],{
#                 terminals['words']: sentences,
#                 terminals['labels']: pos_tags,
#                 terminals['lengths']: lens
#             })
#
#             if ind % 10 == 0:
#
#                 sentences, pos_tags, lens = hold_out
#
#                 loss_val, acc_val, am = sess.run([terminals['loss'], terminals['accuracy'], terminals['argmax']], {
#                     terminals['words']: sentences,
#                     terminals['labels']: pos_tags,
#                     terminals['lengths']: lens
#                 })
#
#         # print(t_sents[0])
#         # print(test[0])
#         # print([i_t_map[i] for i in am[0, :lens[0]]])
#
#         print("Epoch %d, loss %.4f, acc %.4f" % (e, loss_val, acc_val))
#
# # lens = map(lambda x: len(x), sents)
# # for w, c in Counter(lens).most_common():
# #     print(w,c)
# print(len(tagset))
# print(len(s_sents))
