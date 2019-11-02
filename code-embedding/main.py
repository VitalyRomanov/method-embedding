import tensorflow as tf
import sys
from Vocabulary import Vocabulary
from Reader import Reader


n_dims = 100
top_words = 20000
epochs = 20
n_contexts = 20  # 10 #20
k = 10  # 10 #20
window_size = 5  # 3 #5
graph_saving_path = "./model/"
ckpt_path = "./model/model.ckpt"

# if len(sxys.argv) != 2:
#     print("Provide input file for training")
#     sys.eit()

data_path = "../method-embeddings/method_callee.txt" #sys.argv[1]


def assemble_graph(n_words, n_dims, learning_rate=0.001):
    """
    Assemble tensorflow graph to train word embeddigns
    :param n_words: number of words in vocabulary
    :param n_dims: embeddign dimensionality
    :return: dictionary with the following tensors:
    - in_words: placeholder for IN words
    - out_words: placeholder for OUT words
    - labels
    - loss
    - train
    - adder: count minibatches. Useful when restoring the session for checkpoint
    - assign_final: calculate final embeddings
    - in_out: return final embeddigns
    """
    counter = tf.Variable(0, dtype=tf.int32)
    adder = tf.assign(counter, counter + 1)

    # embedding matrices
    in_matr = tf.get_variable("IN", shape=(n_words, n_dims), dtype=tf.float32)
    out_matr = tf.get_variable("OUT", shape=(n_words, n_dims), dtype=tf.float32)

    in_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="in_words")
    out_words = tf.placeholder(dtype=tf.int32, shape=(None,), name="out_words")
    labels = tf.placeholder(dtype=tf.float32, shape=(None,), name="labels")

    in_emb = tf.nn.embedding_lookup(in_matr, in_words)
    out_emb = tf.nn.embedding_lookup(out_matr, out_words)

    logits = tf.reduce_sum(in_emb * out_emb, axis=1, name="inner_product")
    per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_mean(per_item_loss, axis=0)

    # train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

    final_emb = tf.get_variable("final", shape=(n_words, n_dims), dtype=tf.float32)
    calculate_final = tf.assign(final_emb, tf.nn.l2_normalize(in_matr + out_matr, axis=1))
    in_out = tf.nn.embedding_lookup(calculate_final, in_words)

    return {
        'in_words': in_words,
        'out_words': out_words,
        'labels': labels,
        'loss': loss,
        'train': train,
        'adder': adder,
        'assign_final': calculate_final,
        'in_out': in_out,
        'in_matr': in_matr,
        'out_matr': out_matr
    }



import time
print("Reading data", time.asctime( time.localtime(time.time()) ))

# Estimate vocabulary from training data
# voc = pickle.load(open("voc.pkl", "rb"))
voc = Vocabulary()
with open(data_path, "r") as data:
    line = data.readline()
    while line:
        tokens = line.strip().split()
        voc.add_words(tokens)
        line = data.readline()
voc.prune(top_words)
voc.export_vocabulary(top_words, "voc.tsv")
voc.save("voc.pkl")

print("Starting training", time.asctime( time.localtime(time.time()) ))

reader = Reader(data_path, voc, n_contexts, window_size, k)

terminals = assemble_graph(top_words, n_dims)

first_batch = None

in_words_ = terminals['in_words']
out_words_ = terminals['out_words']
labels_ = terminals['labels']
train_ = terminals['train']
loss_ = terminals['loss']
adder_ = terminals['adder']
final_ = terminals['assign_final']

saver = tf.train.Saver()
saveloss_ = tf.summary.scalar('loss', loss_)

# batch = reader.next_batch(top_n_for_sampling=top_words)
# while batch is not None:
#     # for a, p, l in zip(batch[0].tolist(), batch[1].tolist(), batch[2].tolist()):
#     #     print(voc.id2word[a], voc.id2word[p], l)
#     batch = reader.next_batch(top_n_for_sampling=top_words)
#     print(time.asctime( time.localtime(time.time()) ))
#
# batch = reader.next_batch(top_n_for_sampling=top_words)
# # for a, p, l in zip(batch[0].tolist(), batch[1].tolist(), batch[2].tolist()):
# #     print(voc.id2word[a], voc.id2word[p], l)
# print(time.asctime( time.localtime(time.time()) ))
#
# sys.exit()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(graph_saving_path, graph=sess.graph)

    # Restore from checkpoint
    # saver.restore(sess, ckpt_path)
    # sess.graph.as_default()

    # import numpy as np
    # in_m = sess.run(terminals['in_matr'])
    # np.savetxt("in_m.txt", in_m, delimiter="\t")
    # out_m = sess.run(terminals['out_matr'])
    # np.savetxt("out_m.txt", out_m, delimiter="\t")

    # sys.exit()

    for e in range(epochs):
        # batch = reader.next_batch()
        # first_batch = batch

        for ind, batch in enumerate(reader.batches()):

            in_words, out_words, labels = batch

            _, batch_count = sess.run([train_, adder_], {
                in_words_: in_words,
                out_words_: out_words,
                labels_: labels
            })

            if batch_count % 1000 == 0:
                # in_words, out_words, labels = first_batch
                loss_val, summary, _ = sess.run([loss_, saveloss_, final_], {
                    in_words_: in_words,
                    out_words_: out_words,
                    labels_: labels
                })
                print("Epoch {}, batch {}, loss {}".format(e, batch_count, loss_val))
                save_path = saver.save(sess, ckpt_path)
                summary_writer.add_summary(summary, batch_count)

            # batch = reader.next_batch()

print("Finished trainig", time.asctime( time.localtime(time.time()) ))