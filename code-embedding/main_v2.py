import tensorflow as tf
import sys
from Vocabulary import Vocabulary
from Reader import Reader
import os

import argparse

parser = argparse.ArgumentParser(description='Train skipgram embeddings')
parser.add_argument('train_corpus', type=str,
                    help='corpus with adjecency list')
parser.add_argument('--model', dest='model_name', default='./model/',
                    help='Path to model')
parser.add_argument('--voc', dest='voc', default='voc.pkl',
                    help='Path to vocabulary')
parser.add_argument('--export', action='store_true', default=False, help="Export weights")

args = parser.parse_args()


n_dims = 100
top_words = 20000
epochs = 20
n_contexts = 20  # 10 #20
k = 10  # 10 #20
window_size = 5  # 3 #5
# graph_saving_path = "./model/"
# ckpt_path = "./model/model.ckpt"
graph_saving_path = args.model_name
ckpt_path = os.path.join(graph_saving_path, "model.ckpt")
export = args.export

voc_path = args.voc

# if len(sxys.argv) != 2:
#     print("Provide input file for training")
#     sys.eit()

# data_path = sys.argv[1]
data_path = args.train_corpus


def assemble_graph(n_words, n_dims, learning_rate=0.001):
    """
    Assemble tensorflow graph to train word embeddings
    :param n_words: number of words in vocabulary
    :param n_dims: embedding dimensionality
    :return: dictionary with the following tensors:
    - in_words: placeholder for IN words
    - out_words: placeholder for OUT words
    - labels
    - loss
    - train
    - adder: count minibatches. Useful when restoring the session for checkpoint
    - assign_final: calculate final embeddings
    - in_out: return final embeddings
    """
    counter = tf.Variable(0, dtype=tf.int32)
    adder = tf.compat.v1.assign(counter, counter + 1)

    # embedding matrices
    in_matr = tf.compat.v1.get_variable("IN", shape=(n_words, n_dims), dtype=tf.float32)
    out_matr = tf.compat.v1.get_variable("OUT", shape=(n_words, n_dims), dtype=tf.float32)

    in_words = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,), name="in_words")
    out_words = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,), name="out_words")
    labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,), name="labels")

    in_emb = tf.nn.embedding_lookup(in_matr, in_words)
    out_emb = tf.nn.embedding_lookup(out_matr, out_words)

    logits = tf.reduce_sum(in_emb * out_emb, axis=1, name="inner_product")
    per_item_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

    loss = tf.reduce_mean(per_item_loss, axis=0)

    # train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    train = tf.contrib.opt.LazyAdamOptimizer(learning_rate).minimize(loss)

    final_emb = tf.compat.v1.get_variable("final", shape=(n_words, n_dims), dtype=tf.float32)
    calculate_final = tf.compat.v1.assign(final_emb, tf.nn.l2_normalize(in_matr + out_matr, axis=1))
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
if export:
    import pickle
    voc = pickle.load(open(voc_path, "rb"))
    print("Exporing...", time.asctime( time.localtime(time.time()) ))
else:
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

saver = tf.compat.v1.train.Saver()
saveloss_ = tf.compat.v1.summary.scalar('loss', loss_)

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

with tf.compat.v1.Session() as sess:
    
    if export:
        # Restore from checkpoint
        saver.restore(sess, ckpt_path)
        sess.graph.as_default()

        import numpy as np
        in_m = sess.run(terminals['in_matr'])
        np.savetxt("in_m.txt", in_m, delimiter="\t")
        out_m = sess.run(terminals['out_matr'])
        np.savetxt("out_m.txt", out_m, delimiter="\t")
        sys.exit()

    sess.run(tf.compat.v1.global_variables_initializer())
    summary_writer = tf.compat.v1.summary.FileWriter(graph_saving_path, graph=sess.graph)

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