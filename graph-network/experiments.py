#%%
import pandas
from os.path import join

from sklearn.model_selection import train_test_split
import numpy as np

# from graphtools import Embedder
from Embedder import Embedder
import pickle

class Experiments:
    def __init__(self,
                 base_path=None,
                 api_seq_path=None,
                 type_use_path=None,
                 node_type_path=None,
                 variable_use_path=None,
                 function_name_path=None):

        self.experiments = {
            'fcall': function_name_path,
            'apicall': api_seq_path,
            'typeuse': type_use_path,
            'varuse': variable_use_path,
            'fname': function_name_path
        }

        self.base_path = base_path

        if base_path is not None:
            self.embed = pickle.load(open(join(self.base_path, "embeddings.pkl"), "rb"))[2]

        # e = pickle.load(open(join(self.base_path, "embeddings.pkl"), "rb"))
        #
        # self.embed = Embedder({0:0}, [])
        # self.embed.e = e.e
        # self.embed.ind = e.ind
        # self.embed.inv = e.inv

    def filter_valid(self, keys):
        return np.array([key for key in keys if key in self.embed.ind], dtype=np.int32)


    def __getitem__(self, type):
        nodes = pandas.read_csv(join(self.base_path, "nodes.csv"))
        edges = pandas.read_csv(join(self.base_path, "held.csv"))
        if type == "link":
            nodes = pandas.read_csv(join(self.base_path, "nodes.csv"))
            held = pandas.read_csv(join(self.base_path, "held.csv"))

            held = held.filter('type == 8')

            return nodes, edges, held

        elif type == "apicall":
            api_seq = pandas.read_csv(self.experiments['apicall'])

            unique_nodes = set(nodes['id'].values.tolist())

            api_seq = api_seq[
                api_seq['src'].apply(lambda nid: nid in unique_nodes)
            ]

            api_seq = api_seq[
                api_seq['dst'].apply(lambda nid: nid in unique_nodes)
            ]

            return Experiment(self.embed, nodes, edges, api_seq)

        elif type == "typeuse":
            held = pandas.read_csv(join(self.base_path), "held.csv")

            held = held.filter('type == 2')

            return Experiment(self.embed, nodes, edges, held)

        elif type == "varuse":
            var_use = pandas.read_csv(self.experiments['varuse'])

            unique_nodes = set(nodes['id'].values.tolist())

            var_use = var_use[
                var_use['src'].apply(lambda nid: nid in unique_nodes)
            ]

            return nodes, edges, var_use

        elif type == "fname":

            fname = pandas.read_csv(self.experiments['fname'])

            unique_nodes = set(nodes['id'].values.tolist())

            fname = fname[
                fname['src'].apply(lambda nid: nid in unique_nodes)
            ]

            return nodes, edges, fname

class Experiment:
    def __init__(self,
                 embeddings: Embedder,
                 nodes: pandas.DataFrame,
                 edges: pandas.DataFrame,
                 target: pandas.DataFrame):

        self.embed = embeddings
        self.nodes = nodes
        self.edges = edges
        self.target = target

        self.train_ind = None
        self.test_ind = None

        self.TEST_FRAC = 0.1
        self.K = 10 # how much more of negative samples should be in training data
        self.last_filtered = 0

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.embed_size = self.embed.e.shape[1]
        self.n_classes = target['dst'].unique().size

    def get_train_test_split(self):
        self.train_ind, self.test_ind = train_test_split(
            np.arange(start=0, stop=self.target.shape[0]),
            test_size=self.TEST_FRAC,
            random_state=42
        )

    def filter_valid(self, keys):
        return keys
        # filtered = np.array([key for key in keys if key in self.embed.ind], dtype=np.int32)
        filtered = np.array(list(filter(lambda key: key in self.embed.ind, keys)), dtype=np.int32)
        self.last_filtered = keys.size - filtered
        return filtered

    def get_random_ind(self, set_size, num):
        return np.random.randint(low=0, high=set_size, size=num)

    def get_negative_edges(self, src_set, dst_set, num):
        src_size = src_set.size
        dst_size = dst_set.size

        src_negative_ind = self.get_random_ind(src_size, num) # * 5)
        dst_negative_ind = self.get_random_ind(dst_size, num) # * 5)

        src_negative = self.filter_valid(src_set[src_negative_ind])
        dst_negative = self.filter_valid(dst_set[dst_negative_ind])

        while min(src_negative.size, dst_negative.size, num) != num:
            # print(min(src_negative.size, dst_negative.size, num))
            src_negative_ind = self.get_random_ind(src_size, num)
            dst_negative_ind = self.get_random_ind(dst_size, num)

            src_negative = np.concatenate([src_negative, self.filter_valid(src_set[src_negative_ind])])
            dst_negative = np.concatenate([dst_negative, self.filter_valid(dst_set[dst_negative_ind])])


        src_negative = src_negative[:num]
        dst_negative = dst_negative[:num]

        negative_edges = np.hstack([src_negative.reshape(-1,1), dst_negative.reshape(-1,1)])
        return negative_edges


    def get_training_data(self):
        self.get_train_test_split()

        src_set = self.target['src'].values
        dst_set = self.target['dst'].values

        train_negative = self.get_negative_edges(src_set, dst_set, self.train_ind.shape[0]) # * self.K)
        test_negative = self.get_negative_edges(src_set, dst_set, self.test_ind.shape[0])

        train_positive = self.target.iloc[self.train_ind].values
        test_positive = self.target.iloc[self.test_ind].values

        # print(train_positive.shape, train_negative.shape, test_positive.shape, test_negative.shape)

        X_train = np.vstack([
            train_positive,
            train_negative
        ])

        X_test = np.vstack([
            test_positive,
            test_negative
        ])

        y_train = np.concatenate([np.ones((self.train_ind.shape[0],)), np.zeros((self.train_ind.shape[0]),)]) # self.train_ind.shape[0]) * self.K
        y_test = np.concatenate([np.ones((self.test_ind.shape[0],)), np.zeros((self.test_ind.shape[0],))])

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        def shuffle(X, y):
            ind_shuffle = np.arange(0, X.shape[0])
            np.random.shuffle(ind_shuffle)
            return X[ind_shuffle], y[ind_shuffle]

        self.X_train, self.y_train = shuffle(X_train, y_train)
        self.X_test, self.y_test = shuffle(X_test, y_test)

        # return X_train, X_test, y_train, y_test

    def _embed(self, edges):
        src = edges[:,0]
        dst = edges[:,1]

        return np.hstack([self.embed[src], self.embed[dst]])

    def batch(self, X, y, size=256):

        def encode_binary(y):
            y_encoded = np.zeros((y.shape[0], 2))
            y_encoded[:, y.astype(np.int32)] = 1
            return y_encoded

        for i in range(0, X.shape[0], size):
            if i + size >= X.shape[0]: continue

            X_b = self._embed(X[i: i+size])
            # y_b = encode_binary(y[i: i+size])
            y_b = y[i: i+size]

            # TODO
            # dimensionality is wrong

            assert y_b.shape[0] == X_b.shape[0]
            yield X_b, y_b
            yield np.ones((10,10)), y_b

    def test_batches(self):
        if self.X_test is None:
            self.get_training_data()

        return self.batch(self.X_test, self.y_test)

    def train_batches(self):
        if self.X_train is None:
            self.get_training_data()

        return self.batch(self.X_train, self.y_train)



#%%

BASE_PATH = "/home/ltv/data/local_run/graph-network/models/GAT-2020-03-23-10-07-17-549418"
# API_SEARCH = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/04_api_sequence_calls/flat_calls.csv"
API_SEQ = "/home/ltv/data/datasets/source_code/python-source-graph/04_api_sequence_calls/flat_calls.csv"

e = Experiments(base_path=BASE_PATH,
                api_seq_path=API_SEQ,
                type_use_path=None,
                node_type_path=None,
                variable_use_path=None,
                function_name_path=None
                )

experiment = e["apicall"]

#%%
####################################################################
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
#
# lr = LogisticRegression(max_iter=1000)
#
# experiment.get_training_data()
# X_train, y_train = experiment._embed(experiment.X_train), experiment.y_train
# X_test, y_test = experiment._embed(experiment.X_test), experiment.y_test
#
# lr.fit(X_train, y_train)
#
# print(pandas.DataFrame(classification_report(y_test, lr.predict(X_test), output_dict=True)))
#
# print(accuracy_score(y_test, lr.predict(X_test)))
#
# # print(test_positive_dst.size / (test_positive_dst.size + test_negative_dst.size))

#####################################################################

from classifiers import LRClassifier, NNClassifier
import tensorflow as tf

clf = LRClassifier(experiment.embed_size)

# clf.compile(optimizer='adam',
#             loss='sparse_categorical_crossentropy')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = clf(images, training=True)
    # print(": )", labels.shape, predictions.shape)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, clf.trainable_variables)
  optimizer.apply_gradients(zip(gradients, clf.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = clf(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()


  for X, y in experiment.train_batches():
    print(X.shape, y.shape)
    train_step(X, y)

  for X, y in experiment.test_batches():
    test_step(X, y)

  # print(clf.count_params())

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

