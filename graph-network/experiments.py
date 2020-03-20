#%%
import pandas
from os.path import join

from sklearn.model_selection import train_test_split
import numpy as np

from graphtools import Embedder
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

        self.embed = pickle.load(open(join(self.base_path, "embeddings.pkl"), "rb"))[0]

        # e = pickle.load(open(join(self.base_path, "embeddings.pkl"), "rb"))
        #
        # self.embed = Embedder({0:0}, [])
        # self.embed.e = e.e
        # self.embed.ind = e.ind
        # self.embed.inv = e.inv

    def filter_valid(self, keys):
        return np.array([key for key in keys if key in self.embed.ind], dtype=np.int32)


    def __getitem__(self, type):
        nodes = pandas.read_csv(join(self.base_path, "nodes.bz2"))
        edges = pandas.read_csv(join(self.base_path, "held.bz2"))
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
                 embeddings,
                 nodes,
                 edges,
                 target):
        self.embed = embeddings
        self.nodes = nodes
        self.edges = edges
        self.target = target

        self.train_ind = None
        self.test_ind = None

        self.TEST_FRAC = 0.1
        self.K = 10 # how much more of negative samples should be in training data
        self.last_filtered = 0

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

        src_negative_ind = self.get_random_ind(src_size, num * 5)
        dst_negative_ind = self.get_random_ind(dst_size, num * 5)

        src_negative = self.filter_valid(src_set[src_negative_ind])
        dst_negative = self.filter_valid(dst_set[dst_negative_ind])

        while min(src_negative.size, dst_negative.size, num) != num:
            print(min(src_negative.size, dst_negative.size, num))
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

        train_negative = self.get_negative_edges(src_set, dst_set, self.train_ind.shape[0] * self.K)
        test_negative = self.get_negative_edges(src_set, dst_set, self.test_ind.shape[0])

        train_positive = self.target.iloc[self.train_ind].values
        test_positive = self.target.iloc[self.test_ind].values

        print(train_positive.shape, train_negative.shape, test_positive.shape, test_negative.shape)

        X_train = np.vstack([
            train_positive,
            train_negative
        ])

        X_test = np.vstack([
            test_positive,
            test_negative
        ])

        y_train = np.concatenate([np.ones((self.train_ind.shape[0],)), np.zeros((self.train_ind.shape[0] * self.K,))])
        y_test = np.concatenate([np.ones((self.test_ind.shape[0],)), np.zeros((self.test_ind.shape[0],))])

        return self._embed(X_train), self._embed(X_test), y_train, y_test

    def _embed(self, edges):
        src = edges[:,0]
        dst = edges[:,1]

        return np.hstack([self.embed[src], self.embed[dst]])

    def batches(self, size=256):
        X_train, X_test, y_train, y_test = self.get_training_data()
        for i in range(0, X_train.shape[0], size):
            if i + size >= X_train.shape[0]: continue
            yield self._embed(X_train[i: i+size]), y_train[i: i+size]




#%%

e = Experiments(base_path="/Users/LTV/GAT-2020-03-05-05-02-24-428772"
        ,api_seq_path="/Volumes/External/datasets/Code/source-graphs/python-source-graph/04_api_sequence_calls/flat_calls.csv")

experiment = e["apicall"]

X_train, X_test, y_train, y_test = experiment.get_training_data()

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#%%

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
#
# lr = LogisticRegression()
#
# lr.fit(X_train, y_train)
#
# print(pandas.DataFrame(classification_report(y_test, lr.predict(X_test), output_dict=True)))
#
# print(accuracy_score(y_test, lr.predict(X_test)))
#
# # print(test_positive_dst.size / (test_positive_dst.size + test_negative_dst.size))