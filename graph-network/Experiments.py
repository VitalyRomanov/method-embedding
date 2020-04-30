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
            # self.embed.e = np.random.rand(self.embed.e.shape[0], self.embed.e.shape[1])

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
            # nodes = pandas.read_csv(join(self.base_path, "nodes.csv"))
            held = pandas.read_csv(join(self.base_path, "held.csv"))

            held = held.query('type == 8')[['src', 'dst']]

            return Experiment(self.embed, nodes, edges, held)

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
            held = pandas.read_csv(join(self.base_path, "held.csv"))

            held = held.query('type == 2')[['src', 'dst']]

            return Experiment(self.embed, nodes, edges, held)

        elif type == "varuse":
            var_use = pandas.read_csv(self.experiments['varuse'])

            unique_nodes = set(nodes['id'].values.tolist())

            var_use = var_use[
                var_use['src'].apply(lambda nid: nid in unique_nodes)
            ]

            return Experiment2(self.embed, nodes, edges, var_use)

        elif type == "fname":

            # fname = pandas.read_csv(self.experiments['fname'])
            functions = nodes.query('label == 4096')
            functions['fname'] = functions['name'].apply(lambda name: name.split(".")[-1])

            functions['src'] = functions['id']
            functions['dst'] = functions['fname']

            # unique_nodes = set(nodes['id'].values.tolist())
            #
            # fname = fname[
            #     fname['src'].apply(lambda nid: nid in unique_nodes)
            # ]

            return Experiment2(self.embed, nodes, edges, functions[['src', 'dst']])

        elif type == "nodetype":

            types = nodes.copy()
            types['src'] = nodes['id']
            types['dst'] = nodes['label']

            return Experiment3(self.embed, nodes, edges, types[['src', 'dst']])


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
        # self.n_classes = target['dst'].unique().size

        self.unique_src = self.target['src'].unique()
        self.unique_dst = self.target['dst'].unique()

        counts = self.target['dst'].value_counts(normalize=True)
        freq = counts.values ** 3/4
        self.freq = freq / sum(freq)
        self.dst_idxs = counts.index
        self.dst_neg_sampling = lambda size: np.random.choice(self.dst_idxs, size, replace=True, p=self.freq)


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
        # self.last_filtered = keys.size - filtered
        # return filtered

    def get_random_ind(self, set_size, num):
        return np.random.randint(low=0, high=set_size, size=num)

    def get_negative_edges(self, src_set, dst_set, num):
        src_size = src_set.size
        dst_size = dst_set.size

        # TODO
        # try word2vec like unigram sampling

        src_negative_ind = self.get_random_ind(src_size, num) # * 5)
        dst_negative_ind = self.get_random_ind(dst_size, num) # * 5)

        src_negative = self.filter_valid(src_set[src_negative_ind])
        dst_negative = self.filter_valid(dst_set[dst_negative_ind])
        # dst_negative = self.dst_neg_sampling(num)

        # while min(src_negative.size, dst_negative.size, num) != num:
        #     # print(min(src_negative.size, dst_negative.size, num))
        #     src_negative_ind = self.get_random_ind(src_size, num)
        #     dst_negative_ind = self.get_random_ind(dst_size, num)
        #
        #     src_negative = np.concatenate([src_negative, self.filter_valid(src_set[src_negative_ind])])
        #     dst_negative = np.concatenate([dst_negative, self.filter_valid(dst_set[dst_negative_ind])])


        # src_negative = src_negative[:num]
        # dst_negative = dst_negative[:num]

        negative_edges = np.hstack([src_negative.reshape(-1,1), dst_negative.reshape(-1,1)])
        return negative_edges


    def get_training_data(self):
        self.get_train_test_split()

        src_set = self.target['src'].values
        dst_set = self.target['dst'].values

        # TODO
        # number of negative samples is currently the same as positive

        # train_negative = self.get_negative_edges(src_set, dst_set, self.train_ind.shape[0]) # * self.K)
        # test_negative = self.get_negative_edges(src_set, dst_set, self.test_ind.shape[0])

        train_positive = self.target.iloc[self.train_ind].values
        test_positive = self.target.iloc[self.test_ind].values

        # # print(train_positive.shape, train_negative.shape, test_positive.shape, test_negative.shape)
        # print(f"Working with {train_positive.shape[0]} positive and {train_negative.shape[0]} negative samples in the train set, {test_positive.shape[0]} and {test_negative.shape[0]} - in test set")

        X_train = train_positive
        X_test = test_positive

        y_train = np.ones((self.train_ind.shape[0],))
        y_test = np.ones((self.test_ind.shape[0],))

        # X_train = np.vstack([
        #     train_positive,
        #     train_negative
        # ])

        # X_test = np.vstack([
        #     test_positive,
        #     test_negative
        # ])

        # y_train = np.concatenate([np.ones((self.train_ind.shape[0],)), np.zeros((self.train_ind.shape[0]),)]) # self.train_ind.shape[0]) * self.K
        # y_test = np.concatenate([np.ones((self.test_ind.shape[0],)), np.zeros((self.test_ind.shape[0],))])

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        def shuffle(X, y):
            ind_shuffle = np.arange(0, X.shape[0])
            np.random.shuffle(ind_shuffle)
            return X[ind_shuffle], y[ind_shuffle]

        self.X_train, self.y_train = shuffle(X_train, y_train)
        self.X_test, self.y_test = shuffle(X_test, y_test)

        print(f"Splitting into {self.X_train.shape[0]} train and {self.X_test.shape[0]} test samples")

        # return X_train, X_test, y_train, y_test

    def _embed(self, edges):
        src = edges[:,0]
        dst = edges[:,1]

        return np.hstack([self.embed[src], self.embed[dst]])

    def batch(self, X, y, size=256, K=15):

        def encode_binary(y):
            y_encoded = np.zeros((y.shape[0], 2))
            y_encoded[:, y.astype(np.int32)] = 1
            return y_encoded

        for i in range(0, X.shape[0], size):
            if i + size >= X.shape[0]: continue

            neg = self.get_negative_edges(X[i: i + size, 0], self.unique_dst, size * K)
            X_b = np.vstack([X[i: i+size], neg])

            X_b_e = self._embed(X_b)
            # y_b = encode_binary(y[i: i+size])
            y_b = np.concatenate([y[i: i+size], np.zeros(size * K,)])


            assert y_b.shape[0] == X_b.shape[0] == size * (1 + K)
            yield {"x": X_b_e, "y": y_b}
            # yield np.ones((10,10)), y_b

        last_piece = X[X.shape[0] // size * size:, :]
        neg = self.get_negative_edges(last_piece[:, 0], self.unique_dst, last_piece.shape[0] * K)
        X_b = np.vstack([last_piece, neg])
        X_b_e = self._embed(X_b)
        y_b = np.concatenate([y[X.shape[0] // size * size:], np.zeros(last_piece.shape[0] * K, )])

        assert y_b.shape[0] == X_b.shape[0]

        yield {"x": X_b_e, "y": y_b}

    def test_batches(self):
        if self.X_test is None:
            self.get_training_data()

        return self.batch(self.X_test, self.y_test, K = 1)

    def train_batches(self):
        if self.X_train is None:
            self.get_training_data()

        return self.batch(self.X_train, self.y_train, K = 1)


def compact_property(values):
    uniq = np.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    return prop2pid

class Experiment2(Experiment):
    def __init__(self, embeddings: Embedder,
                 nodes: pandas.DataFrame,
                 edges: pandas.DataFrame,
                 target: pandas.DataFrame):
        super(Experiment2, self).__init__(embeddings, nodes, edges, target)

        # def compact_property(values):
        #     uniq = np.unique(values)
        #     prop2pid = dict(zip(uniq, range(uniq.size)))
        #     return prop2pid

        self.name_map = compact_property(target['dst'])
        self.dst_orig = target['dst']
        target['dst'] = target['dst'].apply(lambda name: self.name_map[name])

        self.unique_elements = len(self.name_map)

        print(f"Doing experiment with {len(self.name_map)} distinct target elements")

        self.unique_src = self.target['src'].unique()
        self.unique_dst = self.target['dst'].unique()

    def _embed(self, edges):
        pass
        # src = edges[:,0]
        # dst = edges[:,1]
        #
        # return np.hstack([self.embed[src], self.embed[dst]])

    def batch(self, X, y, size=256, K=15):
        for i in range(0, X.shape[0], size):
            if i + size >= X.shape[0]: continue

            neg = self.get_negative_edges(X[i: i + size, 0], self.unique_dst, size * K)
            X_ = np.vstack([X[i: i+size], neg])
            src = X_[:, 0]
            dst = X_[:, 1]

            # src = X[i: i+size, 0]
            # dst = X[i: i+size, 1]

            X_src = self.embed[src]

            # y_b = y[i: i+size]
            y_b = np.concatenate([y[i: i + size], np.zeros(size * K, )])

            assert y_b.shape[0] == X_src.shape[0] == dst.shape[0]
            yield {"x": X_src, "elements": dst, "y": y_b}

        last_piece = X[X.shape[0] // size * size:]
        neg = self.get_negative_edges(last_piece[:, 0], self.unique_dst, last_piece.shape[0] * K)
        X_ = np.vstack([last_piece, neg])
        src = X_[:, 0]
        dst = X_[:, 1]

        # src = X[X.shape[0] // size * size:, 0]
        # dst = X[X.shape[0] // size * size:, 1]
        X_src = self.embed[src]
        # y_b = y[X.shape[0] // size * size:]
        y_b = np.concatenate([y[X.shape[0] // size * size:], np.zeros(last_piece.shape[0] * K, )])
        assert y_b.shape[0] == X_src.shape[0] == dst.shape[0]
        yield {"x": X_src, "elements": dst, "y": y_b}


class Experiment3(Experiment2):
    def __init__(self, embeddings: Embedder,
                 nodes: pandas.DataFrame,
                 edges: pandas.DataFrame,
                 target: pandas.DataFrame):
        super(Experiment3, self).__init__(embeddings, nodes, edges, target)

    def get_training_data(self):

        self.get_train_test_split()

        train_positive = self.target.iloc[self.train_ind].values
        test_positive = self.target.iloc[self.test_ind].values

        X_train, y_train = train_positive[:,0].reshape(-1,1), train_positive[:,1].reshape(-1,1)
        X_test, y_test = test_positive[:, 0].reshape(-1,1), test_positive[:, 1].reshape(-1,1)

        def shuffle(X, y):
            ind_shuffle = np.arange(0, X.shape[0])
            np.random.shuffle(ind_shuffle)
            return X[ind_shuffle], y[ind_shuffle]

        self.X_train, self.y_train = shuffle(X_train, y_train)
        self.X_test, self.y_test = shuffle(X_test, y_test)


    def batch(self, X, y, size=256, **kwargs):
        for i in range(0, X.shape[0], size):
            if i + size >= X.shape[0]: continue

            X_src = self.embed[X[i: i+size, 0]]

            yield {"x": X_src, "y": y[i: i+size, :]}

        X_src = self.embed[X[X.shape[0] // size * size:, 0]]
        yield {"x": X_src, "y": y[X.shape[0] // size * size: , :]}

#%%


