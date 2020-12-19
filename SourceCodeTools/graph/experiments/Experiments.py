# %%
import pandas
from os.path import join

from sklearn.model_selection import train_test_split
import numpy as np

# from graphtools import Embedder
from SourceCodeTools.graph.model.Embedder import Embedder
import pickle
import random as rnd
import torch
import os


def keep_from_set(table, pool):
    table['src'] = table['src'].apply(lambda nid: nid if nid in pool else None)
    table['dst'] = table['dst'].apply(lambda nid: nid if nid in pool else None)
    return table.dropna(axis=0)


def get_nodes_with_split_ids(embedder, split_ids):
    return np.fromiter((embedder.inv[s] for s in split_ids), dtype=np.int32)
    # return nodes[
    #     nodes['global_graph_id'].apply(lambda x: x in split_ids)
    # ]['id'].to_numpy()


class Experiments:
    """
    Provides convenient interface for creating experiments.
    link - experiment that tries to predict function call edges based on heldout set
    apicall - experiment that tries to predict which function is called after the current function
    typeuse - experimen that tries to predict typeuse edges based on heldout set
    varuse - experiment that tries to predict which variable names are used in the current function
    fname - experiment that tries to predict the name of a function. valid only for function nodes. information is
            extracted from training data
    typeann -
    """

    def __init__(self,
                 base_path=None,
                 api_seq_path=None,
                 type_use_path=None,
                 type_link_path=None,
                 type_link_train_path=None,
                 type_link_test_path=None,
                 node_type_path=None,
                 variable_use_path=None,
                 function_name_path=None,
                 type_ann=None,
                 gnn_layer=-1):
        """

        :param base_path: path tp trained gnn model
        :param api_seq_path: path to api call edges
        :param type_use_path: path to type use edges. currently not used, edges extracted from gnn model heldout set
        :param node_type_path: path to node type. currently not used, edges extracted from gnn model heldout set
        :param variable_use_path: path to variable use edges
        :param function_name_path: path to function name edges
        :param gnn_layer: which gnn layer is used for node embeddings
        """

        self.experiments = {
            'fcall': function_name_path,
            'apicall': api_seq_path,
            'typeuse': type_use_path,
            'varuse': variable_use_path,
            'fname': function_name_path,
            'typeann': type_ann,
            'typelink': type_link_path,
            'typelink_train': type_link_train_path,
            'typelink_test': type_link_test_path
        }

        self.base_path = base_path

        # TODO
        # load splits for state dict and use them for training
        # self.splits = torch.load(os.path.join(self.base_path, "state_dict.pt"))["splits"]

        if base_path is not None:
            self.embed = pickle.load(open(join(self.base_path, "embeddings.pkl"), "rb"))[gnn_layer]
            # alternative_nodes = pickle.load(open("nodes.pkl", "rb"))
            # self.embed.e = alternative_nodes
            # self.embed.e = np.random.randn(self.embed.e.shape[0], self.embed.e.shape[1])
            # self.embed.e = np.ones((self.embed.e.shape[0], self.embed.e.shape[1]))

        # e = pickle.load(open(join(self.base_path, "embeddings.pkl"), "rb"))
        #
        # self.embed = Embedder({0:0}, [])
        # self.embed.e = e.e
        # self.embed.ind = e.ind
        # self.embed.inv = e.inv

    def filter_valid(self, keys):
        """
        Verify that there are embeddings for nodes specified by keys
        :param keys: iterable with original node ids (as opposed to compacted graph node ids)
        :return: ids that have gnn embeddings
        """
        return np.fromiter(filter(lambda id: id in self.embed.ind, keys), dtype=np.int32)
        # return np.fromiter((key for key in keys if key in self.embed.ind), dtype=np.int32)

    def __getitem__(self, type: str):
        """
        Return object that allows creating batches for the choosen experiment. Several experiments available
        link - experiment that tries to predict function call edges based on heldout set
        apicall - experiment that tries to predict which function is called after the current function
        typeuse - experimen that tries to predict typeuse edges based on heldout set
        varuse - experiment that tries to predict which variable names are used in the current function
        fname - experiment that tries to predict the name of a function. valid only for function nodes. information is
                extracted from training data
        :param type: str description of the experiment
        :return: Experiment object
        """
        nodes = pandas.read_csv(join(self.base_path, "nodes.csv"))
        edges = pandas.read_csv(join(self.base_path, "held.csv")).astype({"src": "int32", "dst": "int32"})

        global_ids = nodes['global_graph_id'].values
        self.splits = (
            global_ids[nodes['train_mask'].values],
            global_ids[nodes['val_mask'].values],
            global_ids[nodes['test_mask'].values],
        )
        if type == "link":
            # nodes = pandas.read_csv(join(self.base_path, "nodes.csv"))
            held = pandas.read_csv(join(self.base_path, "held.csv")).astype({"src": "int32", "dst": "int32"})

            held = held.query('type == 8')[['src', 'dst']]

            # node_pool = set(self.splits[2])
            # held = keep_from_set(held, node_pool)

            return Experiment(self.embed, nodes, edges, held, split_on="nodes", neg_sampling_strategy="word2vec",
                              compact_dst=False)

        elif type == "apicall":
            api_seq = pandas.read_csv(self.experiments['apicall']).astype({"src": "int32", "dst": "int32"})

            # unique_nodes = set(nodes['id'].values.tolist())

            # api_seq_test = api_seq.copy()
            # api_seq_test['src'] = api_seq_test['src'].apply(lambda nid: nid if nid in unique_nodes else None)
            # api_seq_test['dst'] = api_seq_test['dst'].apply(lambda nid: nid if nid in unique_nodes else None)
            # api_seq_test.dropna(axis=0, inplace=True)

            # disabled for testing
            # api_seq = api_seq[
            #     api_seq['src'].apply(lambda nid: nid in unique_nodes)
            # ]
            #
            # api_seq = api_seq[
            #     api_seq['dst'].apply(lambda nid: nid in unique_nodes)
            # ]

            # node_pool = set(self.splits[2])
            node_pool = set(get_nodes_with_split_ids(self.embed, self.splits[2]))
            api_seq = keep_from_set(api_seq, node_pool)

            return Experiment(self.embed, nodes, edges, api_seq, split_on="nodes", neg_sampling_strategy="word2vec",
                              compact_dst=False)

        elif type == "typeuse":
            held = pandas.read_csv(join(self.base_path, "held.csv")).astype({"src": "int32", "dst": "int32"})

            held = held.query('type == 2')[['src', 'dst']]

            # node_pool = set(self.splits[2])
            # held = keep_from_set(held, node_pool)

            return Experiment(self.embed, nodes, edges, held, split_on="nodes", neg_sampling_strategy="word2vec",
                              compact_dst=False)

        elif type == "typelink":
            typelink = pandas.read_csv(self.experiments['typelink']).astype({"src": "int32", "dst": "int32"})

            node_pool = set(nodes['id'].values.tolist())

            typelink = keep_from_set(typelink, node_pool)

            return Experiment(self.embed, nodes, edges, typelink, split_on="nodes", neg_sampling_strategy="word2vec",
                              compact_dst=False)

        elif type == "typelink_tt":
            typelink_train = pandas.read_csv(self.experiments['typelink_train']).astype(
                {"src": "int32", "dst": "int32"})
            typelink_test = pandas.read_csv(self.experiments['typelink_test']).astype(
                {"src": "int32", "dst": "int32"})

            node_pool = set(nodes['id'].values.tolist())

            return Experiment_tt(self.embed, nodes, edges, target_train=typelink_train, target_test=typelink_test, neg_sampling_strategy="word2vec",
                              compact_dst=False)

        elif type == "typeann":
            type_ann = pandas.read_csv(self.experiments['typeann']).astype({"src": "int32", "dst": "str"})

            # node_pool = set(self.splits[2]).union(self.splits[1]).union(self.splits[0])
            # node_pool = set(nodes['id'].values.tolist())
            # node_pool = set(get_nodes_with_split_ids(nodes, set(self.splits[2]).union(self.splits[1]).union(self.splits[0])))
            node_pool = set(
                get_nodes_with_split_ids(self.embed, set(self.splits[2]).union(self.splits[1]).union(self.splits[0])))

            type_ann = type_ann[
                type_ann['src'].apply(lambda nid: nid in node_pool)
            ]

            # return Experiment2(self.embed, nodes, edges, type_ann, split_on="nodes", neg_sampling_strategy="word2vec")

            return Experiment3(self.embed, nodes, edges, type_ann, split_on="edges",
                               neg_sampling_strategy="word2vec", compact_dst=True)


        elif type == "varuse":
            var_use = pandas.read_csv(self.experiments['varuse']).astype({"src": "int32", "dst": "str"})

            # unique_nodes = set(nodes['id'].values.tolist())
            # node_pool = set(self.splits[2])
            # node_pool = set(get_nodes_with_split_ids(nodes, set(self.splits[2])))
            node_pool = set(get_nodes_with_split_ids(self.embed, self.splits[2]))

            var_use = var_use[
                var_use['src'].apply(lambda nid: nid in node_pool)
            ]

            return Experiment2(self.embed, nodes, edges, var_use, split_on="nodes", neg_sampling_strategy="word2vec")

        elif type == "fname":

            # fname = pandas.read_csv(self.experiments['fname'])
            functions = nodes.query('label == 4096')
            functions['fname'] = functions['name'].apply(lambda name: name.split(".")[-1])

            functions['src'] = functions['id']
            functions['dst'] = functions['fname']

            # unique_nodes = set(nodes['id'].values.tolist())
            # node_pool = set(self.splits[2])
            # node_pool = set(get_nodes_with_split_ids(nodes, set(self.splits[2])))
            node_pool = set(get_nodes_with_split_ids(self.embed, self.splits[2]))

            functions = functions[
                functions['src'].apply(lambda nid: nid in node_pool)
            ]

            # use edge splits when outgoing degree is 1

            return Experiment2(self.embed, nodes, edges, functions[['src', 'dst']], split_on="edges",
                               neg_sampling_strategy="word2vec")

        elif type == "nodetype":

            types = nodes.copy()
            types['src'] = nodes['id']
            types['dst'] = nodes['label']

            print("WARNING: Make sure that you target label is stored in the field: label")
            # raise Warning("Make sure that you target label is stored in the field: label")

            node_pool = set(self.splits[2])

            types['src'] = types['src'].apply(lambda nid: nid if nid in node_pool else None)
            types = types.dropna(axis=0)

            return Experiment3(self.embed, nodes, edges, types[['src', 'dst']], split_on="edges",
                               neg_sampling_strategy="word2vec")
        else:
            raise ValueError(
                f"Unknown experiment: {type}. The following experiments are available: [apicall|link|typeuse|varuse|fname|nodetype].")


class Experiment:
    def __init__(self,
                 embeddings: Embedder,
                 nodes: pandas.DataFrame,
                 edges: pandas.DataFrame,
                 target: pandas.DataFrame,
                 split_on="nodes",
                 neg_sampling_strategy="word2vec",
                 K=1,
                 test_frac=0.1, compact_dst=True):

        # store local copies"
        self.embed = embeddings
        self.nodes = nodes
        self.edges = edges
        self.target = target
        # internal variables
        self.split_on = split_on
        self.neg_smpl_strategy = neg_sampling_strategy
        self.K = K
        self.TEST_FRAC = test_frac

        # make sure to drop duplicate edges to prevent leakage into the test set
        # do it before creating experiment?
        self.target.drop_duplicates(['src', 'dst'], inplace=True, ignore_index=True)

        # size of embeddigns given by gnn model
        self.embed_size = self.embed.e.shape[1]
        # self.n_classes = target['dst'].unique().size

        # initialize negative sampler
        self.unique_src = self.target['src'].unique()
        self.unique_dst = self.target['dst'].unique()
        self.init_negative_sampler(strategy=self.neg_smpl_strategy)

        if self.split_on == "edges":
            # train test split will be stored here
            # these arrays store indices with respect to target data (edge indices)
            self.train_edge_ind, self.test_edge_ind = train_test_split(np.arange(start=0, stop=self.target.shape[0]),
                                                                       test_size=self.TEST_FRAC, random_state=42)
            #
            # # these arrays sore edge lists and labels for these edges
            # self.X_train = None
            # self.X_test = None
            # self.y_train = None
            # self.y_test = None

        elif self.split_on == "nodes":
            # # this is useful if you use ElementEmbedder
            self.target['id'] = self.target['src']
            from SourceCodeTools.graph.model.ElementEmbedderBase import ElementEmbedderBase
            self.ee = ElementEmbedderBase(self.target, compact_dst=compact_dst)
            self.train_nodes, self.test_nodes = train_test_split(self.unique_src, test_size=self.TEST_FRAC,
                                                                 random_state=42)
        else:
            raise ValueError("Unsupported split mode:", self.split_on)

    def init_negative_sampler(self, unigram_power=3 / 4, strategy="word2vec"):
        """
        Initialize word2vec style negative sampler. Unigram for target nodes is computed. Negative samples
        are drawn from modified unigram distribution.
        :return: np array of dst nodes (as given in self.target table)
        """
        if strategy == "word2vec":
            counts = self.target['dst'].value_counts(normalize=True)
            freq = counts.values ** unigram_power
            self.freq = freq / sum(freq)
            self.dst_idxs = counts.index
            self.dst_neg_sampling = lambda size: np.random.choice(self.dst_idxs, size, replace=True, p=self.freq)
        elif strategy == "uniform":
            self.dst_neg_sampling = lambda size: np.random.choice(self.unique_dst, size, replace=True)

    # def filter_valid(self, keys):
    #     # not used, all nodes are considered to be already filtered
    #     return keys
    #     # filtered = np.array([key for key in keys if key in self.embed.ind], dtype=np.int32)
    #     filtered = np.fromiter(filter(lambda key: key in self.embed.ind, keys), dtype=np.int32)
    #     # self.last_filtered = keys.size - filtered
    #     # return filtered

    # def get_random_ind(self, set_size, num):
    #     """
    #     Used for simple negative sampling strategy.
    #     :param set_size: range of values from which negative samples are grawn
    #     :param num: size of negative sample
    #     :return: array of contiguous indices from range 0:set_size
    #     """
    #     return np.random.randint(low=0, high=set_size, size=num)

    def get_negative_edges(self, src_set, dst_set, num):
        """
        Sample negative edges.
        :param src_set: Specifies the set of src nodes for negative edges
        :param dst_set: Specifies the set of src nodes for negative edges
        :param num: size of negative sample
        :return: edge list in the form of 2d numpy array
        """
        src_size = src_set.size
        dst_size = dst_set.size

        src_negative_ind = np.random.randint(low=0, high=src_size, size=num)
        # src_negative_ind = self.get_random_ind(src_size, num)
        # dst_negative_ind = self.get_random_ind(dst_size, num)

        # src_negative = self.filter_valid(src_set[src_negative_ind])
        # dst_negative = self.filter_valid(dst_set[dst_negative_ind])
        src_negative = src_set[src_negative_ind]
        dst_negative = self.dst_neg_sampling(num)

        # while min(src_negative.size, dst_negative.size, num) != num:
        #     # print(min(src_negative.size, dst_negative.size, num))
        #     src_negative_ind = self.get_random_ind(src_size, num)
        #     dst_negative_ind = self.get_random_ind(dst_size, num)
        #
        #     src_negative = np.concatenate([src_negative, self.filter_valid(src_set[src_negative_ind])])
        #     dst_negative = np.concatenate([dst_negative, self.filter_valid(dst_set[dst_negative_ind])])

        # src_negative = src_negative[:num]
        # dst_negative = dst_negative[:num]

        negative_edges = np.hstack([src_negative.reshape(-1, 1), dst_negative.reshape(-1, 1)])
        return negative_edges

    def get_training_data(self):
        """
        Split edges provided by self.target into train and test sets
        :return:
        """

        # this actually never was a set
        # src_set = self.target['src'].values
        # dst_set = self.target['dst'].values

        # train_negative = self.get_negative_edges(src_set, dst_set, self.train_ind.shape[0]) # * self.K)
        # test_negative = self.get_negative_edges(src_set, dst_set, self.test_ind.shape[0])

        train_positive = self.target.iloc[self.train_edge_ind].values
        test_positive = self.target.iloc[self.test_edge_ind].values

        # # print(train_positive.shape, train_negative.shape, test_positive.shape, test_negative.shape)
        # print(f"Working with {train_positive.shape[0]} positive and {train_negative.shape[0]} negative samples in the train set, {test_positive.shape[0]} and {test_negative.shape[0]} - in test set")

        X_train = train_positive
        X_test = test_positive

        y_train = np.ones((self.train_edge_ind.shape[0],))
        y_test = np.ones((self.test_edge_ind.shape[0],))

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
        src = edges[:, 0]
        dst = edges[:, 1]

        return np.hstack([self.embed[src], self.embed[dst]])
        # return np.hstack([np.random.rand(src.shape[0], self.embed.e.shape[1]), np.random.rand(src.shape[0], self.embed.e.shape[1])])

    def batch_edges(self, X, size=128, K=15):
        """
        Generate batch
        :param X: input edge list in 2d numpy array
        :param y: labels
        :param size: number of positive samples in the batch
        :param K: negative oversampling factor
        :return: dictionary ready to be fed to classifier model
        """

        # def encode_binary(y):
        #     y_encoded = np.zeros((y.shape[0], 2))
        #     y_encoded[:, y.astype(np.int32)] = 1
        #     return y_encoded

        # # assume negative samples are pre-generated
        # for i in range(0, X.shape[0], size):
        #     if i + size >= X.shape[0]: continue
        #
        #     X_b = self._embed(X[i: i+size])
        #     y_b = y[i: i+size]
        #
        #     assert y_b.shape[0] == X_b.shape[0]
        #     yield {"x": X_b, "y": y_b}
        #
        # X_b = self._embed(X[X.shape[0] // size * size:])
        # y_b = y[X.shape[0] // size * size:]
        # yield {"x": X_b, "y": y_b}

        for i in range(0, X.shape[0], size):
            if i + size >= X.shape[0]: continue

            neg = self.get_negative_edges(X[i: i + size, 0], self.unique_dst, size * K)
            # neg = self.get_negative_edges(self.unique_src, self.unique_dst, size * K)
            X_b = np.vstack([X[i: i + size], neg])

            X_b_e = self._embed(X_b)
            y_b = np.concatenate([np.ones(size, ), np.zeros(size * K, )])

            assert np.average(y_b) == 1 / (1 + K)
            assert y_b.shape[0] == X_b.shape[0] == size * (1 + K)

            yield {"x": X_b_e, "y": y_b}

        last_piece = X[X.shape[0] // size * size:, :]
        neg = self.get_negative_edges(last_piece[:, 0], self.unique_dst, last_piece.shape[0] * K)
        # neg = self.get_negative_edges(self.unique_src, self.unique_dst, last_piece.shape[0] * K)
        X_b = np.vstack([last_piece, neg])
        y_b = np.concatenate([np.ones(last_piece.shape[0], ), np.zeros(last_piece.shape[0] * K, )])

        assert np.average(y_b) == 1 / (1 + K)
        assert y_b.shape[0] == X_b.shape[0] == last_piece.shape[0] * (1 + K)

        yield {"x": self._embed(X_b), "y": y_b}

    def test_batches(self):
        # if self.X_test is None:
        #     self.get_training_data()
        if self.split_on == "nodes":
            return self.batch_nodes(self.test_nodes, K=1, test=True)
        elif self.split_on == "edges":
            return self.batch_edges(self.target.iloc[self.test_edge_ind][['src', 'dst']].values, K=1)

    def train_batches(self):
        # if self.X_train is None:
        #     self.get_training_data()
        if self.split_on == "nodes":
            return self.batch_nodes(self.train_nodes, K=1)
        elif self.split_on == "edges":
            return self.batch_edges(self.target.iloc[self.train_edge_ind][['src', 'dst']].values, K=1)

    def batch_nodes(self, indices, size=128, K=15, test=False):

        if not test and indices.shape[0] < size:
            raise ValueError("The amount of training data is too small")

        if test:
            n_batches = 1
            sample_size = indices.shape[0]
        else:
            n_batches = indices.shape[0] // size + 1
            sample_size = size

        for _ in range(n_batches):
            batch_ind = np.random.choice(indices, sample_size, replace=False)
            positive_in = self.embed[batch_ind]
            positive_out = self.embed[self.ee[batch_ind]]

            negative_in = np.repeat(positive_in, K, axis=0)
            negative_out = self.embed[self.ee.sample_negative(sample_size * K)]
            # negative_out = self.embed[np.random.choice(self.unique_dst, size=size * K)]
            X_b = np.concatenate([
                np.concatenate([
                    positive_in, positive_out
                ], axis=1),
                np.concatenate([
                    negative_in, negative_out
                ], axis=1)
            ], axis=0)

            y_b = np.concatenate([np.ones(sample_size, ), np.zeros(sample_size * K, )])

            yield {"x": X_b, "y": y_b}


class Experiment_tt(Experiment):
    def __init__(self,
                 embeddings: Embedder,
                 nodes: pandas.DataFrame,
                 edges: pandas.DataFrame,
                 target_train: pandas.DataFrame,
                 target_test: pandas.DataFrame,
                 neg_sampling_strategy="word2vec",
                 K=1, compact_dst=False):

        # store local copies"
        self.embed = embeddings
        self.nodes = nodes
        self.edges = edges
        self.target = pandas.concat([target_train, target_test], axis=0)
        # internal variables
        self.split_on = "nodes"
        self.neg_smpl_strategy = neg_sampling_strategy
        self.K = K

        # make sure to drop duplicate edges to prevent leakage into the test set
        # do it before creating experiment?
        self.target.drop_duplicates(['src', 'dst'], inplace=True, ignore_index=True)

        train_nodes = set(target_train['src'].values)
        test_nodes = set(target_test['src'].values)

        train_nodes -= test_nodes
        if len(target_train) != len(train_nodes):
            print(f"Removed {len(target_train) - len(train_nodes)} nodes from train in favor of the test set")

        train_nodes = np.array(train_nodes)
        test_nodes = np.array(test_nodes)

        # size of embeddings given by gnn model
        self.embed_size = self.embed.e.shape[1]

        # initialize negative sampler
        self.unique_src = self.target['src'].unique()
        self.unique_dst = self.target['dst'].unique()
        self.init_negative_sampler(strategy=self.neg_smpl_strategy)

        if self.split_on == "edges":
            raise NotImplementedError("This strategy is not valid for this experiment")

        elif self.split_on == "nodes":
            # # this is useful if you use ElementEmbedder
            self.target['id'] = self.target['src']
            from SourceCodeTools.graph.model.ElementEmbedderBase import ElementEmbedderBase
            self.ee = ElementEmbedderBase(self.target, compact_dst=compact_dst)
            self.train_nodes, self.test_nodes = train_nodes, test_nodes
        else:
            raise ValueError("Unsupported split mode:", self.split_on)


def compact_property(values):
    uniq = np.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    return prop2pid


class Experiment2(Experiment):
    def __init__(self, embeddings: Embedder,
                 nodes: pandas.DataFrame,
                 edges: pandas.DataFrame,
                 target: pandas.DataFrame,
                 split_on="nodes",
                 neg_sampling_strategy="word2vec",
                 K=1,
                 test_frac=0.1, compact_dst=True):
        super(Experiment2, self).__init__(embeddings, nodes, edges, target, split_on=split_on,
                                          neg_sampling_strategy=neg_sampling_strategy, K=K, test_frac=test_frac,
                                          compact_dst=compact_dst)

        # def compact_property(values):
        #     uniq = np.unique(values)
        #     prop2pid = dict(zip(uniq, range(uniq.size)))
        #     return prop2pid

        self.name_map = compact_property(target['dst'])
        self.dst_orig = target['dst']
        target['dst'] = target['dst'].apply(lambda name: self.name_map[name])

        print(f"Doing experiment with {len(self.name_map)} distinct target targets")

        self.unique_src = self.target['src'].unique()
        self.unique_dst = self.target['dst'].unique()
        self.unique_elements = self.unique_dst.size

        self.init_negative_sampler(strategy=self.neg_smpl_strategy)

    def _embed(self, edges):
        pass
        # src = edges[:,0]
        # dst = edges[:,1]
        #
        # return np.hstack([self.embed[src], self.embed[dst]])

    def batch_nodes(self, indices, size=256, K=15, test=False):

        if not test and indices.shape[0] < size:
            raise ValueError("The amount of training data is too small")

        if test:
            n_batches = 1
            sample_size = indices.shape[0]
        else:
            n_batches = indices.shape[0] // size + 1
            sample_size = size

        for _ in range(n_batches):
            batch_ind = np.random.choice(indices, sample_size)
            positive_in = self.embed[batch_ind]
            positive_out = self.ee[batch_ind]

            negative_in = np.repeat(positive_in, K, axis=0)
            negative_out = self.ee.sample_negative(sample_size * K)
            # negative_out = self.embed[np.random.choice(self.unique_dst, size=size * K)]
            in_ = np.concatenate([
                positive_in, negative_in
            ], axis=0)
            dst_ = np.concatenate([
                positive_out, negative_out
            ], axis=0)

            y_b = np.concatenate([np.ones(sample_size, ), np.zeros(sample_size * K, )])

            yield {"x": in_, 'elements': dst_, "y": y_b}

    def batch_edges(self, X, size=256, K=15):
        for i in range(0, X.shape[0], size):
            if i + size >= X.shape[0]: continue

            neg = self.get_negative_edges(X[i: i + size, 0], self.unique_dst, size * K)
            X_ = np.vstack([X[i: i + size], neg])

            src = X_[:, 0]
            dst = X_[:, 1]

            X_src = self.embed[src]

            y_b = np.concatenate([np.ones(size, ), np.zeros(size * K, )])

            assert y_b.shape[0] == X_src.shape[0] == dst.shape[0] == size * (1 + K)
            yield {"x": X_src, "elements": dst, "y": y_b}

        last_piece = X[X.shape[0] // size * size:]
        neg = self.get_negative_edges(last_piece[:, 0], self.unique_dst, last_piece.shape[0] * K)
        X_ = np.vstack([last_piece, neg])

        src = X_[:, 0]
        dst = X_[:, 1]

        X_src = self.embed[src]

        y_b = np.concatenate([np.ones(last_piece.shape[0], ), np.zeros(last_piece.shape[0] * K, )])

        assert y_b.shape[0] == X_src.shape[0] == dst.shape[0] == last_piece.shape[0] * (1 + K)

        yield {"x": X_src, "elements": dst, "y": y_b}


class Experiment3(Experiment2):
    def __init__(self, embeddings: Embedder,
                 nodes: pandas.DataFrame,
                 edges: pandas.DataFrame,
                 target: pandas.DataFrame,
                 split_on="nodes",
                 neg_sampling_strategy="word2vec",
                 K=1,
                 test_frac=0.1, compact_dst=True
                 ):
        super(Experiment3, self).__init__(embeddings, nodes, edges, target, split_on=split_on,
                                          neg_sampling_strategy=neg_sampling_strategy, K=K, test_frac=test_frac,
                                          compact_dst=compact_dst)

    def get_training_data(self):

        # self.get_train_test_split()

        train_positive = self.target.iloc[self.train_edge_ind].values
        test_positive = self.target.iloc[self.test_edge_ind].values

        X_train, y_train = train_positive[:, 0].reshape(-1, 1), train_positive[:, 1].reshape(-1, 1)
        X_test, y_test = test_positive[:, 0].reshape(-1, 1), test_positive[:, 1].reshape(-1, 1)

        def shuffle(X, y):
            ind_shuffle = np.arange(0, X.shape[0])
            np.random.shuffle(ind_shuffle)
            return X[ind_shuffle], y[ind_shuffle]

        self.X_train, self.y_train = shuffle(X_train, y_train)
        self.X_test, self.y_test = shuffle(X_test, y_test)

    def batch(self, X, y, size=256, **kwargs):
        for i in range(0, X.shape[0], size):
            if i + size >= X.shape[0]: continue

            X_src = self.embed[X[i: i + size, 0]]

            yield {"x": X_src, "y": y[i: i + size, :]}

        X_src = self.embed[X[X.shape[0] // size * size:, 0]]
        yield {"x": X_src, "y": y[X.shape[0] // size * size:, :]}

    def train_batches(self):
        if not hasattr(self, "X_train"):
            self.get_training_data()

        return self.batch(self.X_train, self.y_train)

    def test_batches(self):
        if not hasattr(self, "X_test"):
            self.get_training_data()

        return self.batch(self.X_test, self.y_test)

# %%
