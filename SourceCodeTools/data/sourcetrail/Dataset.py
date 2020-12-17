import pandas
# import dgl
import numpy
import pickle
# import torch

from os.path import join

def load_data(node_path, edge_path):
    nodes = pandas.read_csv(node_path, dtype={'id': int, 'type': int, 'serialized_name': str}, escapechar='\\')
    edges = pandas.read_csv(edge_path, dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})

    nodes_ = nodes.rename(mapper={
        'serialized_name': 'name'
    }, axis=1).astype({"name":str, "id": int, "type": int})

    edges_ = edges.rename(mapper={
        'source_node_id': 'src',
        'target_node_id': 'dst'
    }, axis=1).astype({'id': int, 'type': int, 'src': int, 'dst': int})

    nodes_['libname'] = nodes_['name'].apply(lambda name: name.split(".")[0])

    return nodes_, edges_


def create_mask(size, idx):
    mask = numpy.full((size,), False, dtype=numpy.bool)
    mask[idx] = True
    return mask


# def compact_prop(df, prop):
#     uniq = df[prop].unique()
#     prop2pid = dict(zip(uniq, range(uniq.size)))
#     compactor = lambda type: prop2pid[type]
#     df['compact_' + prop] = df[prop].apply(compactor)
#     return df

def compact_property(values):
    uniq = numpy.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    # prop2pid = dict(list(zip(uniq, list(range(uniq.size)))))
    return prop2pid


def get_train_test_val_indices(labels, train_frac=0.6):
    # numpy.random.seed(42)

    indices = numpy.arange(start=0, stop=labels.size)
    numpy.random.shuffle(indices)

    train = int(indices.size * train_frac)
    test = int(indices.size * (train_frac + (1 - train_frac)/2))

    print("Splitting into train {}, test {}, and validation {} sets".format(train, test - train, indices.size - test))

    return indices[:train], indices[train: test], indices[test:]


class SourceGraphDataset:
    def __init__(self, nodes_path, edges_path,
                 label_from, node_types=False,
                 edge_types=False, filter=None, holdout_frac=0.001, restore_state=False, self_loops=False,
                 holdout=None, train_frac=0.6):
        """
        Prepares the data for training GNN model. The graph is prepared in the following way:
            1. Edges are split into the train set and holdout set. Holdout set is used in the future experiments.
                Without holdout set the results of the future experiments may be biased. After removing holdout edges
                from the main graph, the disconnected nodes are filtered, so that he graph remain connected.
            2. Since training objective will be defined on the node embeddings, the nodes are split into train, test,
                and validation sets. The test set should be used in the future experiments for training. Validation and
                test sets are equal in size and constitute 40% of all nodes.
            3. The default label is assumed to be node type. Node types can be incorporated into the model by setting
                node_types flag to True.
            4. Graphs require contiguous indexing of nodes. For this reason additional mapping is created that tracks
                the relationship between the new graph id and the original node id from the training data.
        :param nodes_path: path to csv or compressed csv for nodes witch columns
                "id", "type", {"name", "serialized_name"}, {any column with labels}
        :param edges_path: path to csv or compressed csv for edges with columns
                "id", "type", {"source_node_id", "src"}, {"target_node_id", "dst"}
        :param label_from: the column where the labels are taken from
        :param node_types: boolean value, whether to use node types or not
                (node-heterogeneous graph}
        :param edge_types: boolean value, whether to use edge types or not
                (edge-heterogeneous graph}
        :param filter: list[str], the types of edges to filter from graph
        :param holdout_frac: float in [0, 1]
        """
        # TODO
        # 1. test with node types and RGCN
        # 2. GAT RGCN model
        # 3. GGNN model
        # 4. create model where variable names are in the graph. This way we can feed
        #       information about ast inside he function embedding

        # self.nodes = pandas.read_csv(nodes_path)
        # self.edges = pandas.read_csv(edges_path)

        self.holdout_frac = holdout_frac

        self.nodes, self.edges = load_data(nodes_path, edges_path)

        if self_loops:
            self.nodes, self.edges = SourceGraphDataset.assess_need_for_self_loops(self.nodes, self.edges)

        # if restore_state:
        #     self.nodes, self.edges, self.held = pickle.load(open("../../../graph-network/tmp_edgesplits.pkl", "rb"))
        #     print("Restored graph from saved state")
        # else:
        # the next line will delete isolated nodes
        if holdout is None:
            self.nodes, self.edges, self.held = SourceGraphDataset.holdout(self.nodes, self.edges, self.holdout_frac)
        else:
            self.held = pandas.read_csv(holdout)
        # pickle.dump((self.nodes, self.edges, self.held), open("../../../graph-network/tmp_edgesplits.pkl", "wb"))

        # # ablation
        # print("Edges before filtering", self.edges.shape[0])
        # # 16 inheritance *
        # # 512 import *
        # # 4 use 
        # # 2 typeuse *
        # # 8 call *
        # # 1 contain *
        # self.edges = self.edges.query("type != 512")
        # print("Edges after filtering", self.edges.shape[0])
        if filter is not None:
            for e_type in filter:
                self.edges = self.edges.query(f"type != {e_type}")

        self.nodes_have_types = node_types
        self.edges_have_types = edge_types
        self.labels_from = label_from

        self.g = None

        # compact labels
        self.nodes['label'] = self.nodes[label_from]
        self.label_map = compact_property(self.nodes['label'])
        # self.nodes['label_comp']

        assert any(pandas.isna(self.nodes['label'])) == False

        self.nodes['type_backup'] = self.nodes['type']
        if not self.nodes_have_types:
            self.nodes['type'] = 0

        self.nodes, self.label_map = self.add_compact_labels()
        self.nodes, self.id_map = self.add_graph_ids()
        self.nodes, self.typed_id_map = self.add_typed_ids()
        self.edges = self.add_compact_edges()

        # if restore_state:
        #     self.splits = pickle.load(open("../../../graph-network/tmp_splits.pkl", "rb"))
        #     print("Restored node splits from saved state")
        # else:
        self.splits = get_train_test_val_indices(self.nodes.index, train_frac=train_frac)
        # pickle.dump(self.splits, open("../../../graph-network/tmp_splits.pkl", "wb"))

        self.add_splits()

        self.create_graph()

        self.update_global_id()

        self.nodes.sort_values('global_graph_id', inplace=True)

        v = self.nodes['global_graph_id'].values

        self.splits = (
            v[self.nodes['train_mask'].values],
            v[self.nodes['test_mask'].values],
            v[self.nodes['val_mask'].values]
        )

    def add_splits(self):
        self.nodes['train_mask'] = False
        self.nodes.loc[self.nodes.index[self.splits[0]], 'train_mask'] = True
        self.nodes['test_mask'] = False
        self.nodes.loc[self.nodes.index[self.splits[1]], 'test_mask'] = True
        self.nodes['val_mask'] = False
        self.nodes.loc[self.nodes.index[self.splits[2]], 'val_mask'] = True

    def add_graph_ids(self):

        nodes = self.nodes.copy()

        id_map = compact_property(nodes['id'])

        nodes['graph_id'] = nodes['id'].apply(lambda old_id: id_map[old_id])

        return nodes, id_map

    def add_typed_ids(self):
        nodes = self.nodes.copy()

        typed_id_map = {}

        for type in nodes['type'].unique():
            type_ind = nodes[nodes['type'] == type].index

            id_map = compact_property(nodes.loc[type_ind, 'id'])

            nodes.loc[type_ind, 'typed_id'] = nodes.loc[type_ind, 'id'].apply(lambda old_id: id_map[old_id])

            typed_id_map[str(type)] = id_map

        assert any(pandas.isna(nodes['typed_id'])) == False

        return nodes, typed_id_map

    def add_compact_labels(self):

        nodes = self.nodes.copy()

        label_map = compact_property(nodes['label'])

        nodes['compact_label'] = nodes['label'].apply(lambda old_id: label_map[old_id])

        return nodes, label_map

    def add_compact_edges(self):

        edges = self.edges.copy()

        node_type_map = dict(zip(self.nodes['id'].values, self.nodes['type']))

        edges['src_type'] = edges['src'].apply(lambda src_id: node_type_map[src_id])
        edges['dst_type'] = edges['dst'].apply(lambda dst_id: node_type_map[dst_id])

        typed_map = dict(zip(self.nodes['id'].values, self.nodes['typed_id']))

        edges['src_type_graph_id'] = edges['src'].apply(lambda src_id: self.id_map[src_id])
        edges['dst_type_graph_id'] = edges['dst'].apply(lambda dst_id: self.id_map[dst_id])
        edges['src_type_typed_id'] = edges['src'].apply(lambda src_id: typed_map[src_id])
        edges['dst_type_typed_id'] = edges['dst'].apply(lambda dst_id: typed_map[dst_id])

        return edges

    def update_global_id(self):
        if self.edges_have_types:
            orig_id = [];
            graph_id = [];
            prev_offset = 0

            # typed_node_id_maps = self.typed_node_id_maps
            typed_node_id_maps = self.typed_id_map

            for type in self.g.ntypes:
                from_id, to_id = zip(*typed_node_id_maps[type].items())
                orig_id.extend(from_id)
                graph_id.extend([t + prev_offset for t in to_id])
                prev_offset += self.g.number_of_nodes(type)

            global_map = dict(zip(orig_id, graph_id))
        else:
            global_map = self.id_map

        self.nodes['global_graph_id'] = self.nodes['id'].apply(lambda old_id: global_map[old_id])

    @property
    def global_id_map(self):
        self.update_global_id()
        self.nodes.sort_values('global_graph_id', inplace=True)
        return dict(zip(self.nodes['id'].values, self.nodes['global_graph_id'].values))

    @property
    def labels(self):
        self.update_global_id()
        self.nodes.sort_values('global_graph_id', inplace=True)
        return self.nodes['compact_label'].values
        # compact_label = lambda lbl: self.label_map[lbl]
        #
        # if self.edges_have_types:
        #     # typed_labels = self.typed_labels
        #     typed_labels = self.het_labels
        #     return numpy.concatenate(
        #         [
        #             numpy.array([compact_label(lbl) for lbl in typed_labels[ntype]])
        #             for ntype in self.g.ntypes
        #         ]
        #     )
        # else:
        #     return self.nodes['compact_label'].values
        #     # return self.nodes['label'].apply(compact_label).values

    # @property
    # def typed_labels(self):
    #     typed_labels = dict()
    #
    #     unique_types = self.nodes['type'].unique()
    #
    #     for type_id, type in enumerate(unique_types):
    #         nodes_of_type = self.nodes[self.nodes['type'] == type]
    #
    #         typed_labels[str(type)] = self.nodes.loc[
    #             nodes_of_type.index, 'label'
    #         ].values
    #
    #     return typed_labels

    # @property
    # def node_id_maps(self):
    #     if self.node_id_map_ is not None:
    #         if self.edges_have_types:
    #             orig_id = []
    #             graph_id = []
    #             prev_offset = 0
    #
    #             for type in self.g.ntypes:
    #                 from_id, to_id = zip(*self.node_id_map_[type].items())
    #                 orig_id.extend(from_id)
    #                 graph_id.extend([t + prev_offset for t in to_id])
    #                 prev_offset += self.g.number_of_nodes(type)
    #
    #             return dict(zip(orig_id, graph_id))
    #         else:
    #             return self.node_id_map_
    #     else:
    #         return None

    @property
    def num_classes(self):
        return numpy.unique(self.labels).size

    def create_graph(self):
        if not self.nodes_have_types and not self.edges_have_types:
            self.create_directed_graph()
        elif self.edges_have_types:
            self.create_hetero_graph(
                node_types=self.nodes_have_types
            )
        else:
            raise NotImplemented("Edges should have type")

    # @property
    # def typed_node_id_maps(self):
    #     typed_id_maps = dict()
    #
    #     unique_types = self.nodes['type'].unique()
    #
    #     for type_id, type in enumerate(unique_types):
    #         nodes_of_type = self.nodes[self.nodes['type'] == type]
    #         typed_node_id2graph_id = compact_property(
    #             self.nodes.loc[nodes_of_type.index, 'id'].values
    #         )
    #
    #         typed_id_maps[str(type)] = typed_node_id2graph_id
    #
    #     return typed_id_maps
    #
    # @property
    # def node_id_map(self):
    #     if self.edges_have_types:
    #         orig_id = []; graph_id = []; prev_offset = 0
    #
    #         # typed_node_id_maps = self.typed_node_id_maps
    #         typed_node_id_maps = self.het_id_maps
    #
    #         for type in self.g.ntypes:
    #             from_id, to_id = zip(*typed_node_id_maps[type].items())
    #             orig_id.extend(from_id)
    #             graph_id.extend([t + prev_offset for t in to_id])
    #             prev_offset += self.g.number_of_nodes(type)
    #
    #         return dict(zip(orig_id, graph_id))
    #
    #     else:
    #         # return self.dg_id_maps
    #         return compact_property(self.nodes['id'])

    @property
    def typed_node_counts(self):

        typed_node_counts = dict()

        unique_types = self.nodes['type'].unique()

        for type_id, type in enumerate(unique_types):
            nodes_of_type = self.nodes[self.nodes['type'] == type]
            typed_node_counts[str(type)] = nodes_of_type.shape[0]

        return typed_node_counts

    def create_directed_graph(self):
        # nodes = self.nodes.copy()
        # edges = self.edges.copy()

        # from graphtools import create_graph
        # g, labels, id_maps = create_graph(nodes, edges)
        # self.dg_labels = labels
        # self.dg_id_maps = id_maps
        # self.g = g
        # return

        # node_id2graph_id = self.node_id_map

        # assert nodes.shape[0] == len(node_id2graph_id)

        # id2new = lambda id: node_id2graph_id[id]

        # graph_src = edges['src'].apply(id2new).values.tolist()
        # graph_dst = edges['dst'].apply(id2new).values.tolist()

        type2id = compact_property(self.edges['type'])
        edge_types = self.edges['type'].apply(lambda x: type2id[x]).values

        import dgl, torch
        g = dgl.DGLGraph()
        g.add_nodes(self.nodes.shape[0])
        g.add_edges(self.edges['src_type_graph_id'].values.tolist(), self.edges['dst_type_graph_id'].values.tolist())

        g.ndata['labels'] = torch.tensor(self.nodes['compact_label'].values, dtype=torch.int64)
        g.edata['etypes'] = torch.tensor(edge_types, dtype=torch.int64)

        masks = self.nodes[['typed_id', 'train_mask', 'test_mask', 'val_mask']].sort_values('typed_id')
        self.g.ndata['train_mask'] = torch.tensor(masks['train_mask'].values, dtype=bool)
        self.g.ndata['test_mask'] = torch.tensor(masks['test_mask'].values, dtype=bool)
        self.g.ndata['val_mask'] = torch.tensor(masks['val_mask'].values, dtype=bool)

        self.g = g

    def create_hetero_graph(self, node_types=False, edge_types=False):
        # TODO
        # arguments are still not used

        nodes = self.nodes.copy()
        edges = self.edges.copy()

        # from graphtools import create_hetero_graph
        # g, labels, id_maps = create_hetero_graph(nodes, edges)
        # self.g = g
        # self.het_labels = labels
        # self.het_id_maps = id_maps
        # return

        # TODO
        # this is a hack when where are only outgoing connections from this node type
        # nodes, edges = Dataset.assess_need_for_self_loops(nodes, edges)

        typed_node_id = dict(zip(nodes['id'], nodes['typed_id']))

        # typed_node_id_maps = self.typed_node_id_maps
        # typed_node_id_maps = self.typed_id_map

        # node2type = dict(zip(nodes['id'].values, nodes['type'].values))

        # def graphid_lookup(nid):
        #     for type, maps in typed_node_id_maps.items():
        #         if nid in maps:
        #             return maps[nid]
        #
        # type_lookup = lambda id: node2type[id]

        # edges['src_type'] = edges['src'].apply(type_lookup)
        # edges['dst_type'] = edges['dst'].apply(type_lookup)

        possible_edge_signatures = edges[['src_type', 'type', 'dst_type']].drop_duplicates(
            ['src_type', 'type', 'dst_type']
        )

        # typed_subgraphs is a dictionary with subset_signature as a key,
        # the dictionary stores directed edge lists
        typed_subgraphs = {}

        for ind, row in possible_edge_signatures.iterrows():
            subgraph_signature = (str(row.src_type), str(row.type), str(row.dst_type))

            subset = edges.query('src_type == %s and type==%s and dst_type==%s' % subgraph_signature)

            typed_subgraphs[(subgraph_signature)] = list(
                zip(
                    map(lambda old_id: typed_node_id[old_id], subset['src'].values),
                    map(lambda old_id: typed_node_id[old_id], subset['dst'].values)
                )
            )

        import dgl, torch
        self.g = dgl.heterograph(typed_subgraphs, self.typed_node_counts)

        # self_loop_signatures = edges[['type', 'dst_type']].drop_duplicates(['type', 'dst_type'])
        # for ind, row in self_loop_signatures.iterrows():
        #     subgraph_signature = (str(row.dst_type), str(row.type), str(row.dst_type))
        #     self.g = dgl.add_self_loop(self.g, etype=subgraph_signature)

        for ntype in self.g.ntypes:
            masks = self.nodes.query(f"type == {ntype}")[['typed_id', 'train_mask', 'test_mask', 'val_mask', 'compact_label']].sort_values('typed_id')
            self.g.nodes[ntype].data['train_mask'] = torch.tensor(masks['train_mask'].values, dtype=bool)
            self.g.nodes[ntype].data['test_mask'] = torch.tensor(masks['test_mask'].values, dtype=bool)
            self.g.nodes[ntype].data['val_mask'] = torch.tensor(masks['val_mask'].values, dtype=bool)
            self.g.nodes[ntype].data['labels'] = torch.tensor(masks['compact_label'].values, dtype=torch.int64)

        # self.typed_labels_ = typed_labels
        # self.typed_id_maps_ = typed_id_maps

        # self.labels_ = numpy.concatenate([typed_labels[ntype] for ntype in self.g.ntypes])
        #
        # orig_id = []
        # graph_id = []
        # prev_offset = 0
        #
        # for type in self.g.ntypes:
        #     from_id, to_id = zip(*typed_id_maps[type].items())
        #     orig_id.extend(from_id)
        #     graph_id.extend([t + prev_offset for t in to_id])
        #     prev_offset += self.g.number_of_nodes(type)
        #
        # self.node_id_map = dict(zip(orig_id, graph_id))

    # def add_typed_ids(self, nodes):
    #     nodes = nodes.copy()
    #
    #     nodes = compact_prop(nodes, 'type')
    #     nodes['type'] = 0
    #
    #     nodes['typed_id'] = None
    #     uniq_types = nodes['type'].unique()
    #     type_counts = dict()
    #
    #     typed_labels = {}
    #
    #     typed_id_maps = {}
    #
    #     for type_id, type in enumerate(uniq_types):
    #         nodes_of_type = nodes[nodes['type'] == type].shape[0]
    #         nodes.loc[nodes[nodes['type'] == type].index, 'typed_id'] = list(range(nodes_of_type))
    #         type_counts[str(type)] = nodes_of_type
    #
    #         typed_id_maps[str(type)] = dict(
    #             zip(
    #                 nodes.loc[nodes[nodes['type'] == type].index, 'id'].values,
    #                 nodes.loc[nodes[nodes['type'] == type].index, 'typed_id'].values
    #             )
    #         )
    #
    #         # TODO
    #         # node types as labels
    #         typed_labels[str(type)] = nodes.loc[
    #             nodes[nodes['type'] == type].index, 'label'
    #         ].apply(
    #             lambda type: self.label_map[type]
    #         ).values
    #
    #     assert any(nodes['typed_id'].isna()) == False
    #
    #     return nodes, type_counts, typed_labels, typed_id_maps

    @classmethod
    def assess_need_for_self_loops(cls, nodes, edges):
        need_self_loop = set(edges['src'].values.tolist()) - set(edges['dst'].values.tolist())
        for nid in need_self_loop:
            edges = edges.append({
                "id": -1,
                "type": 99,
                "src": nid,
                "dst": nid
            }, ignore_index=True)

        return nodes, edges

    @classmethod
    def holdout(cls, nodes, edges, HOLDOUT_FRAC):
        train, test = split(edges, HOLDOUT_FRAC)

        nodes, train_edges = ensure_connectedness(nodes, train)

        nodes, test_edges = ensure_valid_edges(nodes, test)

        return nodes, train_edges, test_edges


def split(edges, HOLDOUT_FRAC):
    edges_shuffled = edges.sample(frac=1., random_state=42)

    train_frac = int(edges_shuffled.shape[0] * (1. - HOLDOUT_FRAC))

    train = edges_shuffled \
                .iloc[:train_frac]
    test = edges_shuffled \
               .iloc[train_frac:]
    print("Splitting edges into train and test set. Train: {}. Test: {}. Fraction: {}". \
          format(train.shape[0], test.shape[0], HOLDOUT_FRAC))
    return train, test


def ensure_connectedness(nodes: pandas.DataFrame, edges: pandas.DataFrame):
    """
    Filtering isolated nodes
    :param nodes: DataFrame
    :param edges: DataFrame
    :return:
    """

    print("Filtering isolated nodes. Starting from {} nodes and {} edges...".format(nodes.shape[0], edges.shape[0]),
          end="")
    unique_nodes = set(edges['src'].values.tolist() +
                       edges['dst'].values.tolist())

    nodes = nodes[
        nodes['id'].apply(lambda nid: nid in unique_nodes)
    ]

    print("ending up with {} nodes and {} edges".format(nodes.shape[0], edges.shape[0]))

    return nodes, edges


def ensure_valid_edges(nodes, edges):
    """
    Filter edges that link to nodes that do not exist
    :param nodes:
    :param edges:
    :return:
    """
    print("Filtering edges to invalid nodes. Starting from {} nodes and {} edges...".format(nodes.shape[0],
                                                                                            edges.shape[0]),
          end="")

    unique_nodes = set(nodes['id'].values.tolist())

    edges = edges[
        edges['src'].apply(lambda nid: nid in unique_nodes)
    ]

    edges = edges[
        edges['dst'].apply(lambda nid: nid in unique_nodes)
    ]

    print("ending up with {} nodes and {} edges".format(nodes.shape[0], edges.shape[0]))

    return nodes, edges


def read_or_create_dataset(args, model_base, model_name, LABELS_FROM="type"):
    if args.restore_state:
        # i'm not happy with this behaviour that differs based on the flag status
        dataset = pickle.load(open(join(model_base, "dataset.pkl"), "rb"))
    else:

        if model_name == "GCNSampling" or model_name == "GATSampler" or model_name == "GAT" or model_name == "GGNN":
            dataset = SourceGraphDataset(args.node_path, args.edge_path, label_from=LABELS_FROM,
                                         restore_state=args.restore_state, filter=args.filter_edges,
                                         self_loops=args.self_loops,
                                         holdout=args.holdout, train_frac=args.train_frac)
        elif model_name == "RGCNSampling" or model_name == "RGCN":
            dataset = SourceGraphDataset(args.node_path,
                                         args.edge_path,
                                         label_from=LABELS_FROM,
                                         node_types=args.use_node_types,
                                         edge_types=True,
                                         restore_state=args.restore_state,
                                         filter=args.filter_edges,
                                         self_loops=args.self_loops,
                                         holdout=args.holdout,
                                         train_frac=args.train_frac
                                         )
        else:
            raise Exception(f"Unknown model: {model_name}")

        # save dataset state for recovery
        pickle.dump(dataset, open(join(model_base, "dataset.pkl"), "wb"))

    return dataset