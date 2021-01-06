import pandas
import numpy
import pickle

from os.path import join

from SourceCodeTools.data.sourcetrail.file_utils import *
from SourceCodeTools.common import compact_property


def load_data(node_path, edge_path):
    nodes = unpersist(node_path)
    edges = unpersist(edge_path)

    nodes_ = nodes.rename(mapper={
        'serialized_name': 'name'
    }, axis=1)

    edges_ = edges.rename(mapper={
        'source_node_id': 'src',
        'target_node_id': 'dst'
    }, axis=1)

    return nodes_, edges_


def create_mask(size, idx):
    mask = numpy.full((size,), False, dtype=numpy.bool)
    mask[idx] = True
    return mask


def get_train_test_val_indices(labels, train_frac=0.6, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)
        logging.warning("Random state for splitting dataset is fixed")

    indices = numpy.arange(start=0, stop=labels.size)
    numpy.random.shuffle(indices)

    train = int(indices.size * train_frac)
    test = int(indices.size * (train_frac + (1 - train_frac) / 2))

    logging.info(
        f"Splitting into train {train}, test {test - train}, and validation {indices.size - test} sets"
    )

    return indices[:train], indices[train: test], indices[test:]


class SourceGraphDataset:
    g = None
    nodes = None
    edges = None
    node_types = None
    edge_types = None

    train_frac = None
    random_seed = None
    labels_from = None
    use_node_types = None
    use_edge_types = None
    filter = None
    self_loops = None

    def __init__(self, nodes_path, edges_path,
                 label_from, use_node_types=False,
                 use_edge_types=False, filter=None, self_loops=False,
                 train_frac=0.6, random_seed=None):
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
        :param use_node_types: boolean value, whether to use node types or not
                (node-heterogeneous graph}
        :param use_edge_types: boolean value, whether to use edge types or not
                (edge-heterogeneous graph}
        :param filter: list[str], the types of edges to filter from graph
        """
        # TODO
        # 1. GGNN model

        self.random_seed = random_seed

        self.nodes, self.edges = load_data(nodes_path, edges_path)

        self.compress_node_types()
        self.compress_edge_types()

        if self_loops:
            self.nodes, self.edges = SourceGraphDataset.assess_need_for_self_loops(self.nodes, self.edges)

        if filter is not None:
            for e_type in filter:
                logging.info(f"Filtering edge type {e_type}")
                self.edges = self.edges.query(f"type != {e_type}")

        self.nodes_have_types = use_node_types
        self.edges_have_types = use_edge_types
        self.labels_from = label_from

        self.g = None

        # compact labels
        self.nodes['label'] = self.nodes[label_from]
        self.label_map = compact_property(self.nodes['label'])

        assert any(pandas.isna(self.nodes['label'])) is False

        self.nodes['type_backup'] = self.nodes['type']
        if not self.nodes_have_types:
            self.nodes['type'] = "node_"

        self.edges['type_backup'] = self.edges['type']
        if not self.edges_have_types:
            self.edges['type'] = "edge_"

        logging.info(f"Unique node types in the graph: {len(self.nodes['type'].unique())}")
        logging.info(f"Unique edge types in the graph: {len(self.edges['type'].unique())}")

        self.nodes, self.label_map = self.add_compact_labels()
        self.nodes, self.typed_id_map = self.add_typed_ids()
        self.edges = self.add_node_types_to_edges()

        self.add_splits(train_frac=train_frac)

        self.create_hetero_graph()

        self.update_global_id()

        self.nodes.sort_values('global_graph_id', inplace=True)

        self.splits = SourceGraphDataset.get_global_graph_id_splits(self.nodes)

    @classmethod
    def get_global_graph_id_splits(cls, nodes):

        splits = (
            nodes.query("train_mask == True")['global_graph_id'].values,
            nodes.query("val_mask == True")['global_graph_id'].values,
            nodes.query("test_mask == True")['global_graph_id'].values,
        )

        return splits

    def compress_node_types(self):
        node_type_map = compact_property(self.nodes['type'])
        self.node_types = pd.DataFrame(
            {"str_type": k, "int_type": v} for k, v in compact_property(self.nodes['type']).items()
        )

        self.nodes['type'] = self.nodes['type'].apply(lambda x: node_type_map[x])

    def compress_edge_types(self):
        edge_type_map = compact_property(self.edges['type'])
        self.edge_types = pd.DataFrame(
            {"str_type": k, "int_type": v} for k, v in compact_property(self.edges['type']).items()
        )

        self.edges['type'] = self.edges['type'].apply(lambda x: edge_type_map[x])

    def add_splits(self, train_frac):

        splits = get_train_test_val_indices(
            self.nodes.index,
            train_frac=train_frac,
            random_seed=self.random_seed
        )

        self.nodes['train_mask'] = False
        self.nodes.loc[self.nodes.index[splits[0]], 'train_mask'] = True
        self.nodes['test_mask'] = False
        self.nodes.loc[self.nodes.index[splits[1]], 'test_mask'] = True
        self.nodes['val_mask'] = False
        self.nodes.loc[self.nodes.index[splits[2]], 'val_mask'] = True

    def add_typed_ids(self):
        nodes = self.nodes.copy()

        typed_id_map = {}

        node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))

        for type in nodes['type'].unique():
            # need to use indexes because will need to reference
            # the original table
            type_ind = nodes[nodes['type'] == type].index

            id_map = compact_property(nodes.loc[type_ind, 'id'])

            nodes.loc[type_ind, 'typed_id'] = nodes.loc[type_ind, 'id'].apply(lambda old_id: id_map[old_id])

            typed_id_map[node_types[type]] = id_map

        assert any(pandas.isna(nodes['typed_id'])) is False

        nodes = nodes.astype({"typed_id": "int"})

        return nodes, typed_id_map

    def add_compact_labels(self):

        nodes = self.nodes.copy()

        label_map = compact_property(nodes['label'])

        nodes['compact_label'] = nodes['label'].apply(lambda old_id: label_map[old_id])

        return nodes, label_map

    def add_node_types_to_edges(self):

        edges = self.edges.copy()

        node_type_map = dict(zip(self.nodes['id'].values, self.nodes['type']))

        edges['src_type'] = edges['src'].apply(lambda src_id: node_type_map[src_id])
        edges['dst_type'] = edges['dst'].apply(lambda dst_id: node_type_map[dst_id])

        return edges

    def update_global_id(self):
        orig_id = []
        graph_id = []
        prev_offset = 0

        typed_node_id_maps = self.typed_id_map

        for type in self.g.ntypes:
            from_id, to_id = zip(*typed_node_id_maps[type].items())
            orig_id.extend(from_id)
            graph_id.extend([t + prev_offset for t in to_id])
            prev_offset += self.g.number_of_nodes(type)

        global_map = dict(zip(orig_id, graph_id))

        self.nodes['global_graph_id'] = self.nodes['id'].apply(lambda old_id: global_map[old_id])

    @property
    def typed_node_counts(self):

        typed_node_counts = dict()

        unique_types = self.nodes['type'].unique()

        node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))

        for type_id, type in enumerate(unique_types):
            nodes_of_type = len(self.nodes.query(f"type == {type}"))
            typed_node_counts[node_types[type]] = nodes_of_type

        return typed_node_counts

    def create_hetero_graph(self):

        nodes = self.nodes.copy()
        edges = self.edges.copy()

        typed_node_id = dict(zip(nodes['id'], nodes['typed_id']))

        possible_edge_signatures = edges[['src_type', 'type', 'dst_type']].drop_duplicates(
            ['src_type', 'type', 'dst_type']
        )

        node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))
        edge_types = dict(zip(self.edge_types['int_type'], self.edge_types['str_type']))

        # typed_subgraphs is a dictionary with subset_signature as a key,
        # the dictionary stores directed edge lists
        typed_subgraphs = {}

        for ind, row in possible_edge_signatures.iterrows():
            subgraph_signature = (node_types[row['src_type']], edge_types[row['type']], node_types[row['dst_type']])

            subset = edges.query(
                f"src_type == {row['src_type']} and type=={row['type']} and dst_type=={row['dst_type']}"
            )

            typed_subgraphs[subgraph_signature] = list(
                zip(
                    subset['src'].map(lambda old_id: typed_node_id[old_id]),
                    subset['dst'].map(lambda old_id: typed_node_id[old_id])
                )
            )

        logging.info(
            f"Unique triplet types in the graph: {len(typed_subgraphs.keys())}"
        )

        import dgl, torch
        self.g = dgl.heterograph(typed_subgraphs, self.typed_node_counts)

        node_types = dict(zip(self.node_types['str_type'], self.node_types['int_type']))

        for ntype in self.g.ntypes:
            int_type = node_types[ntype]

            masks = self.nodes.query(
                f"type == {int_type}"
            )[[
                'typed_id', 'train_mask', 'test_mask', 'val_mask', 'compact_label'
            ]].sort_values('typed_id')

            self.g.nodes[ntype].data['train_mask'] = torch.tensor(masks['train_mask'].values, dtype=torch.bool)
            self.g.nodes[ntype].data['test_mask'] = torch.tensor(masks['test_mask'].values, dtype=torch.bool)
            self.g.nodes[ntype].data['val_mask'] = torch.tensor(masks['val_mask'].values, dtype=torch.bool)
            self.g.nodes[ntype].data['labels'] = torch.tensor(masks['compact_label'].values, dtype=torch.int64)

    @classmethod
    def assess_need_for_self_loops(cls, nodes, edges):
        # this is a hack when where are only outgoing connections from this node type
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
    def holdout(cls, nodes, edges, holdout_frac, random_seed):
        """
        Create a set of holdout edges, ensure that there are no orphan nodes after these edges are removed.
        :param nodes:
        :param edges:
        :param holdout_frac:
        :param random_seed:
        :return:
        """

        train, test = split(edges, holdout_frac, random_seed=random_seed)

        nodes, train_edges = ensure_connectedness(nodes, train)

        nodes, test_edges = ensure_valid_edges(nodes, test)

        return nodes, train_edges, test_edges


def split(edges, holdout_frac, random_seed=None):
    if random_seed is not None:
        edges_shuffled = edges.sample(frac=1., random_state=42)
        logging.warning("Random state for splitting edges is fixed")
    else:
        edges_shuffled = edges.sample(frac=1.)

    train_frac = int(edges_shuffled.shape[0] * (1. - holdout_frac))

    train = edges_shuffled.iloc[:train_frac]
    test = edges_shuffled.iloc[train_frac:]
    logging.info(
        f"Splitting edges into train and test set. "
        f"Train: {train.shape[0]}. Test: {test.shape[0]}. Fraction: {holdout_frac}"
    )
    return train, test


def ensure_connectedness(nodes: pandas.DataFrame, edges: pandas.DataFrame):
    """
    Filtering isolated nodes
    :param nodes: DataFrame
    :param edges: DataFrame
    :return:
    """

    logging.info(
        f"Filtering isolated nodes. "
        f"Starting from {nodes.shape[0]} nodes and {edges.shape[0]} edges...",
    )
    unique_nodes = set(edges['src'].values.tolist() +
                       edges['dst'].values.tolist())

    nodes = nodes[
        nodes['id'].apply(lambda nid: nid in unique_nodes)
    ]

    logging.info(
        f"Ending up with {nodes.shape[0]} nodes and {edges.shape[0]} edges"
    )

    return nodes, edges


def ensure_valid_edges(nodes, edges, ignore_src=False):
    """
    Filter edges that link to nodes that do not exist
    :param nodes:
    :param edges:
    :param ignore_src:
    :return:
    """
    print(
        f"Filtering edges to invalid nodes. "
        f"Starting from {nodes.shape[0]} nodes and {edges.shape[0]} edges...",
        end=""
    )

    unique_nodes = set(nodes['id'].values.tolist())

    if not ignore_src:
        edges = edges[
            edges['src'].apply(lambda nid: nid in unique_nodes)
        ]

    edges = edges[
        edges['dst'].apply(lambda nid: nid in unique_nodes)
    ]

    print(
        f"ending up with {nodes.shape[0]} nodes and {edges.shape[0]} edges"
    )

    return nodes, edges


def read_or_create_dataset(args, model_base, labels_from="type"):
    if args.restore_state:
        # i'm not happy with this behaviour that differs based on the flag status
        dataset = pickle.load(open(join(model_base, "dataset.pkl"), "rb"))
    else:
        dataset = SourceGraphDataset(
            args.node_path, args.edge_path,
            label_from=labels_from,
            use_node_types=args.use_node_types,
            use_edge_types=True,
            filter=args.filter_edges,
            self_loops=args.self_loops,
            train_frac=args.train_frac
        )

        # save dataset state for recovery
        pickle.dump(dataset, open(join(model_base, "dataset.pkl"), "wb"))

    return dataset


def test_dataset():
    import sys

    nodes_path = sys.argv[1]
    edges_path = sys.argv[2]

    dataset = SourceGraphDataset(
        nodes_path, edges_path,
        label_from='type',
        use_node_types=True,
        use_edge_types=True,
    )

    print(dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    test_dataset()
