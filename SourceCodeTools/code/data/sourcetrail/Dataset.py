import pandas
import numpy
import pickle

from os.path import join

from SourceCodeTools.code.data.sourcetrail.file_utils import *
from SourceCodeTools.code.python_ast import PythonSharedNodes
from SourceCodeTools.nlp.embed.bpe import make_tokenizer, load_bpe_model
from SourceCodeTools.tabular.common import compact_property
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types
from SourceCodeTools.code.data.sourcetrail.sourcetrail_extract_node_names import extract_node_names


def load_data(node_path, edge_path):
    nodes = unpersist(node_path)
    edges = unpersist(edge_path)

    nodes_ = nodes.rename(mapper={
        'serialized_name': 'name'
    }, axis=1).astype({
        'type': 'category'
    })

    edges_ = edges.rename(mapper={
        'source_node_id': 'src',
        'target_node_id': 'dst'
    }, axis=1).astype({
        'type': 'category'
    })

    return nodes_, edges_


def create_mask(size, idx):
    mask = numpy.full((size,), False, dtype=numpy.bool)
    mask[idx] = True
    return mask


def create_train_val_test_masks(nodes, train_idx, val_idx, test_idx):
    nodes['train_mask'] = False
    nodes.loc[train_idx, 'train_mask'] = True
    nodes['val_mask'] = False
    nodes.loc[val_idx, 'val_mask'] = True
    nodes['test_mask'] = False
    nodes.loc[test_idx, 'test_mask'] = True


def get_train_val_test_indices(indices, train_frac=0.6, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)
        logging.warning("Random state for splitting dataset is fixed")

    indices = indices.to_numpy()

    numpy.random.shuffle(indices)

    train = int(indices.size * train_frac)
    test = int(indices.size * (train_frac + (1 - train_frac) / 2))

    logging.info(
        f"Splitting into train {train}, validation {test - train}, and test {indices.size - test} sets"
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

    def __init__(self, data_path,
                 label_from, use_node_types=False,
                 use_edge_types=False, filter=None, self_loops=False,
                 train_frac=0.6, random_seed=None, tokenizer_path=None):
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

        self.random_seed = random_seed
        self.nodes_have_types = use_node_types
        self.edges_have_types = use_edge_types
        self.labels_from = label_from
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path

        nodes_path = join(data_path, "nodes.bz2")
        edges_path = join(data_path, "edges.bz2")

        self.nodes, self.edges = load_data(nodes_path, edges_path)

        # index is later used for sampling and is assumed to be unique
        assert len(self.nodes) == len(self.nodes.index.unique())
        assert len(self.edges) == len(self.edges.index.unique())

        if self_loops:
            self.nodes, self.edges = SourceGraphDataset.assess_need_for_self_loops(self.nodes, self.edges)

        if filter is not None:
            for e_type in filter:
                logging.info(f"Filtering edge type {e_type}")
                self.edges = self.edges.query(f"type != {e_type}")

        self.nodes['type_backup'] = self.nodes['type']
        if not self.nodes_have_types:
            self.nodes['type'] = "node_"
            self.nodes = self.nodes.astype({'type': 'category'})

        self.add_embeddable_flag()
        self.add_op_tokens()

        # need to do this to avoid issues insode dgl library
        self.edges['type'] = self.edges['type'].apply(lambda x: f"{x}_")
        self.edges['type_backup'] = self.edges['type']
        if not self.edges_have_types:
            self.edges['type'] = "edge_"
            self.edges = self.edges.astype({'type': 'category'})

        # compact labels
        # self.nodes['label'] = self.nodes[label_from]
        # self.nodes = self.nodes.astype({'label': 'category'})
        # self.label_map = compact_property(self.nodes['label'])
        # assert any(pandas.isna(self.nodes['label'])) is False

        logging.info(f"Unique nodes: {len(self.nodes)}, node types: {len(self.nodes['type'].unique())}")
        logging.info(f"Unique edges: {len(self.edges)}, edge types: {len(self.edges['type'].unique())}")

        # self.nodes, self.label_map = self.add_compact_labels()
        self.add_typed_ids()

        self.add_splits(train_frac=train_frac)

        # self.mark_leaf_nodes()

        self.create_hetero_graph()

        self.update_global_id()

        self.nodes.sort_values('global_graph_id', inplace=True)

        # self.splits = SourceGraphDataset.get_global_graph_id_splits(self.nodes)

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

    def add_embeddable_flag(self):
        embeddable_types = PythonSharedNodes.shared_node_types

        if len(self.nodes.query("type_backup == 'subword'")) > 0:
            # some of the types should not be embedded if subwords were generated
            embeddable_types = embeddable_types - {"#attr#"}
            embeddable_types = embeddable_types - {"#keyword#"}

        # self.nodes['embeddable'] = False
        self.nodes.eval(
            "embeddable = type_backup in @embeddable_types",
            local_dict={"embeddable_types": embeddable_types},
            inplace=True
        )

    def add_op_tokens(self):
        if self.tokenizer_path is None:
            from SourceCodeTools.code.python_tokens_to_bpe_subwords import python_ops_to_bpe
            logging.info("Using heuristic tokenization for ops")

            def op_tokenize(op_name):
                return python_ops_to_bpe[op_name] if op_name in python_ops_to_bpe else None
        else:
            # from SourceCodeTools.code.python_tokens_to_bpe_subwords import python_ops_to_literal
            from SourceCodeTools.code.python_tokens_to_bpe_subwords import op_tokenize_or_none

            tokenizer = make_tokenizer(load_bpe_model(self.tokenizer_path))

            def op_tokenize(op_name):
                return op_tokenize_or_none(op_name, tokenizer)

        self.nodes.eval("name_alter_tokens = name.map(@op_tokenize)",
                        local_dict={"op_tokenize": op_tokenize}, inplace=True)

    def add_splits(self, train_frac):

        # now take only nodes that represent functions. need to add functionality to use classes
        # and variables
        splits = get_train_val_test_indices(
            self.nodes.query(f"type_backup == '{node_types[4096]}'").index,
            train_frac=train_frac, random_seed=self.random_seed
        )

        create_train_val_test_masks(self.nodes, *splits)

    def add_typed_ids(self):
        nodes = self.nodes.copy()

        typed_id_map = {}

        # node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))

        for type in nodes['type'].unique():
            # need to use indexes because will need to reference
            # the original table
            type_ind = nodes[nodes['type'] == type].index

            id_map = compact_property(nodes.loc[type_ind, 'id'])

            nodes.loc[type_ind, 'typed_id'] = nodes.loc[type_ind, 'id'].apply(lambda old_id: id_map[old_id])

            # typed_id_map[node_types[type]] = id_map
            typed_id_map[type] = id_map

        assert any(pandas.isna(nodes['typed_id'])) is False

        nodes = nodes.astype({"typed_id": "int"})

        self.nodes, self.typed_id_map = nodes, typed_id_map
        # return nodes, typed_id_map

    # def add_compact_labels(self):
    #     nodes = self.nodes.copy()
    #     label_map = compact_property(nodes['label'])
    #     nodes['compact_label'] = nodes['label'].apply(lambda old_id: label_map[old_id])
    #     return nodes, label_map

    def add_node_types_to_edges(self, nodes, edges):

        # nodes = self.nodes
        # edges = self.edges.copy()

        node_type_map = dict(zip(nodes['id'].values, nodes['type']))

        edges['src_type'] = edges['src'].apply(lambda src_id: node_type_map[src_id])
        edges['dst_type'] = edges['dst'].apply(lambda dst_id: node_type_map[dst_id])
        edges = edges.astype({'src_type': 'category', 'dst_type': 'category'})

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
        import torch
        for ntype in self.g.ntypes:
            self.g.nodes[ntype].data['global_graph_id'] = torch.LongTensor(
                list(map(lambda x: global_map[x], self.g.nodes[ntype].data['original_id'].tolist()))
            )

        self.node_id_to_global_id = dict(zip(self.nodes["id"], self.nodes["global_graph_id"]))

    @property
    def typed_node_counts(self):

        typed_node_counts = dict()

        unique_types = self.nodes['type'].unique()

        # node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))

        for type_id, type in enumerate(unique_types):
            nodes_of_type = len(self.nodes.query(f"type == '{type}'"))
            # typed_node_counts[node_types[type]] = nodes_of_type
            typed_node_counts[type] = nodes_of_type

        return typed_node_counts

    def create_hetero_graph(self):

        nodes = self.nodes.copy()
        edges = self.edges.copy()
        edges = self.add_node_types_to_edges(nodes, edges)

        typed_node_id = dict(zip(nodes['id'], nodes['typed_id']))

        possible_edge_signatures = edges[['src_type', 'type', 'dst_type']].drop_duplicates(
            ['src_type', 'type', 'dst_type']
        )

        # node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))
        # edge_types = dict(zip(self.edge_types['int_type'], self.edge_types['str_type']))

        # typed_subgraphs is a dictionary with subset_signature as a key,
        # the dictionary stores directed edge lists
        typed_subgraphs = {}

        for ind, row in possible_edge_signatures.iterrows():
            # subgraph_signature = (node_types[row['src_type']], edge_types[row['type']], node_types[row['dst_type']])
            subgraph_signature = (row['src_type'], row['type'], row['dst_type'])

            subset = edges.query(
                f"src_type == '{row['src_type']}' and type == '{row['type']}' and dst_type == '{row['dst_type']}'"
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

        # node_types = dict(zip(self.node_types['str_type'], self.node_types['int_type']))

        for ntype in self.g.ntypes:
            # int_type = node_types[ntype]

            node_data = self.nodes.query(
                f"type == '{ntype}'"
            )[[
                'typed_id', 'train_mask', 'test_mask', 'val_mask', 'id' # 'compact_label',
            ]].sort_values('typed_id')

            self.g.nodes[ntype].data['train_mask'] = torch.tensor(node_data['train_mask'].values, dtype=torch.bool)
            self.g.nodes[ntype].data['test_mask'] = torch.tensor(node_data['test_mask'].values, dtype=torch.bool)
            self.g.nodes[ntype].data['val_mask'] = torch.tensor(node_data['val_mask'].values, dtype=torch.bool)
            # self.g.nodes[ntype].data['labels'] = torch.tensor(node_data['compact_label'].values, dtype=torch.int64)
            self.g.nodes[ntype].data['typed_id'] = torch.tensor(node_data['typed_id'].values, dtype=torch.int64)
            self.g.nodes[ntype].data['original_id'] = torch.tensor(node_data['id'].values, dtype=torch.int64)

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

    # def mark_leaf_nodes(self):
    #     leaf_types = {'subword', "Op", "Constant", "Name"}  # the last is used in graphs without subwords
    #
    #     self.nodes['is_leaf'] = self.nodes['type_backup'].apply(lambda type_: type_ in leaf_types)

    def get_typed_node_id(self, node_id, node_type):
        return self.typed_id_map[node_type][node_id]

    def get_global_node_id(self, node_id, node_type=None):
        return self.node_id_to_global_id[node_id]

    def load_node_names(self):
        for_training = self.nodes[
            self.nodes['train_mask'] | self.nodes['test_mask'] | self.nodes['val_mask']
        ][['id', 'type_backup', 'name']]\
            .rename({"name": "serialized_name", "type_backup": "type"}, axis=1)

        node_names = extract_node_names(for_training, 1)

        return node_names
        # path = join(self.data_path, "node_names.bz2")
        # return unpersist(path)

    def load_var_use(self):
        path = join(self.data_path, "common_function_variable_pairs.bz2")
        return unpersist(path)

    def load_api_call(self):
        path = join(self.data_path, "common_call_seq.bz2")
        return unpersist(path)


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
            # args.node_path, args.edge_path,
            args.data_path,
            label_from=labels_from,
            use_node_types=args.use_node_types,
            use_edge_types=args.use_edge_types,
            filter=args.filter_edges,
            self_loops=args.self_loops,
            train_frac=args.train_frac,
            tokenizer_path=args.tokenizer
        )

        # save dataset state for recovery
        pickle.dump(dataset, open(join(model_base, "dataset.pkl"), "wb"))

    return dataset


def test_dataset():
    import sys

    data_path = sys.argv[1]
    # nodes_path = sys.argv[1]
    # edges_path = sys.argv[2]

    dataset = SourceGraphDataset(
        data_path,
        # nodes_path, edges_path,
        label_from='type',
        use_node_types=True,
        use_edge_types=True,
    )

    print(dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    test_dataset()
