# from collections import Counter
# from itertools import chain
from collections import Counter

import pandas
import numpy
import pickle

from os.path import join

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker, NodeNameMasker
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


def get_name_group(name):
    parts = name.split("@")
    if len(parts) == 1:
        return pd.NA
    elif len(parts) == 2:
        local_name, group = parts
        return group
    return pd.NA


def filter_dst_by_freq(elements, freq=1):
    counter = Counter(elements["dst"])
    allowed = {item for item, count in counter.items() if count >= freq}
    target = elements.query("dst in @allowed", local_dict={"allowed": allowed})
    return target


def create_train_val_test_masks(nodes, train_idx, val_idx, test_idx):
    nodes['train_mask'] = True
    # nodes.loc[train_idx, 'train_mask'] = True
    nodes['val_mask'] = False
    nodes.loc[val_idx, 'val_mask'] = True
    nodes['test_mask'] = False
    nodes.loc[test_idx, 'test_mask'] = True
    nodes['train_mask'] = nodes['train_mask'] ^ (nodes['val_mask'] | nodes['test_mask'])
    starts_with = lambda x: x.startswith("##node_type")
    nodes.loc[nodes.eval("name.map(@starts_with)", local_dict={"starts_with": starts_with}), ['train_mask', 'val_mask', 'test_mask']] = False


def get_train_val_test_indices(indices, train_frac=0.6, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)
        logging.warning("Random state for splitting dataset is fixed")
    else:
        logging.info("Random state is not set")

    indices = indices.to_numpy()

    numpy.random.shuffle(indices)

    train = int(indices.size * train_frac)
    test = int(indices.size * (train_frac + (1 - train_frac) / 2))

    logging.info(
        f"Splitting into train {train}, validation {test - train}, and test {indices.size - test} sets"
    )

    return indices[:train], indices[train: test], indices[test:]


def get_train_val_test_indices_on_packages(nodes, package_names, train_frac=0.6, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)
        logging.warning("Random state for splitting dataset is fixed")
    else:
        logging.info("Random state is not set")

    nodes = nodes.copy()

    package_names = [name.replace("\n", "").replace("-","_").replace(".","_") for name in package_names]

    package_names = numpy.array(package_names)
    numpy.random.shuffle(package_names)

    train = int(package_names.size * train_frac)
    test = int(package_names.size * (train_frac + (1 - train_frac) / 2))

    logging.info(
        f"Splitting into train {train}, validation {test - train}, and test {package_names.size - test} packages"
    )

    train, valid, test = package_names[:train], package_names[train: test], package_names[test:]

    train = set(train.tolist())
    valid = set(valid.tolist())
    test = set(test.tolist())

    nodes["node_package_names"] = nodes["name"].map(lambda name: name.split(".")[0])

    def get_split_indices(split):
        global_types = {val for _, val in node_types.items()}

        def is_global_type(type):
            return type in global_types

        def in_split(name):
            return name in split

        global_nodes = nodes.query(
            "node_package_names.map(@in_split) and type_backup.map(@is_global_type)",
            local_dict={"in_split": in_split, "is_global_type": is_global_type}
        )["id"]
        global_nodes = set(global_nodes.tolist())

        def in_global(node_id):
            return node_id in global_nodes

        ast_nodes = nodes.query("mentioned_in.map(@in_global)", local_dict={"in_global": in_global})["id"]

        split_nodes = global_nodes | set(ast_nodes.tolist())

        def nodes_in_split(node_id):
            return node_id in split_nodes

        return nodes.query("id.map(@nodes_in_split)", local_dict={"nodes_in_split": nodes_in_split}).index

    return get_split_indices(train), get_split_indices(valid), get_split_indices(test)


def get_global_edges():
    """
    :return: Set of global edges and their reverses
    """
    from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping, node_types
    types = set()

    for key, value in special_mapping.items():
        types.add(key)
        types.add(value)

    for _, value in node_types.items():
        types.add(value + "_name")

    return types


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
                 train_frac=0.6, random_seed=None, tokenizer_path=None, min_count_for_objectives=1,
                 no_global_edges=False, remove_reverse=False, package_names=None):
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
        self.min_count_for_objectives = min_count_for_objectives
        self.no_global_edges = no_global_edges
        self.remove_reverse = remove_reverse

        nodes_path = join(data_path, "nodes.bz2")
        edges_path = join(data_path, "edges.bz2")

        self.nodes, self.edges = load_data(nodes_path, edges_path)

        # index is later used for sampling and is assumed to be unique
        assert len(self.nodes) == len(self.nodes.index.unique())
        assert len(self.edges) == len(self.edges.index.unique())

        if self_loops:
            self.nodes, self.edges = SourceGraphDataset.assess_need_for_self_loops(self.nodes, self.edges)

        if filter is not None:
            for e_type in filter.split(","):
                logging.info(f"Filtering edge type {e_type}")
                self.edges = self.edges.query(f"type != '{e_type}'")

        if self.remove_reverse:
            self.remove_reverse_edges()

        if self.no_global_edges:
            self.remove_global_edges()

        if use_node_types is False and use_edge_types is False:
            new_nodes, new_edges = self.create_nodetype_edges()
            self.nodes = self.nodes.append(new_nodes, ignore_index=True)
            self.edges = self.edges.append(new_edges, ignore_index=True)

        self.nodes['type_backup'] = self.nodes['type']
        if not self.nodes_have_types:
            self.nodes['type'] = "node_"
            self.nodes = self.nodes.astype({'type': 'category'})

        self.add_embeddable_flag()

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

        self.add_splits(train_frac=train_frac, package_names=package_names)

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
            embeddable_types = embeddable_types - PythonSharedNodes.tokenizable_types

        embeddable_types |= {"node_type"}

        # self.nodes['embeddable'] = False
        self.nodes.eval(
            "embeddable = type_backup in @embeddable_types",
            local_dict={"embeddable_types": embeddable_types},
            inplace=True
        )

    def op_tokens(self):
        if self.tokenizer_path is None:
            from SourceCodeTools.code.python_tokens_to_bpe_subwords import python_ops_to_bpe
            logging.info("Using heuristic tokenization for ops")

            # def op_tokenize(op_name):
            #     return python_ops_to_bpe[op_name] if op_name in python_ops_to_bpe else None
            return python_ops_to_bpe
        else:
            # from SourceCodeTools.code.python_tokens_to_bpe_subwords import op_tokenize_or_none

            tokenizer = make_tokenizer(load_bpe_model(self.tokenizer_path))

            # def op_tokenize(op_name):
            #     return op_tokenize_or_none(op_name, tokenizer)

            from SourceCodeTools.code.python_tokens_to_bpe_subwords import python_ops_to_literal
            return {
                op_name: tokenizer(op_literal)
                for op_name, op_literal in python_ops_to_literal.items()
            }

        # self.nodes.eval("name_alter_tokens = name.map(@op_tokenize)",
        #                 local_dict={"op_tokenize": op_tokenize}, inplace=True)

    def add_splits(self, train_frac, package_names=None):
        """
        Generates train, validation, and test masks
        Store the masks is pandas table for nodes
        :param train_frac:
        :return:
        """

        assert len(self.nodes.index) == self.nodes.index.max() + 1
        # generate splits for all nodes, additional filtering will be applied later
        # by an objective

        if package_names is None:
            splits = get_train_val_test_indices(
                self.nodes.index,
                train_frac=train_frac, random_seed=self.random_seed
            )
        else:
            splits = get_train_val_test_indices_on_packages(
                self.nodes, package_names,
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

    def create_nodetype_edges(self):
        node_new_id = self.nodes["id"].max() + 1
        edge_new_id = self.edges["id"].max() + 1

        new_nodes = []
        new_edges = []
        added_type_nodes = {}

        node_slice = self.nodes[["id", "type"]].values

        for id, type in node_slice:
            if type not in added_type_nodes:
                added_type_nodes[type] = node_new_id
                node_new_id += 1

                new_nodes.append({
                    "id": added_type_nodes[type],
                    "name": f"##node_type_{type}",
                    "type": "node_type",
                    "mentioned_in": pd.NA
                })

            new_edges.append({
                "id": edge_new_id,
                "type": "node_type",
                "src": added_type_nodes[type],
                "dst": id,
                "file_id": pd.NA,
                "mentioned_in": pd.NA
            })
            edge_new_id += 1

        return pd.DataFrame(new_nodes), pd.DataFrame(new_edges)

    def remove_ast_edges(self):
        global_edges = get_global_edges()
        global_edges.add("subword")
        is_global = lambda type: type in global_edges
        edges = self.edges.query("type_backup.map(@is_global)", local_dict={"is_global": is_global})
        self.nodes, self.edges = ensure_connectedness(self.nodes, edges)

    def remove_global_edges(self):
        global_edges = get_global_edges()
        global_edges.add("global_mention")
        is_ast = lambda type: type not in global_edges
        edges = self.edges.query("type.map(@is_ast)", local_dict={"is_ast": is_ast})
        self.edges = edges
        # self.nodes, self.edges = ensure_connectedness(self.nodes, edges)

    def remove_reverse_edges(self):
        from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping
        global_reverse = {val for _, val in special_mapping.items()}

        not_reverse = lambda type: not (type.endswith("_rev") or type in global_reverse)
        edges = self.edges.query("type.map(@not_reverse)", local_dict={"not_reverse": not_reverse})
        self.edges = edges

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

        # node_mapper = lambda old_id: typed_node_id[old_id]
        # for src_type, type, dst_type, src, dst in edges[['src_type', 'type', 'dst_type', "src", "dst"]].values:
        #     subgraph_signature = (src_type, type, dst_type)
        #     if subgraph_signature in typed_subgraphs:
        #         typed_subgraphs[subgraph_signature].add((node_mapper(src), node_mapper(dst)))
        #     else:
        #         typed_subgraphs[subgraph_signature] = {node_mapper(src), node_mapper(dst)}

        for ind, row in possible_edge_signatures.iterrows():  #
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

    # @classmethod
    # def holdout(cls, nodes, edges, holdout_frac, random_seed):
    #     """
    #     Create a set of holdout edges, ensure that there are no orphan nodes after these edges are removed.
    #     :param nodes:
    #     :param edges:
    #     :param holdout_frac:
    #     :param random_seed:
    #     :return:
    #     """
    #
    #     train, test = split(edges, holdout_frac, random_seed=random_seed)
    #
    #     nodes, train_edges = ensure_connectedness(nodes, train)
    #
    #     nodes, test_edges = ensure_valid_edges(nodes, test)
    #
    #     return nodes, train_edges, test_edges

    # def mark_leaf_nodes(self):
    #     leaf_types = {'subword', "Op", "Constant", "Name"}  # the last is used in graphs without subwords
    #
    #     self.nodes['is_leaf'] = self.nodes['type_backup'].apply(lambda type_: type_ in leaf_types)

    def get_typed_node_id(self, node_id, node_type):
        return self.typed_id_map[node_type][node_id]

    def get_global_node_id(self, node_id, node_type=None):
        return self.node_id_to_global_id[node_id]

    def load_node_names(self):
        """
        :return: DataFrame that contains mappings from nodes to names that appear more than once in the graph
        """
        for_training = self.nodes[
            self.nodes['train_mask'] | self.nodes['test_mask'] | self.nodes['val_mask']
        ][['id', 'type_backup', 'name']]\
            .rename({"name": "serialized_name", "type_backup": "type"}, axis=1)

        node_names = extract_node_names(for_training, 2)
        node_names = filter_dst_by_freq(node_names, freq=self.min_count_for_objectives)

        return node_names
        # path = join(self.data_path, "node_names.bz2")
        # return unpersist(path)

    def load_var_use(self):
        """
        :return: DataFrame that contains mapping from function ids to variable names that appear in those functions
        """
        path = join(self.data_path, "common_function_variable_pairs.bz2")
        var_use = unpersist(path)
        var_use = filter_dst_by_freq(var_use, freq=self.min_count_for_objectives)
        return var_use

    def load_api_call(self):
        path = join(self.data_path, "common_call_seq.bz2")
        api_call = unpersist(path)
        api_call = filter_dst_by_freq(api_call, freq=self.min_count_for_objectives)
        return api_call

    def load_token_prediction(self):
        """
        Return names for all nodes that represent local mentions
        :return: DataFrame that contains mappings from local mentions to names that these mentions represent. Applies
            only to nodes that have subwords and have appeared in a scope (names have `@` in their names)
        """
        if self.use_edge_types:
            edges = self.edges.query("type == 'subword_'")
        else:
            edges = self.edges.query("type_backup == 'subword_'")

        target_nodes = set(edges["dst"].to_list())
        target_nodes = self.nodes.query("id in @target_nodes", local_dict={"target_nodes": target_nodes})[["id", "name"]]

        name_extr = lambda x: x.split('@')[0]
        # target_nodes.eval("group = name.map(@get_group)", local_dict={"get_group": get_name_group}, inplace=True)
        # target_nodes.dropna(axis=0, inplace=True)
        target_nodes.eval("name = name.map(@name_extr)", local_dict={"name_extr": name_extr}, inplace=True)
        target_nodes.rename({"id": "src", "name": "dst"}, axis=1, inplace=True)
        target_nodes = filter_dst_by_freq(target_nodes, freq=self.min_count_for_objectives)
        # target_nodes.eval("cooccurr = dst.map(@occ)", local_dict={"occ": lambda name: name_cooccurr_freq.get(name, Counter())}, inplace=True)

        return target_nodes

    def load_global_edges_prediction(self):

        nodes_path = join(self.data_path, "nodes.bz2")
        edges_path = join(self.data_path, "edges.bz2")

        _, edges = load_data(nodes_path, edges_path)

        global_edges = get_global_edges()
        global_edges = global_edges - {"defines", "defined_in"}  # these edges are already in AST?
        global_edges.add("global_mention")

        is_global = lambda type: type in global_edges
        edges = edges.query("type.map(@is_global)", local_dict={"is_global": is_global})

        edges.rename(
            {
                "source_node_id": "src",
                "target_node_id": "dst"
            }, inplace=True, axis=1
        )

        return edges[["src", "dst"]]

    def load_docstring(self):

        docstrings_path = os.path.join(self.data_path, "common_source_graph_bodies.bz2")

        dosctrings = unpersist(docstrings_path)[["id", "docstring"]]

        from nltk import sent_tokenize

        def normalize(text):
            if text is None or len(text.strip()) == 0:
                return pd.NA
            return "\n".join(sent_tokenize(text)[:3]).replace("\n", " ")

        dosctrings.eval("docstring = docstring.map(@normalize)", local_dict={"normalize": normalize}, inplace=True)
        dosctrings.dropna(axis=0, inplace=True)

        dosctrings.rename({
            "id": "src",
            "docstring": "dst"
        }, axis=1, inplace=True)

        return dosctrings

    def buckets_from_pretrained_embeddings(self, pretrained_path, n_buckets):

        from SourceCodeTools.nlp.embed.fasttext import load_w2v_map
        from SourceCodeTools.nlp import token_hasher
        pretrained = load_w2v_map(pretrained_path)

        import numpy as np

        embs_init = np.random.randn(n_buckets, pretrained.n_dims).astype(np.float32)

        for word in pretrained.keys():
            ind = token_hasher(word, n_buckets)
            embs_init[ind, :] = pretrained[word]

        def op_embedding(op_tokens):
            embedding = None
            for token in op_tokens:
                token_emb = pretrained.get(token, None)
                if embedding is None:
                    embedding = token_emb
                else:
                    embedding = embedding + token_emb
            return embedding

        python_ops_to_bpe = self.op_tokens()
        for op, op_tokens in python_ops_to_bpe.items():
            op_emb = op_embedding(op_tokens)
            if op_emb is not None:
                op_ind = token_hasher(op, n_buckets)
                embs_init[op_ind, :] = op_emb

        return embs_init

    def create_subword_masker(self):
        """
        :return: SubwordMasker for all nodes that have subwords. Suitable for token prediction objective.
        """
        return SubwordMasker(self.nodes, self.edges)

    def create_variable_name_masker(self, tokenizer_path):
        """
        :param tokenizer_path: path to bpe tokenizer
        :return: SubwordMasker for function nodes. Suitable for variable name use prediction objective
        """
        return NodeNameMasker(self.nodes, self.edges, self.load_var_use(), tokenizer_path)

    def create_node_name_masker(self, tokenizer_path):
        """
        :param tokenizer_path: path to bpe tokenizer
        :return: SubwordMasker for function nodes. Suitable for node name use prediction objective
        """
        return NodeNameMasker(self.nodes, self.edges, self.load_node_names(), tokenizer_path)


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
            tokenizer_path=args.tokenizer,
            random_seed=args.random_seed,
            min_count_for_objectives=args.min_count_for_objectives,
            no_global_edges=args.no_global_edges,
            remove_reverse=args.remove_reverse,
            package_names=open(args.packages_file).readlines() if args.packages_file is not None else None
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
        use_node_types=False,
        use_edge_types=True,
    )

    # sm = dataset.create_subword_masker()
    print(dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    test_dataset()
