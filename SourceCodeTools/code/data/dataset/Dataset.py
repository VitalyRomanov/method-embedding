import pickle
from collections import Counter
from functools import partial
from os.path import join
from typing import List, Optional

import dgl
import numpy
import pandas
import torch

from SourceCodeTools.code.ast.python_ast2 import PythonSharedNodes
from SourceCodeTools.code.data.GraphStorage import OnDiskGraphStorage
from SourceCodeTools.code.data.SQLiteStorage import SQLiteStorage
from SourceCodeTools.code.data.dataset.SubwordMasker import SubwordMasker, NodeNameMasker, NodeClfMasker
from SourceCodeTools.code.data.dataset.partition_strategies import SGPartitionStrategies, SGLabelSpec
from SourceCodeTools.code.data.file_utils import *
# from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types
from SourceCodeTools.code.data.sourcetrail.sourcetrail_extract_node_names import extract_node_names
# from SourceCodeTools.code.ast.python_ast import PythonSharedNodes
from SourceCodeTools.models.graph.TargetLoader import TargetLoader
from SourceCodeTools.nlp import token_hasher
from SourceCodeTools.nlp.embed.bpe import make_tokenizer, load_bpe_model
from SourceCodeTools.tabular.common import compact_property


pandas.options.mode.chained_assignment = None


def filter_dst_by_freq(elements, freq=1):
    counter = Counter(elements["dst"])
    allowed = {item for item, count in counter.items() if count >= freq}
    target = elements.query("dst in @allowed", local_dict={"allowed": allowed})
    return target


class DatasetCache:
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path).joinpath("__cache__")

        if not self.cache_path.is_dir():
            self.cache_path.mkdir()

        self.warned = False

    def get_path_for_level(self, level):
        if level is not None:
            cache_path = self.cache_path.joinpath(level)
        else:
            cache_path = self.cache_path
        return cache_path

    @staticmethod
    def get_filename(cache_path, cache_key):
        return cache_path.joinpath(f"{cache_key}.pkl")

    def load_cached(self, cache_key, level=None):
        cache_path = self.get_path_for_level(level)
        filename = self.get_filename(cache_path, cache_key)

        if filename.is_file():
            if not self.warned:
                logging.warning("Using cached results. Remove cache if you have updated the dataset.")
                self.warned = True
            return pickle.load(open(filename, "rb"))
        else:
            return None

    def write_to_cache(self, obj, cache_key, level=None):
        cache_path = self.get_path_for_level(level)

        if not cache_path.is_dir():
            cache_path.mkdir()

        filename = self.get_filename(cache_path, cache_key)
        pickle.dump(obj, open(filename, "wb"))


class NodeStringPool:
    def __init__(self, node_id2string_id, string_id2string):
        self.node_id2string_id = node_id2string_id
        self.string_id2string = string_id2string

    def __getitem__(self, node_id):
        return self.string_id2string[self.node_id2string_id[node_id]]

    def items(self):
        for node_id, name_id in self.node_id2string_id.items():
            yield node_id, self.string_id2string[name_id]


class NodeNamePool(NodeStringPool):
    def __init__(self, node_id2name_id, name_id2name):
        super(NodeNamePool, self).__init__(node_id2name_id, name_id2name)


class NodeTypePool(NodeStringPool):
    def __init__(self, node_id2type_id, type_id2type):
        super(NodeTypePool, self).__init__(node_id2type_id, type_id2type)


class PartitionIndex:
    def __init__(self, partition_table=None):
        self._partition_label_order = ["train_mask", "test_mask", "val_mask"]
        self._index = None
        self._exclusive_label = None

        if partition_table is not None:
            self.populate_index(partition_table)

    def populate_index(self, partition_table):
        self._index = dict()
        self._train_set = set()
        self._test_set = set()
        self._val_set = set()

        columns = ["id"] + self._partition_label_order
        # variable order depends on label order in __init__
        assert self._partition_label_order == ["train_mask", "test_mask", "val_mask"]
        for id_, train_mask, test_mask, val_mask in partition_table[columns].values:
            self._index[id_] = (train_mask, test_mask, val_mask)
            if train_mask is True:
                self._train_set.add(id_)
            if test_mask is True:
                self._test_set.add(id_)
            if val_mask is True:
                self._val_set.add(id_)

    def create_exclusive(self, exclusive_label):
        new_index = self.__class__()
        new_index._index = self._index
        new_index._exclusive_label = self._partition_label_order.index(exclusive_label)
        return new_index

    def is_partition(self, node_id, partition_label):
        return node_id in self.get_partition_ids(partition_label)

    def get_partition_ids(self, partition_label):
        if partition_label == "any":
            partition_label_source = {"any": self._train_set | self._test_set | self._val_set}
        else:
            partition_label_source = {
                "train_mask": self._train_set,
                "test_mask": self._test_set,
                "val_mask": self._val_set,
            }
        if partition_label in partition_label_source:
            p = partition_label_source[partition_label]
        else:
            raise ValueError(
                f"Supported partition labels are {list(partition_label_source.keys())}, "
                f"but {repr(partition_label)} is given"
            )
        return p

    def is_validation(self, node_id):
        return node_id in self._val_set

    def is_test(self, node_id):
        return node_id in self._test_set

    def all_ids(self):
        return self._train_set | self._val_set | self._test_set

    def __contains__(self, item):
        return item in self._index

    def __getitem__(self, item):
        if self._exclusive_label is None:
            return self._index[item]
        else:
            return self._index[item][self._exclusive_label]

    def get(self, item, default):
        if item not in self._index:
            return default
        return self[item]


class SourceGraphDataset:
    node_types = None
    edge_types = None

    train_frac = None
    random_seed = None
    labels_from = None
    use_node_types = None
    use_edge_types = None
    filter_edges = None
    self_loops = None

    partition_columns_names = {
        "train": "train_mask",
        "val": "val_mask",
        "test": "test_mask",
        "any": "any"
    }

    def __init__(
            self, data_path: Union[str, Path], partition, use_node_types: bool = False, use_edge_types: bool = False,
            filter_edges: Optional[List[str]] = None, self_loops: bool = False,
            train_frac: float = 0.6, random_seed: Optional[int] = None, tokenizer_path: Union[str, Path] = None,
            min_count_for_objectives: int = 1,
            no_global_edges: bool = False, remove_reverse: bool = False, custom_reverse: Optional[List[str]] = None,
            # package_names: Optional[List[str]] = None,
            restricted_id_pool: Optional[List[int]] = None, use_ns_groups: bool = False,
            subgraph_id_column=None, subgraph_partition=None
    ):
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
                the relationship between the new graph id and the original node id from the training data
        :param data_path: path to the directory with dataset files stored in `bz2` format
        :param use_node_types:  whether to use node types in the graph
        :param use_edge_types:  whether to use edge types in the graph
        :param filter_edges: edge types to be removed from the graph
        :param self_loops: whether to include self-loops
        :param train_frac: fraction of the nodes that will be used for training
        :param random_seed: seed for generating random splits
        :param tokenizer_path:  path to bpe tokenizer, needed to process op names correctly
        :param min_count_for_objectives: minimum degree of nodes, after which they are excluded from training data
        :param no_global_edges: whether to remove global edges from the dataset
        :param remove_reverse: whether to remove reverse edges from the dataset
        :param custom_reverse: list of edges for which reverse types should be added.
            Used together with `remove_reverse`
        :param restricted_id_pool: path to csv file with column `node_id` that stores nodes that should be involved into
            training and testing
        :param use_ns_groups: currently not used

        """
        self.random_seed = random_seed
        self.use_node_types = use_node_types
        self.use_edge_types = use_edge_types
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.min_count_for_objectives = min_count_for_objectives
        self.no_global_edges = no_global_edges
        self.remove_reverse = remove_reverse
        self.custom_reverse = custom_reverse
        self.subgraph_id_column = subgraph_id_column
        self.subgraph_partition = subgraph_partition
        self.partition = PartitionIndex(unpersist(partition))
        self.n_buckets = 200000
        self._cache = DatasetCache(self.data_path)

        self.use_ns_groups = use_ns_groups

        self._open_dataset_db()

        self.edge_types_to_remove = set()
        if filter_edges is not None:
            self._filter_edges(filter_edges)

        if self.remove_reverse:
            self._remove_reverse_edges()

        if self.no_global_edges:
            self._remove_global_edges()

        # self.edge_types_to_remove = set(f"{etype}_" for etype in self.edge_types_to_remove)

        # if self.custom_reverse is not None:
        #     self._add_custom_reverse()

        # if use_node_types is False and use_edge_types is False:
        #     new_nodes, new_edges = self._create_nodetype_edges(self.nodes, self.edges)
        #     self.nodes = self.nodes.append(new_nodes, ignore_index=True)
        #     self.edges = self.edges.append(new_edges, ignore_index=True)

        # self.nodes['type_backup'] = self.nodes['type']
        # if not self.nodes_have_types:
        #     self.nodes['type'] = "node_"
        #     self.nodes = self.nodes.astype({'type': 'category'})

        # self.embeddable_names, self.name_pool = self._get_embeddable_names()

        # # need to do this to avoid issues inside dgl library
        # self.edges['type'] = self.edges['type'].apply(lambda x: f"{x}_")
        # self.edges['type_backup'] = self.edges['type']
        # if not self.edges_have_types:
        #     self.edges['type'] = "edge_"
        #     self.edges = self.edges.astype({'type': 'category'})

        # compact labels
        # self.nodes['label'] = self.nodes[label_from]
        # self.nodes = self.nodes.astype({'label': 'category'})
        # self.label_map = compact_property(self.nodes['label'])
        # assert any(pandas.isna(self.nodes['label'])) is False

        # logging.info(f"Unique nodes: {len(self.nodes)}, node types: {len(self.nodes['type'].unique())}")
        # logging.info(f"Unique edges: {len(self.edges)}, edge types: {len(self.edges['type'].unique())}")

    def _remove_edges_with_restricted_types(self, edges):
        edges.query(
            "type not in @restricted_types", local_dict={"restricted_types": self.edge_types_to_remove}, inplace=True
        )

    def _open_dataset_db(self):
        # self.dataset_db = N4jGraphStorage()
        dataset_db_path = join(self.data_path, "dataset.db")
        if not os.path.isfile(dataset_db_path):
            self.dataset_db = OnDiskGraphStorage(dataset_db_path)
            self.dataset_db.import_from_files(self.data_path)
        else:
            self.dataset_db = OnDiskGraphStorage(dataset_db_path)
        #     self.dataset_db.import_from_files(self.data_path)

    def _filter_edges(self, types_to_filter):
        # logging.info(f"Filtering edge types: {types_to_filter}")
        self.edge_types_to_remove.update(types_to_filter)

    # def _get_all_embeddable_names(self):
    #     id2embeddable_name = dict()
    #     name_pool = dict()
    #     name_pool_rev = dict()
    #     for nodes in self.dataset_db.iterate_nodes_with_chunks():
    #         for node_id, embeddable_name in zip(
    #                 nodes["id"],
    #                 nodes["name"].apply(self.get_embeddable_name)
    #         ):
    #             if embeddable_name not in name_pool:
    #                 name_pool[embeddable_name] = len(name_pool)
    #                 name_pool_rev[name_pool[embeddable_name]] = embeddable_name
    #             id2embeddable_name[node_id] = name_pool[embeddable_name]
    #     return id2embeddable_name, name_pool_rev

    def _get_embeddable_names(self, nodes):
        id2embeddable_name = dict()
        name_pool = dict()
        name_pool_rev = dict()
        for node_id, embeddable_name in zip(
                nodes["id"],
                nodes["name"].apply(self._get_embeddable_name)
        ):
            if embeddable_name not in name_pool:
                name_pool[embeddable_name] = len(name_pool)
                name_pool_rev[name_pool[embeddable_name]] = embeddable_name
            id2embeddable_name[node_id] = name_pool[embeddable_name]
        return NodeNamePool(id2embeddable_name, name_pool_rev)

    # def _add_embeddable_flag(self):
    #     embeddable_types = PythonSharedNodes.shared_node_types | set(list(node_types.values()))
    #
    #     if len(self.nodes.query("type_backup == 'subword'")) > 0:
    #         # some of the types should not be embedded if subwords were generated
    #         embeddable_types = embeddable_types - PythonSharedNodes.tokenizable_types
    #
    #     embeddable_types |= {"node_type"}
    #
    #     # self.nodes['embeddable'] = False
    #     self.nodes.eval(
    #         "embeddable = type_backup in @embeddable_types",
    #         local_dict={"embeddable_types": embeddable_types},
    #         inplace=True
    #     )
    #
    #     self.nodes["embeddable_name"] = self.nodes["name"].apply(self.get_embeddable_name)

    def _op_tokens(self):
        if self.tokenizer_path is None:
            from SourceCodeTools.code.ast.python_tokens_to_bpe_subwords import python_ops_to_bpe
            # logging.info("Using heuristic tokenization for ops")

            # def op_tokenize(op_name):
            #     return python_ops_to_bpe[op_name] if op_name in python_ops_to_bpe else None
            return python_ops_to_bpe
        else:
            # from SourceCodeTools.code.python_tokens_to_bpe_subwords import op_tokenize_or_none

            tokenizer = make_tokenizer(load_bpe_model(self.tokenizer_path))

            # def op_tokenize(op_name):
            #     return op_tokenize_or_none(op_name, tokenizer)

            from SourceCodeTools.code.ast.python_tokens_to_bpe_subwords import python_ops_to_literal
            return {
                op_name: tokenizer(op_literal)
                for op_name, op_literal in python_ops_to_literal.items()
            }

        # self.nodes.eval("name_alter_tokens = name.map(@op_tokenize)",
        #                 local_dict={"op_tokenize": op_tokenize}, inplace=True)

    # def _add_splits(self, train_frac, package_names=None, restricted_id_pool=None):
    #     """
    #     Generates train, validation, and test masks
    #     Store the masks is pandas table for nodes
    #     :param train_frac:
    #     :return:
    #     """
    #
    #     self.nodes.reset_index(drop=True, inplace=True)
    #     assert len(self.nodes.index) == self.nodes.index.max() + 1
    #     # generate splits for all nodes, additional filtering will be applied later
    #     # by an objective
    #
    #     if package_names is None:
    #         splits = self.get_train_val_test_indices(
    #             self.nodes.index,
    #             train_frac=train_frac, random_seed=self.random_seed
    #         )
    #     else:
    #         splits = self.get_train_val_test_indices_on_packages(
    #             self.nodes, package_names,
    #             train_frac=train_frac, random_seed=self.random_seed
    #         )
    #
    #     self.create_train_val_test_masks(self.nodes, *splits)
    #
    #     if restricted_id_pool is not None:
    #         node_ids = set(pd.read_csv(restricted_id_pool)["node_id"].tolist()) | \
    #                    set(self.nodes.query("type_backup == 'FunctionDef' or type_backup == 'mention'")["id"].tolist())
    #         to_keep = self.nodes["id"].apply(lambda id_: id_ in node_ids)
    #         self.nodes["train_mask"] = self.nodes["train_mask"] & to_keep
    #         self.nodes["test_mask"] = self.nodes["test_mask"] & to_keep
    #         self.nodes["val_mask"] = self.nodes["val_mask"] & to_keep

    # def _add_typed_ids(self):
    #     nodes = self.nodes.copy()
    #
    #     typed_id_map = {}
    #
    #     # node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))
    #
    #     for type_ in nodes['type'].unique():
    #         type_mask = nodes['type'] == type_
    #         id_map = compact_property(nodes.loc[type_mask, 'id'])
    #         nodes.loc[type_mask, 'typed_id'] = nodes.loc[type_mask, 'id'].apply(lambda old_id: id_map[old_id])
    #         typed_id_map[type_] = id_map
    #
    #     assert any(pandas.isna(nodes['typed_id'])) is False
    #
    #     nodes = nodes.astype({"typed_id": "int"})
    #
    #     self.nodes, self.typed_id_map = nodes, typed_id_map
    #     # return nodes, typed_id_map

    # def add_compact_labels(self):
    #     nodes = self.nodes.copy()
    #     label_map = compact_property(nodes['label'])
    #     nodes['compact_label'] = nodes['label'].apply(lambda old_id: label_map[old_id])
    #     return nodes, label_map

    @staticmethod
    def _add_node_types_to_edges(nodes, edges):

        node_type_map = dict(zip(nodes['id'].values, nodes['type']))

        edges.eval("src_type = src.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges.eval("dst_type = dst.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges = edges.astype({'src_type': 'category', 'dst_type': 'category'}, copy=False)

        return edges

    # @staticmethod
    # def _create_nodetype_edges(nodes, edges):
    #     node_new_id = nodes["id"].max() + 1
    #     edge_new_id = edges["id"].max() + 1
    #
    #     new_nodes = []
    #     new_edges = []
    #     added_type_nodes = {}
    #
    #     node_slice = nodes[["id", "type"]].values
    #
    #     for id, type in node_slice:
    #         if type not in added_type_nodes:
    #             added_type_nodes[type] = node_new_id
    #             node_new_id += 1
    #
    #             new_nodes.append({
    #                 "id": added_type_nodes[type],
    #                 "name": f"##node_type_{type}",
    #                 "type": "node_type",
    #                 "mentioned_in": pd.NA
    #             })
    #
    #         new_edges.append({
    #             "id": edge_new_id,
    #             "type": "node_type",
    #             "src": added_type_nodes[type],
    #             "dst": id,
    #             "file_id": pd.NA,
    #             "mentioned_in": pd.NA
    #         })
    #         edge_new_id += 1
    #
    #     return pd.DataFrame(new_nodes), pd.DataFrame(new_edges)

    # def _remove_ast_edges(self):
    #     global_edges = self.get_global_edges()
    #     global_edges.add("subword")
    #     is_global = lambda type: type in global_edges
    #     edges = self.edges.query("type_backup.map(@is_global)", local_dict={"is_global": is_global})
    #     self.nodes, self.edges = self.ensure_connectedness(self.nodes, edges)

    def _remove_global_edges(self):
        global_edges = self.get_global_edges()
        self.edge_types_to_remove.update(global_edges)
        # global_edges.add("global_mention")
        # global_edges |= set(edge + "_rev" for edge in global_edges)
        # is_ast = lambda type: type not in global_edges
        # edges = self.edges.query("type.map(@is_ast)", local_dict={"is_ast": is_ast})
        # self.edges = edges
        # self.nodes, self.edges = ensure_connectedness(self.nodes, edges)

    def _remove_reverse_edges(self):
        from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping
        global_reverse = {key for key, val in special_mapping.items()}
        self.edge_types_to_remove.update(global_reverse)

        all_edge_types = self.dataset_db.get_edge_type_descriptions()
        self.edge_types_to_remove.update(filter(lambda edge_type: edge_type.endswith("_rev"), all_edge_types))

    def _add_custom_reverse(self, edges):
        to_reverse = edges.query("type in @custom_reverse", local_dict={"custom_reverse": self.custom_reverse})

        to_reverse.eval("type = type.map(@add_rev)", local_dict={"add_rev": lambda type_: type_ + "_rev"}, inplace=True)
        to_reverse.rename({"src": "dst", "dst": "src"}, axis=1, inplace=True)
        # to_reverse.eval("src = src", inplace=True)
        # to_reverse["type"] = to_reverse["type"].apply(lambda type_: type_ + "_rev")
        # tmp = to_reverse["src"].copy()
        # to_reverse["src"] = to_reverse["dst"]
        # to_reverse["dst"] = tmp

        new_id = edges["id"].max() + 1
        to_reverse["id"] = range(new_id, new_id + len(to_reverse))

        return edges.append(to_reverse[["src", "dst", "type"]])

    # def _update_global_id(self):
    #     orig_id = []
    #     graph_id = []
    #     prev_offset = 0
    #
    #     typed_node_id_maps = self.typed_id_map
    #
    #     for type in self.g.ntypes:
    #         from_id, to_id = zip(*typed_node_id_maps[type].items())
    #         orig_id.extend(from_id)
    #         graph_id.extend([t + prev_offset for t in to_id])
    #         prev_offset += self.g.number_of_nodes(type)
    #
    #     global_map = dict(zip(orig_id, graph_id))
    #
    #     self.nodes['global_graph_id'] = self.nodes['id'].apply(lambda old_id: global_map[old_id])
    #     import torch
    #     for ntype in self.g.ntypes:
    #         self.g.nodes[ntype].data['global_graph_id'] = torch.LongTensor(
    #             list(map(lambda x: global_map[x], self.g.nodes[ntype].data['original_id'].tolist()))
    #         )
    #
    #     self.node_id_to_global_id = dict(zip(self.nodes["id"], self.nodes["global_graph_id"]))

    # @property
    # def typed_node_counts(self):
    #
    #     typed_node_counts = dict()
    #
    #     unique_types = self.nodes['type'].unique()
    #
    #     # node_types = dict(zip(self.node_types['int_type'], self.node_types['str_type']))
    #
    #     for type_id, type in enumerate(unique_types):
    #         nodes_of_type = len(self.nodes.query(f"type == '{type}'"))
    #         # typed_node_counts[node_types[type]] = nodes_of_type
    #         typed_node_counts[type] = nodes_of_type
    #
    #     return typed_node_counts

    def _create_hetero_graph(self, nodes, edges):

        def get_canonical_type(dtype):
            default_values = [
                ("int64", "int64"),
                ("int32", "int64"),
                ("bool", "bool"),
                ("Int32", "int32"),
                ("Int64", "int64")
            ]
            for candidate, value in default_values:
                if dtype == candidate:
                    return value
            else:
                raise KeyError("Unrecognized type: ", dtype)

        def get_torch_dtype(canonical_type):
            torch_types = {
                "int32": torch.int32,
                "int64": torch.int64,
                "bool": torch.bool,
            }
            return torch_types[canonical_type]

        # def get_default_value(canonical_type):
        #     default = {
        #         "int32": -1,
        #         "int64": -1,
        #         "bool": False,
        #     }
        #     return default[canonical_type]

        def unpack_node_data(nodes):
            node_data = {}
            for col_name, dtype in zip(nodes.columns, nodes.dtypes):
                if col_name in {"id", "type", "name", "type_backup"}:
                    continue
                # if col_name == "label":
                #     labels = nodes[["id", "label"]].dropna()
                #     mapping = dict(zip(labels["id"], labels[col_name]))
                #     has_label_mask = torch.tensor(
                #         [node_id in mapping for node_id in range(nodes["id"].max() + 1)],
                #         dtype=torch.bool
                #     )
                #     node_data["has_label"] = has_label_mask
                # else:
                mapping = dict(zip(nodes["id"], nodes[col_name]))
                node_data[col_name] = torch.tensor(
                    [mapping.get(node_id, -1) for node_id in range(num_nodes)],
                    dtype=get_torch_dtype(get_canonical_type(dtype))
                )
            return node_data

        edges = self._add_node_types_to_edges(nodes, edges)

        possible_edge_signatures = edges[['src_type', 'type', 'dst_type']].drop_duplicates(
            ['src_type', 'type', 'dst_type']
        )

        typed_subgraphs = {}

        nodes_table = SQLiteStorage(":memory:")
        nodes_table.add_records(nodes, "nodes", create_index=["id", "type"])
        edges_table = SQLiteStorage(":memory:")
        edges_table.add_records(edges, "edges", create_index=["src_type", "type", "dst_type"])

        for src_type, type, dst_type in possible_edge_signatures[["src_type", "type", "dst_type"]].values:  #
            subgraph_signature = (src_type, type, dst_type)

            subset = edges_table.query(
                f"select * from edges where src_type = '{src_type}' and type = '{type}' and dst_type = '{dst_type}'"
            )
            # subset = edges.query(
            #     f"src_type == '{src_type}' and type == '{type}' and dst_type == '{dst_type}'"
            # )

            typed_subgraphs[subgraph_signature] = list(
                zip(
                    subset['src'],
                    subset['dst']
                )
            )

        # logging.info(
        #     f"Unique triplet types in the graph: {len(typed_subgraphs.keys())}"
        # )

        num_nodes = nodes["id"].max() + 1
        graph = dgl.heterograph(typed_subgraphs, num_nodes_dict={ntype: num_nodes for ntype in nodes["type"].unique()})

        def attach_node_data(graph, nodes):

            node_data = unpack_node_data(nodes)

            # def add_node_data_to_graph(graph, col_name, data, canonical_type):
            #     data = torch.tensor(data, dtype=get_torch_dtype(canonical_type))
            #     graph.nodes[ntype].data[col_name] = data

            for ntype in graph.ntypes:
                valid_subset = set(nodes_table.query(f"select id from nodes where type = '{ntype}'")["id"])
                node_ids_for_ntype = graph.nodes(ntype)
                valid_mask = torch.tensor(
                    # check memory consumption here
                    [node_id in valid_subset for node_id in node_ids_for_ntype.tolist()],
                    dtype=get_torch_dtype("bool")
                )
                graph.nodes[ntype].data["valid"] = valid_mask
                for col_name in node_data:
                    graph.nodes[ntype].data[col_name] = node_data[col_name]

        attach_node_data(graph, nodes)
        return graph

        # node_types = dict(zip(self.node_types['str_type'], self.node_types['int_type']))

        # for ntype in self.g.ntypes:
        #     # int_type = node_types[ntype]
        #
        #     node_data = self.nodes.query(
        #         f"type == '{ntype}'"
        #     )[[
        #         'typed_id', 'train_mask', 'test_mask', 'val_mask', 'id' # 'compact_label',
        #     ]].sort_values('typed_id')
        #
        #     self.g.nodes[ntype].data['train_mask'] = torch.tensor(node_data['train_mask'].values, dtype=torch.bool)
        #     self.g.nodes[ntype].data['test_mask'] = torch.tensor(node_data['test_mask'].values, dtype=torch.bool)
        #     self.g.nodes[ntype].data['val_mask'] = torch.tensor(node_data['val_mask'].values, dtype=torch.bool)
        #     # self.g.nodes[ntype].data['labels'] = torch.tensor(node_data['compact_label'].values, dtype=torch.int64)
        #     self.g.nodes[ntype].data['typed_id'] = torch.tensor(node_data['typed_id'].values, dtype=torch.int64)
        #     self.g.nodes[ntype].data['original_id'] = torch.tensor(node_data['id'].values, dtype=torch.int64)

    # @staticmethod
    # def _assess_need_for_self_loops(nodes, edges):
    #     # this is a hack when where are only outgoing connections from this node type
    #     need_self_loop = set(edges['src'].values.tolist()) - set(edges['dst'].values.tolist())
    #     for nid in need_self_loop:
    #         edges = edges.append({
    #             "id": -1,
    #             "type": 99,
    #             "src": nid,
    #             "dst": nid
    #         }, ignore_index=True)
    #
    #     return nodes, edges

    # def _get_negative_sample_groups(self):
    #     # TODO
    #     return self.nodes[["id", "mentioned_in"]].dropna(axis=0)

    @staticmethod
    def _get_name_group(name):
        parts = name.split("@")
        if len(parts) == 1:
            return pd.NA
        elif len(parts) == 2:
            local_name, group = parts
            return group
        return pd.NA

    @staticmethod
    def _get_embeddable_name(name):
        if "@" in name:
            return name.split("@")[0]
        elif "_0x" in name:
            return name.split("_0x")[0]
        else:
            return name

    # @classmethod
    # def load(cls, path, args):
    #     dataset = pickle.load(open(path, "rb"))
    #     dataset.data_path = args["data_path"]
    #     if dataset.tokenizer_path is not None:
    #         dataset.tokenizer_path = args["tokenizer"]
    #     return dataset

    def _get_node_name2bucket_mapping(self, node_id2name):
        return {key: token_hasher(val, self.n_buckets) for key, val in node_id2name.items()}

    # def _prepare_subgraph(self, seed_nodes, number_of_hops):
    #     seen_nodes = set(seed_nodes)
    #     nodes_for_query = seed_nodes
    #
    #     for _ in range(number_of_hops):
    #         inbound_neighbors = set(self.dataset_db.get_inbound_neighbors(nodes_for_query))
    #         nodes_for_query = inbound_neighbors - seen_nodes
    #         seen_nodes.update(inbound_neighbors)
    #
    #     nodes, edges = self.dataset_db.get_subgraph_from_node_ids(seen_nodes)
    #
    #     edges.query("type not in @restricted_types", local_dict={"restricted_types": self.edge_types_to_remove}, inplace=True)
    #
    #     if len(edges) == 0:
    #         return None
    #
    #     return self._create_hetero_graph(nodes, edges)

    @staticmethod
    def _strip_types_if_needed(value, stripping_flag, stripped_type):
        if not stripping_flag:
            return stripped_type
        else:
            return f"{value}_"

    def _adjust_types(self, nodes, edges):
        nodes["type_backup"] = nodes["type"]
        edges["type_backup"] = edges["type"]

        nodes.eval(
            "type = type.map(@strip)",
            local_dict={
                "strip": partial(self._strip_types_if_needed, stripping_flag=self.use_node_types, stripped_type="node_")
            },
            inplace=True
        )
        edges.eval(
            "type = type.map(@strip)",
            local_dict={
                "strip": partial(self._strip_types_if_needed, stripping_flag=self.use_edge_types, stripped_type="edge_")
            },
            inplace=True
        )

    @staticmethod
    def _prepare_node_type_pool(nodes):
        # could be updated gradually to prevent from repeating this every epoch
        mapping, inv_index = compact_property(nodes["type"], return_order=True)
        nodeid2typeid = dict(zip(nodes["id"], nodes["type"].map(mapping.get)))
        return NodeTypePool(nodeid2typeid, inv_index)

    def _get_partition_ids(self, partition_label):
        partition_label = self.partition_columns_names[partition_label]  # get name for the partition mask
        return self.partition.get_partition_ids(partition_label)

    def _attach_info_to_label(self, labels, labels_for, group_by):
        if labels_for == SGLabelSpec.nodes:
            labels, new_col_name = self.dataset_db.get_info_for_node_ids(labels["src"], group_by)
            labels.rename({new_col_name: "group"}, axis=1, inplace=True)
        elif labels_for == SGLabelSpec.edges:
            raise NotImplementedError("Grouping for labels for edges is currently not supported")
        elif labels_for == SGLabelSpec.subgraphs:
            # no grouping
            pass
        else:
            raise ValueError()

        return labels

    @staticmethod
    def _get_df_hash(table):
        return str(pandas.util.hash_pandas_object(table).sum())

    def _write_to_cache(self, obj, cache_key, level=None):
        self._cache.write_to_cache(obj, cache_key, level)

    def _load_cache_if_exists(self, cache_key, level=None):
        return self._cache.load_cached(cache_key, level)

    def get_proper_partition_column_name(self, partition_label):
        return self.partition_columns_names[partition_label]

    def get_labels_for_partition(self, labels, partition_label, labels_for, group_by=SGPartitionStrategies.package):

        cache_key = f"{self._get_df_hash(labels)}_{partition_label}_{labels_for}_{group_by.name}"
        cached_result = self._load_cache_if_exists(cache_key)
        if cached_result is not None:
            return cached_result

        # allowed_labels_for = {"nodes", "edges", "subgraphs"}
        # assert labels_for in allowed_labels_for, f"{labels_for} not in {allowed_labels_for}"

        original_label_map = dict(zip(labels["src"], labels["dst"]))

        partition_ids = self._get_partition_ids(partition_label)
        labels_from_partition = labels.query("src in @partition_ids", local_dict={"partition_ids": partition_ids})

        labels_ = self._attach_info_to_label(labels_from_partition, labels_for, group_by)
        labels_.eval(
            "dst = src.map(@original_label_map.get)",
            local_dict={"original_label_map": original_label_map},
            inplace=True
        )

        self._write_to_cache(labels_, cache_key)
        return labels_

    def inference_mode(self):
        shared_node_types = PythonSharedNodes.shared_node_types
        type_filter = [ntype for ntype in self.dataset_db.get_node_type_descriptions() if ntype not in shared_node_types]
        nodes = self.dataset_db.get_nodes(type_filter=type_filter)
        nodes["train_mask"] = True
        nodes["test_mask"] = True
        nodes["val_mask"] = True

        self.train_partition = self.partition
        self.partition = PartitionIndex(nodes)
        self.inference_labels = nodes[["id", "type"]].rename({"id": "src", "type": "dst"}, axis=1)

    def iterate_subgraphs(self, how, groups, node_data, edge_data):

        iterator = self.dataset_db.iterate_subgraphs(how, groups)

        def add_data(table, data_dict):
            boolean_fields = {"train_mask", "test_mask", "val_mask", "has_label"}
            for key in data_dict:
                table.eval(
                    f"{key} = id.map(@mapper)",
                    local_dict={"mapper": lambda x: data_dict[key].get(x, pd.NA)},
                    inplace=True
                )
                if key in boolean_fields:
                    table[key].fillna(False, inplace=True)
            if "label" in data_dict:
                table["label"] = table["label"].astype("Int64")
            # if "has_label" in data_dict:
            #     table["has_label"].fillna(False, inplace=True)

        for group, nodes, edges in iterator:
            self._remove_edges_with_restricted_types(edges)

            if self.custom_reverse is not None:
                edges = self._add_custom_reverse(edges)
            self.ensure_connectedness(nodes, edges)

            node_name_mapping = self._get_embeddable_names(nodes)
            node_data["embedding_id"] = self._get_node_name2bucket_mapping(node_name_mapping)
            # node_type_pool = self._prepare_node_type_pool(nodes)

            self._adjust_types(nodes, edges)

            add_data(nodes, node_data)
            add_data(edges, edge_data)

            if len(edges) > 0:
                cache_key = self._get_df_hash(nodes) + self._get_df_hash(edges)
                subgraph = self._load_cache_if_exists(cache_key)
                if subgraph is None:
                    subgraph = self._create_hetero_graph(nodes, edges)
                    self._write_to_cache(subgraph, cache_key)

                yield group, nodes, edges, subgraph

    @staticmethod
    def holdout(nodes: pd.DataFrame, edges: pd.DataFrame, holdout_size=10000, random_seed=42):
        """
        Create a set of holdout edges, ensure that there are no orphan nodes after these edges are removed.
        :param nodes:
        :param edges:
        :param holdout_size:
        :param random_seed:
        :return:
        """

        from collections import Counter

        degree_count = Counter(edges["src"].tolist()) | Counter(edges["dst"].tolist())

        heldout = []

        edges = edges.reset_index(drop=True)
        index = edges.index.to_numpy()
        numpy.random.seed(random_seed)
        numpy.random.shuffle(index)

        for i in index:
            src_id = edges.loc[i].src
            if degree_count[src_id] > 2:
                heldout.append(edges.loc[i].id)
                degree_count[src_id] -= 1
                if len(heldout) >= holdout_size:
                    break

        heldout = set(heldout)

        def is_held(id_):
            return id_ in heldout

        train_edges = edges[
            edges["id"].apply(lambda id_: not is_held(id_))
        ]

        heldout_edges = edges[
            edges["id"].apply(is_held)
        ]

        assert len(edges) == edges["id"].unique().size

        return nodes, train_edges, heldout_edges

    # @staticmethod
    # def create_train_val_test_masks(nodes, train_idx, val_idx, test_idx):
    #     nodes['train_mask'] = True
    #     # nodes.loc[train_idx, 'train_mask'] = True
    #     nodes['val_mask'] = False
    #     nodes.loc[val_idx, 'val_mask'] = True
    #     nodes['test_mask'] = False
    #     nodes.loc[test_idx, 'test_mask'] = True
    #     nodes['train_mask'] = nodes['train_mask'] ^ (nodes['val_mask'] | nodes['test_mask'])
    #     starts_with = lambda x: x.startswith("##node_type")
    #     nodes.loc[
    #         nodes.eval("name.map(@starts_with)", local_dict={"starts_with": starts_with}), ['train_mask', 'val_mask',
    #                                                                                         'test_mask']] = False
    # @staticmethod
    # def get_train_val_test_indices(indices, train_frac=0.6, random_seed=None):
    #     if random_seed is not None:
    #         numpy.random.seed(random_seed)
    #         logging.warning("Random state for splitting dataset is fixed")
    #     else:
    #         logging.info("Random state is not set")
    #
    #     indices = indices.to_numpy()
    #
    #     numpy.random.shuffle(indices)
    #
    #     train = int(indices.size * train_frac)
    #     test = int(indices.size * (train_frac + (1 - train_frac) / 2))
    #
    #     logging.info(
    #         f"Splitting into train {train}, validation {test - train}, and test {indices.size - test} sets"
    #     )
    #
    #     return indices[:train], indices[train: test], indices[test:]

    # @staticmethod
    # def get_train_val_test_indices_on_packages(nodes, package_names, train_frac=0.6, random_seed=None):
    #     if random_seed is not None:
    #         numpy.random.seed(random_seed)
    #         logging.warning("Random state for splitting dataset is fixed")
    #     else:
    #         logging.info("Random state is not set")
    #
    #     nodes = nodes.copy()
    #
    #     package_names = [name.replace("\n", "").replace("-", "_").replace(".", "_") for name in package_names]
    #
    #     package_names = numpy.array(package_names)
    #     numpy.random.shuffle(package_names)
    #
    #     train = int(package_names.size * train_frac)
    #     test = int(package_names.size * (train_frac + (1 - train_frac) / 2))
    #
    #     logging.info(
    #         f"Splitting into train {train}, validation {test - train}, and test {package_names.size - test} packages"
    #     )
    #
    #     train, valid, test = package_names[:train], package_names[train: test], package_names[test:]
    #
    #     train = set(train.tolist())
    #     valid = set(valid.tolist())
    #     test = set(test.tolist())
    #
    #     nodes["node_package_names"] = nodes["name"].map(lambda name: name.split(".")[0])
    #
    #     def get_split_indices(split):
    #         global_types = {val for _, val in node_types.items()}
    #
    #         def is_global_type(type):
    #             return type in global_types
    #
    #         def in_split(name):
    #             return name in split
    #
    #         global_nodes = nodes.query(
    #             "node_package_names.map(@in_split) and type_backup.map(@is_global_type)",
    #             local_dict={"in_split": in_split, "is_global_type": is_global_type}
    #         )["id"]
    #         global_nodes = set(global_nodes.tolist())
    #
    #         def in_global(node_id):
    #             return node_id in global_nodes
    #
    #         ast_nodes = nodes.query("mentioned_in.map(@in_global)", local_dict={"in_global": in_global})["id"]
    #
    #         split_nodes = global_nodes | set(ast_nodes.tolist())
    #
    #         def nodes_in_split(node_id):
    #             return node_id in split_nodes
    #
    #         return nodes.query("id.map(@nodes_in_split)", local_dict={"nodes_in_split": nodes_in_split}).index
    #
    #     return get_split_indices(train), get_split_indices(valid), get_split_indices(test)

    @staticmethod
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

    def get_graph_types(self):
        ntypes = self.dataset_db.get_node_type_descriptions()
        etypes = self.dataset_db.get_edge_type_descriptions()

        return list(
            map(
                partial(self._strip_types_if_needed, stripping_flag=self.use_node_types, stripped_type="node_"),
                ntypes
            )
        ), list(
            map(
                partial(self._strip_types_if_needed, stripping_flag=self.use_edge_types, stripped_type="edge_"),
                etypes
            )
        )

    @staticmethod
    def ensure_connectedness(nodes: pandas.DataFrame, edges: pandas.DataFrame):
        """
        Filtering isolated nodes
        :param nodes: DataFrame
        :param edges: DataFrame
        :return:
        """

        # logging.info(
        #     f"Filtering isolated nodes. "
        #     f"Starting from {nodes.shape[0]} nodes and {edges.shape[0]} edges...",
        # )
        unique_nodes = set(edges['src']) | set(edges['dst'])
        num_unique_existing_nodes = nodes["id"].nunique()

        if len(unique_nodes) == num_unique_existing_nodes:
            return

        nodes.query("id not in @unique_nodes", local_dict={"unique_nodes": unique_nodes}, inplace=True)

        # logging.info(
        #     f"Ending up with {nodes.shape[0]} nodes and {edges.shape[0]} edges"
        # )

    @staticmethod
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

    # def mark_leaf_nodes(self):
    #     leaf_types = {'subword', "Op", "Constant", "Name"}  # the last is used in graphs without subwords
    #
    #     self.nodes['is_leaf'] = self.nodes['type_backup'].apply(lambda type_: type_ in leaf_types)

    # def get_typed_node_id(self, node_id, node_type):
    #     return self.typed_id_map[node_type][node_id]
    #
    # def get_global_node_id(self, node_id, node_type=None):
    #     return self.node_id_to_global_id[node_id]

    def load_node_names(self, nodes=None, min_count=None):
        """
        :return: DataFrame that contains mappings from nodes to names that appear more than once in the graph
        """

        names_for = "mention"

        if min_count is None:
            min_count = self.min_count_for_objectives

        if nodes is None:
            logging.info("Loading node names from database")
            nodes = self.dataset_db.get_nodes(type_filter=[names_for])

        if "train_mask" in nodes.columns:
            for_training = nodes.query(
                "train_mask == True or test_mask == True or val_mask == True"
            )[['id', 'type_backup', 'name']]\
                .rename({"name": "serialized_name", "type_backup": "type"}, axis=1)
        else:
            # assuming all these nodes are in train, test, or validation sets
            for_training = nodes

        for_training.query(f"type == '{names_for}'", inplace=True)

        cache_key = f"{self._get_df_hash(for_training)}_{min_count}"
        result = self._load_cache_if_exists(cache_key)

        if result is None:

            node_names = extract_node_names(for_training, 0)
            node_names = filter_dst_by_freq(node_names, freq=min_count)

            result = node_names
            self._write_to_cache(result, cache_key)

        return result

    def load_subgraph_function_names(self):
        names_path = os.path.join(self.data_path, "common_name_mappings.json.bz2")
        names = unpersist(names_path)

        function_name2graph_name = dict(zip(names["ast_name"], names["proper_names"]))

        # functions = self.nodes.query(
        #     "id in @functions", local_dict={"functions": set(self.nodes["mentioned_in"])}
        # ).query("type_backup == 'FunctionDef'")
        functions = self.dataset_db.get_nodes(type_filter=['FunctionDef'])

        cache_key = f"{self._get_df_hash(names)}_{self._get_df_hash(functions)}"
        result = self._load_cache_if_exists(cache_key)

        if result is None:
            functions["gname"] = functions["name"].apply(lambda x: function_name2graph_name.get(x, pd.NA))
            functions = functions.dropna(axis=0)
            functions["gname"] = functions["gname"].apply(lambda x: x.split(".")[-1])

            result = functions.rename({"id": "src", "gname": "dst"}, axis=1)[["src", "dst"]]

            self._write_to_cache(result, cache_key)

        return result

    def load_var_use(self):
        """
        :return: DataFrame that contains mapping from function ids to variable names that appear in those functions
        """
        path = join(self.data_path, "common_function_variable_pairs.json.bz2")
        var_use = unpersist(path)
        var_use = filter_dst_by_freq(var_use, freq=self.min_count_for_objectives)
        return var_use

    def load_api_call(self):
        path = join(self.data_path, "common_call_seq.json.bz2")
        api_call = unpersist(path)
        api_call = filter_dst_by_freq(api_call, freq=self.min_count_for_objectives)
        return api_call

    def load_token_prediction(self):
        """
        Return names for all nodes that represent local mentions
        :return: DataFrame that contains mappings from local mentions to names that these mentions represent. Applies
            only to nodes that have subwords and have appeared in a scope (names have `@` in their names)
        """

        target_nodes = self.dataset_db.get_nodes_with_subwords()

        cache_key = f"{self._get_df_hash(target_nodes)}_{self.min_count_for_objectives}"
        result = self._load_cache_if_exists(cache_key)

        if result is None:

            def name_extr(name):
                return name.split('@')[0]

            # target_nodes.eval("group = name.map(@get_group)", local_dict={"get_group": self._get_name_group}, inplace=True)
            # target_nodes.dropna(axis=0, inplace=True)
            target_nodes.eval("name = name.map(@name_extr)", local_dict={"name_extr": name_extr}, inplace=True)
            target_nodes.rename({"id": "src", "name": "dst"}, axis=1, inplace=True)
            target_nodes = filter_dst_by_freq(target_nodes, freq=self.min_count_for_objectives)
            # target_nodes.eval("cooccurr = dst.map(@occ)", local_dict={"occ": lambda name: name_cooccurr_freq.get(name, Counter())}, inplace=True)

            result = target_nodes
            self._write_to_cache(result, cache_key)

        return result

    def load_global_edges_prediction(self):

        global_edges = self.get_global_edges()
        global_edges = global_edges - {"defines", "defined_in"}  # these edges are already in AST?
        # global_edges.add("global_mention")

        cache_key = self._get_df_hash(pd.Series(sorted(list(global_edges))))
        result = self._load_cache_if_exists(cache_key)
        if result is None:
            edges = self.dataset_db.get_edges(type_filter=global_edges)

            result = edges[["src", "dst"]]
            self._write_to_cache(result, cache_key)

        return result

    def load_edge_prediction(self):

        edge_types = self.dataset_db.get_edge_type_descriptions()

        # when using this objective remove following edges
        # defined_in_*, executed_*, prev, next, *global_edges
        edge_types_for_prediction = {
            "next", "prev", "defined_in_module", "defined_in_class", "defined_in_function"
        } | {etype for etype in edge_types if etype.startswith("executed_")}

        cache_key = self._get_df_hash(pd.Series(sorted(list(edge_types_for_prediction))))
        result = self._load_cache_if_exists(cache_key)
        if result is None:
            edges = self.dataset_db.get_edges(type_filter=edge_types_for_prediction)

            # if self.use_ns_groups:
            #     groups = self.get_negative_sample_groups()
            #     valid_nodes = valid_nodes.intersection(set(groups["id"].tolist()))

            result = edges[["src", "dst", "type"]]
            self._write_to_cache(result, cache_key)

        return result

    def load_type_prediction(self):

        type_ann = unpersist(join(self.data_path, "type_annotations.json.bz2"))

        cache_key = f"{self._get_df_hash(type_ann)}_{self.min_count_for_objectives}"
        result = self._load_cache_if_exists(cache_key)

        if result is None:
            node2id = self.dataset_db.get_node_types()

            type_ann["src_type"] = type_ann["src"].apply(lambda x: node2id[x])

            type_ann = type_ann[
                type_ann["src_type"].apply(lambda type_: type_ in {"mention"})  # FunctionDef {"arg", "AnnAssign"})
            ]

            type_ann["dst"] = type_ann["dst"].apply(lambda x: x.strip("\"").strip("'").split("[")[0].split(".")[-1])
            type_ann = filter_dst_by_freq(type_ann, self.min_count_for_objectives)
            type_ann = type_ann[["src", "dst"]]
            result = type_ann
            self._write_to_cache(result, cache_key)

        return type_ann

    def load_cubert_subgraph_labels(self):
        filecontent = unpersist(join(self.data_path, "common_filecontent.json.bz2"))
        return filecontent[["id", "label"]].rename({"id": "src", "label": "dst"}, axis=1)

    def load_docstring(self):
        docstrings_path = os.path.join(self.data_path, "common_bodies.json.bz2")

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

    def load_node_classes(self):
        # TODO
        labels = self.dataset_db.get_nodes_for_classification()
        labels.query("src.map(@in_partition)", local_dict={"in_partition": lambda x: x in self.partition}, inplace=True)

        return labels

    # def buckets_from_pretrained_embeddings(self, pretrained_path, n_buckets):
    #
    #     from SourceCodeTools.nlp.embed.fasttext import load_w2v_map
    #     from SourceCodeTools.nlp import token_hasher
    #     pretrained = load_w2v_map(pretrained_path)
    #
    #     import numpy as np
    #
    #     embs_init = np.random.randn(n_buckets, pretrained.n_dims).astype(np.float32)
    #
    #     for word in pretrained.keys():
    #         ind = token_hasher(word, n_buckets)
    #         embs_init[ind, :] = pretrained[word]
    #
    #     def op_embedding(op_tokens):
    #         embedding = None
    #         for token in op_tokens:
    #             token_emb = pretrained.get(token, None)
    #             if embedding is None:
    #                 embedding = token_emb
    #             else:
    #                 embedding = embedding + token_emb
    #         return embedding
    #
    #     python_ops_to_bpe = self._op_tokens()
    #     for op, op_tokens in python_ops_to_bpe.items():
    #         op_emb = op_embedding(op_tokens)
    #         if op_emb is not None:
    #             op_ind = token_hasher(op, n_buckets)
    #             embs_init[op_ind, :] = op_emb
    #
    #     return embs_init

    def create_subword_masker(self, nodes, edges):
        """
        :return: SubwordMasker for all nodes that have subwords. Suitable for token prediction objective.
        """
        # TODO
        return SubwordMasker(nodes, edges)

    def create_variable_name_masker(self, nodes, edges):
        """
        :return: SubwordMasker for function nodes. Suitable for variable name use prediction objective
        """
        return NodeNameMasker(nodes, edges, self.load_var_use(), self.tokenizer_path)

    def create_node_name_masker(self, nodes, edges):
        """
        :return: SubwordMasker for function nodes. Suitable for node name use prediction objective
        """
        return NodeNameMasker(nodes, edges, self.load_node_names(nodes, min_count=0), self.tokenizer_path)

    def create_node_clf_masker(self, nodes, edges):
        """
        :return: SubwordMasker for function nodes. Suitable for node name use prediction objective
        """
        return NodeClfMasker(nodes, edges)


def read_or_create_gnn_dataset(args, model_base, force_new=False, restore_state=False):
    if restore_state and not force_new:
        raise NotImplementedError()
        # dataset = SourceGraphDataset.load(join(model_base, "dataset.pkl"), args)
    else:
        dataset = SourceGraphDataset(**args)

        # save dataset state for recovery
        from SourceCodeTools.models.training_config import save_config
        save_config(args, join(model_base, "dataset.config"))

    return dataset


def test_dataset():
    import sys

    data_path = sys.argv[1]
    partition = sys.argv[2]
    # nodes_path = sys.argv[1]
    # edges_path = sys.argv[2]

    dataset = SourceGraphDataset(
        data_path, partition,
        use_node_types=False,
        use_edge_types=True,
        no_global_edges=True,
        remove_reverse=False,
        custom_reverse=["global_mention"],
        tokenizer_path="/Users/LTV/Downloads/NitroShare/v2_subsample_no_spacy_v3/with_ast/sentencepiece_bpe.model"
    )
    # node_names = unpersist("/Users/LTV/Downloads/NitroShare/v2_subsample_v4_new_ast2_fixed_distinct_types/with_ast/node_names.json.bz2")
    node_names = dataset.load_node_names(min_count=5)
    # token_names = dataset.load_token_prediction()
    # global_edges = dataset.load_global_edges_prediction()
    # type_labels = dataset.load_type_prediction()
    # node_classes = dataset.load_node_classes()
    # mapping = compact_property(node_names['dst'])
    # node_names['dst'] = node_names['dst'].apply(mapping.get)
    # from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
    # from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedder
    # from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedderWithBpeSubwords
    from SourceCodeTools.code.data.dataset.DataLoader import SGNodesDataLoader
    dataloader = SGNodesDataLoader(dataset, labels_for="nodes", number_of_hops=3, batch_size=512, labels=node_names,
        masker_fn=dataset.create_node_name_masker,
        label_loader_class=TargetLoader,
        label_loader_params={
            "emb_size": 100,
            "tokenizer_path": "/Users/LTV/Downloads/NitroShare/v2_subsample_no_spacy_v3/with_ast/sentencepiece_bpe.model"
        })
    dataloader.nodes_dataloader(
        partition_label="train",
    )

    # sm = dataset.create_subword_masker()
    print(dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    test_dataset()
