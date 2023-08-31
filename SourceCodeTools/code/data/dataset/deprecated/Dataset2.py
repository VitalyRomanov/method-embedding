import os.path
import pickle
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, OrderedDict
from functools import partial, lru_cache
from os.path import join
from typing import List, Optional

import dgl
import networkx as nx
import numpy
import pandas
import torch

from SourceCodeTools.cli_arguments.config import save_config, load_config
from SourceCodeTools.code.ast.python_ast2 import PythonSharedNodes, PythonNodeEdgeDefinitions
from SourceCodeTools.code.data.GraphStorage import OnDiskGraphStorageWithFastIteration, InMemoryGraphStorage, \
    OnDiskGraphStorage, OnDiskGraphStorageWithFastIterationNoPandas, AbstractGraphStorage
from SourceCodeTools.code.data.DBStorage import SQLiteStorage
from SourceCodeTools.code.data.dataset.SubwordMasker import SubwordMasker, NodeNameMasker, NodeClfMasker, \
    SubwordMaskerNoPandas
from SourceCodeTools.code.data.dataset.partition_strategies import SGPartitionStrategies, SGLabelSpec
from SourceCodeTools.code.data.file_utils import *
from SourceCodeTools.code.data.multiprocessing_storage_adapter import GraphStorageWorkerAdapter, \
    GraphStorageWorkerMetaAdapter
from SourceCodeTools.code.data.sourcetrail.sourcetrail_extract_node_names import extract_node_names
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


class AbstractDataset(ABC):
    use_node_types = False
    use_edge_types = False
    _graph_storage = None

    @abstractmethod
    def get_num_nodes(self):
        ...

    @abstractmethod
    def get_num_edges(self):
        ...

    @abstractmethod
    def get_partition_size(self, partition):
        ...

    @abstractmethod
    def get_partition_slice(self, partition):
        ...

    @abstractmethod
    def iterate_subgraphs(
            self, *args, **kwargs
    ):
        ...

    def _create_hetero_graph(self, nodes, edges, **kwargs):
        ...


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

        self._any_set = self._train_set | self._test_set | self._val_set

    def get_partition_size(self, partition):
        return len(getattr(self, f"_{partition}_set"))

    def create_exclusive(self, exclusive_label):
        new_index = self.__class__()
        new_index._index = self._index
        if exclusive_label == "any_mask":
            new_index._exclusive_label = -1
        else:
            new_index._exclusive_label = self._partition_label_order.index(exclusive_label)
        return new_index

    def is_partition(self, node_id, partition_label):
        return node_id in self.get_partition_ids(partition_label)

    def get_partition_ids(self, partition_label):
        if partition_label == "any_mask":
            partition_label_source = {"any_mask": self._any_set }  # self._train_set | self._test_set | self._val_set}
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
            if self._exclusive_label == -1:
                if item not in self._index:
                    return False
                return any(self._index[item][label] for label in self._index[item])
            if item not in self._index:
                return False
            return self._index[item][self._exclusive_label]

    def get(self, item, default):
        if item not in self._index:
            return default
        return self[item]

    def __len__(self):
        return len(self._index)

    def values(self):
        for item in self._index:
            yield self._index[item][self._exclusive_label]


class SimpleGraphCreator:
    def __init__(self):
        self.use_node_types = False
        self.use_edge_types = False

    @staticmethod
    def _add_node_types_to_edges(nodes, edges):

        node_type_map = dict(zip(nodes['id'].values, nodes['type']))

        edges.eval("src_type = src.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges.eval("dst_type = dst.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges = edges.astype({'src_type': 'category', 'dst_type': 'category'}, copy=False)

        return edges

    @staticmethod
    def _strip_types_if_needed(value, stripping_flag, stripped_type):
        if not stripping_flag:
            return stripped_type
        else:
            return f"{value}_"

    def get_graph_types(self):
        ntypes = InMemoryGraphStorage.get_node_type_descriptions()
        etypes = InMemoryGraphStorage.get_edge_type_descriptions()

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

        def unpack_node_data(nodes):
            node_data = {}
            for col_name, dtype in zip(nodes.columns, nodes.dtypes):
                if col_name in {"id", "type", "name", "type_backup"}:
                    continue
                mapping = dict(zip(nodes["id"], nodes[col_name]))
                node_data[col_name] = torch.tensor(
                    [mapping.get(node_id, -1) for node_id in range(num_nodes)],
                    dtype=get_torch_dtype(get_canonical_type(dtype))
                )
            return node_data

        def assign_dense_ids(nodes, edges):
            nodes["original_id"] = nodes["id"].copy()
            compact_node_ids = compact_property(nodes["id"])
            nodes["id"] = nodes["id"].apply(compact_node_ids.get)
            edges["src"] = edges["src"].apply(compact_node_ids.get)
            edges["dst"] = edges["dst"].apply(compact_node_ids.get)

        def normalize_types(nodes, edges):
            nodes["type_backup"] = nodes["type"].copy()
            nodes["type"] = nodes["type"].apply(partial(self._strip_types_if_needed, stripping_flag=self.use_node_types, stripped_type="node_"))
            edges["type_backup"] = edges["type"].copy()
            edges["type"] = edges["type"].apply(partial(self._strip_types_if_needed, stripping_flag=self.use_edge_types, stripped_type="edge_"))

        normalize_types(nodes, edges)
        nodes.drop("mentioned_in", axis=1, inplace=True)
        nodes.drop("string", axis=1, inplace=True)

        assign_dense_ids(nodes, edges)

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

            typed_subgraphs[subgraph_signature] = list(
                zip(
                    subset['src'],
                    subset['dst']
                )
            )

        # # need to make metagraphs the same for all graphs. lines below assume node types is set to false always
        # assert self.use_node_types is False
        # _, all_edge_types = self.get_graph_types()  # need to have this predefined
        # for etype in set(all_edge_types) - set(possible_edge_signatures["type"]):
        #     typed_subgraphs[("node_", etype, "node_")] = list()

        num_nodes = len(nodes)  # self._num_nodes  # nodes["id"].max() + 1  # the fact that this is needed probably means that graphs are not loading correctly
        graph = dgl.heterograph(typed_subgraphs, num_nodes_dict={ntype: num_nodes for ntype in nodes["type"].unique()})

        def attach_node_data(graph, nodes):

            node_data = unpack_node_data(nodes)

            for ntype in graph.ntypes:
                current_type_subset = set(nodes_table.query(f"select id from nodes where type = '{ntype}'")["id"])
                node_ids_for_ntype = graph.nodes(ntype)  # all ntypes contain all nodes
                current_type_mask = torch.tensor(
                    # check memory consumption here
                    [node_id in current_type_subset for node_id in node_ids_for_ntype.tolist()],
                    dtype=get_torch_dtype("bool")
                )
                graph.nodes[ntype].data["current_type_mask"] = current_type_mask
                for col_name in node_data:
                    graph.nodes[ntype].data[col_name] = node_data[col_name]

        attach_node_data(graph, nodes)
        return graph


class SourceGraphDataset(AbstractDataset):
    use_node_types = None
    use_edge_types = None
    filter_edges = None

    type_pred_filename = "type_annotations.json.bz2"

    partition_columns_names = {
        "train": "train_mask",
        "val": "val_mask",
        "test": "test_mask",
        "any": "any_mask"
    }

    @classmethod
    def create_dataset_specification(
            cls, data_path: Union[str, Path], partition, use_node_types: bool = False, use_edge_types: bool = False,
            filter_edges: Optional[List[str]] = None, tokenizer_path: Union[str, Path] = None,
            min_count_for_objectives: int = 1, no_global_edges: bool = False, remove_reverse: bool = False,
            custom_reverse: Optional[List[str]] = None, use_ns_groups: bool = False, type_nodes=False,
            max_type_ann_level=3, k_hops=0, storage_class=None, storage_kwargs=None, n_buckets=100000, **kwargs
    ):
        spec = {
            "dataset_class": cls,
            "use_node_types": use_node_types,
            "use_edge_types": use_edge_types,
            "data_path": data_path,
            "partition": partition,
            "storage_class": storage_class,
            "storage_kwargs": storage_kwargs,
            "tokenizer_path": tokenizer_path,
            "min_count_for_objectives": min_count_for_objectives,
            "no_global_edges": no_global_edges,
            "remove_reverse": remove_reverse,
            "custom_reverse": custom_reverse,
            "use_ns_groups": use_ns_groups,
            "type_nodes": type_nodes,
            "max_type_ann_level": max_type_ann_level,
            "k_hops": k_hops,
            "edge_types_to_remove": set(),
            "n_buckets": n_buckets,
        }

        # cls._initialize_storage(spec, None, **kwargs)

        if filter_edges is not None:
            cls._filter_edges(spec, filter_edges)

        if remove_reverse:
            cls._remove_reverse_edges(spec)

        if no_global_edges:
            cls._remove_global_edges(spec)

        return spec

    @classmethod
    # @lru_cache
    def get_partition_index(cls, spec):
        partition = spec["partition"]
        return PartitionIndex(unpersist(partition)) if partition is not None else None

    @classmethod
    def _remove_edges_with_restricted_types(cls, spec, edges):
        edges.query(
            "type not in @restricted_types", local_dict={"restricted_types": spec["edge_types_to_remove"]}, inplace=True
        )

    @classmethod
    def _initialize_storage(cls, spec, **kwargs):
        _graph_storage_path = OnDiskGraphStorageWithFastIteration.get_storage_file_name(spec["data_path"])
        if not OnDiskGraphStorageWithFastIteration.verify_imported(_graph_storage_path):
            _graph_storage = OnDiskGraphStorageWithFastIteration(_graph_storage_path)
            _graph_storage.import_from_files(spec["data_path"])
            _graph_storage.add_import_completed_flag(_graph_storage_path)

    @classmethod
    def _filter_edges(cls, spec, types_to_filter):
        spec["edge_types_to_remove"].update(types_to_filter)

    @classmethod
    def _get_embeddable_names(cls, nodes):
        id2embeddable_name = dict()
        name_pool = dict()
        name_pool_rev = dict()
        for node_id, embeddable_name in zip(
                nodes["id"],
                map(cls._get_embeddable_name, nodes["name"])  # nodes["name"].apply(self._get_embeddable_name)
        ):
            if embeddable_name not in name_pool:
                name_pool[embeddable_name] = len(name_pool)
                name_pool_rev[name_pool[embeddable_name]] = embeddable_name
            id2embeddable_name[node_id] = name_pool[embeddable_name]
        return NodeNamePool(id2embeddable_name, name_pool_rev)

    @staticmethod
    def _add_node_types_to_edges(nodes, edges):

        node_type_map = dict(zip(nodes['id'].values, nodes['type']))

        edges.eval("src_type = src.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges.eval("dst_type = dst.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges = edges.astype({'src_type': 'category', 'dst_type': 'category'}, copy=False)

        return edges

    @classmethod
    def _remove_global_edges(cls, spec):
        global_edges = cls.get_global_edges()
        spec["edge_types_to_remove"].update(global_edges)

    @classmethod
    def get_storage_instance(cls, spec) -> AbstractGraphStorage:
        return spec["storage_class"](**spec["storage_kwargs"])

    @classmethod
    def _remove_reverse_edges(cls, spec):
        from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping
        global_reverse = {key for key, val in special_mapping.items()}
        spec["edge_types_to_remove"].update(global_reverse)
        spec["edge_types_to_remove"].update(PythonNodeEdgeDefinitions.reverse_edge_exceptions.values())

        storage = cls.get_storage_instance(spec)

        all_edge_types = storage.get_edge_type_descriptions()
        spec["edge_types_to_remove"].update(filter(lambda edge_type: edge_type.endswith("_rev"), all_edge_types))

    @classmethod
    def _add_custom_reverse(cls, spec, edges):
        to_reverse = edges.query("type in @custom_reverse", local_dict={"custom_reverse": spec["custom_reverse"]})

        to_reverse.eval("type = type.map(@add_rev)", local_dict={"add_rev": lambda type_: type_ + "_rev"}, inplace=True)
        to_reverse.rename({"src": "dst", "dst": "src"}, axis=1, inplace=True)

        new_id = edges["id"].max() + 1
        to_reverse["id"] = range(new_id, new_id + len(to_reverse))

        return edges.append(to_reverse[["src", "dst", "type"]])

    @classmethod
    def _create_hetero_graph(cls, spec, nodes, edges, **kwargs):

        def set_canonical_edge_types(edges, edge_subset):
            return edge_subset.astype(edges.dtypes)
            # for col_name, dtype in zip(edges.dtypes.index, edges.dtypes):
            #     if col_name in edge_subset.columns:
            #         edge_subset = edge_subset.astype({col_name: dtype})
            # return edge_subset

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

        def unpack_table(table, columns_to_skip, reference):
            data = {}
            for col_name, dtype in zip(table.columns, table.dtypes):
                if col_name in columns_to_skip:
                    continue
                # if col_name == "label":
                #     labels = table[["id", "label"]].dropna()
                #     mapping = dict(zip(labels["id"], labels[col_name]))
                #     has_label_mask = torch.tensor(
                #         [node_id in mapping for node_id in range(table["id"].max() + 1)],
                #         dtype=torch.bool
                #     )
                #     node_data["has_label"] = has_label_mask
                # else:
                mapping = dict(zip(table["id"], table[col_name]))
                data[col_name] = torch.tensor(
                    [mapping.get(node_id, -1) for node_id in reference],
                    dtype=get_torch_dtype(get_canonical_type(dtype))
                )
            return data

        def unpack_node_data(nodes):
            return unpack_table(nodes, {"id", "type", "name", "type_backup", "mentioned_in", "string"}, reference=range(num_nodes))

        def unpack_edge_data(edges):
            return unpack_table(edges, {
                "id", "type", "type_backup", "unique_file_id", "file_id", "package",
                "src_type", "dst_type", "mentioned_in", "offset_start", "offset_end",
                'src_type', 'dst_type', 'src_name', 'dst_name'
            }, reference=edges["id"])

        def assign_dense_node_ids(nodes, edges):
            nodes["original_id"] = nodes["id"].copy()
            compact_node_ids = compact_property(nodes["id"])
            nodes["id"] = nodes["id"].apply(compact_node_ids.get)
            edges["src"] = edges["src"].apply(compact_node_ids.get)
            edges["dst"] = edges["dst"].apply(compact_node_ids.get)

        def assign_dense_edge_ids(edges):
            edges["original_id"] = edges["id"].copy()
            edges["typed_id"] = edges["id"].copy()
            # compact_edge_ids = compact_property(edges["id"])
            # edges["id"] = edges["id"].apply(compact_edge_ids.get)

        assign_dense_node_ids(nodes, edges)
        assign_dense_edge_ids(edges)

        edges = cls._add_node_types_to_edges(nodes, edges)

        possible_edge_signatures = edges[['src_type', 'type', 'dst_type']].drop_duplicates(
            ['src_type', 'type', 'dst_type']
        )

        typed_subgraphs = {}
        typed_edge_ids = {}

        nodes_table = SQLiteStorage(":memory:")
        nodes_table.add_records(nodes, "nodes", create_index=["id", "type"])
        edges_table = SQLiteStorage(":memory:")
        edges_table.add_records(edges, "edges", create_index=["src_type", "type", "dst_type"])

        for src_type, type, dst_type in possible_edge_signatures[["src_type", "type", "dst_type"]].values:  #
            subgraph_signature = (src_type, type, dst_type)

            subset = edges_table.query(
                f"select * from edges where src_type = '{src_type}' and type = '{type}' and dst_type = '{dst_type}'"
            )

            if len(subset) > 0 and len(subset["id"].value_counts()) == 0:
                raise Exception(
                    "Found edges without ids. This is likely happening because new edges were added without properly "
                    "setting edge ids. The same happens if one removes reverse edges and then adds some of the reverse "
                    "edges back."
                )

            typed_edge_ids[subgraph_signature] = dict(zip(subset["id"], range(len(subset))))

            typed_subgraphs[subgraph_signature] = list(
                zip(
                    subset['src'],
                    subset['dst']
                )
            )

        def assign_typed_edge_ids(edges, typed_edge_ids):
            type_ids = []
            for edge_id, src_type, type_, dst_type in edges[["id", "src_type", "type", "dst_type"]].values:
                subgraph_signature = (src_type, type_, dst_type)
                value = typed_edge_ids[subgraph_signature][edge_id]
                type_ids.append(value)

            assert len(type_ids) == len(edges)
            edges["typed_id"] = type_ids

        assign_typed_edge_ids(edges, typed_edge_ids)
        edges_table = SQLiteStorage(":memory:")
        edges_table.add_records(edges, "edges", create_index=["src_type", "type", "dst_type"])

        # need to make metagraphs the same for all graphs. lines below assume node types is set to false always
        assert spec["use_node_types"] is False
        _, all_edge_types = cls.get_graph_types(spec)
        for etype in set(all_edge_types) - set(possible_edge_signatures["type"]):
            typed_subgraphs[("node_", etype, "node_")] = list()

        # logging.info(
        #     f"Unique triplet types in the graph: {len(typed_subgraphs.keys())}"
        # )

        num_nodes = len(nodes)  # self._num_nodes  # nodes["id"].max() + 1  # the fact that this is needed probably means that graphs are not loading correctly
        graph = dgl.heterograph(typed_subgraphs, num_nodes_dict={ntype: num_nodes for ntype in nodes["type"].unique()})

        def attach_node_data(graph, nodes):

            node_data = unpack_node_data(nodes)

            # def add_node_data_to_graph(graph, col_name, data, canonical_type):
            #     data = torch.tensor(data, dtype=get_torch_dtype(canonical_type))
            #     graph.nodes[ntype].data[col_name] = data

            for ntype in graph.ntypes:
                current_type_subset = set(nodes_table.query(f"select id from nodes where type = '{ntype}'")["id"])
                node_ids_for_ntype = graph.nodes(ntype)  # all ntypes contain all nodes
                current_type_mask = torch.tensor(
                    # check memory consumption here
                    # TODO high memory bandwidth consumption!
                    [node_id in current_type_subset for node_id in node_ids_for_ntype.tolist()],
                    dtype=get_torch_dtype("bool")
                )
                graph.nodes[ntype].data["current_type_mask"] = current_type_mask
                for col_name in node_data:
                    graph.nodes[ntype].data[col_name] = node_data[col_name]

        def attach_edge_data(graph, edges):

            # edge_data = unpack_edge_data(edges)

            # def add_node_data_to_graph(graph, col_name, data, canonical_type):
            #     data = torch.tensor(data, dtype=get_torch_dtype(canonical_type))
            #     graph.nodes[ntype].data[col_name] = data

            for src_type, etype, dst_type in graph.canonical_etypes:
                subgraph_signature = (src_type, etype, dst_type)
                current_type_subset = set_canonical_edge_types(edges, edges_table.query(
                    f"select * from edges "
                    f"where type = '{etype}' and src_type = '{src_type}' and dst_type = '{dst_type}'"
                ))
                if len(current_type_subset) == 0:
                    continue
                edge_data = unpack_edge_data(current_type_subset)
                src_for_etype, dst_for_etype, edge_ids_for_etype = graph.edges(etype=etype, form="all")  # all ntypes contain all nodes
                apparent_src, apparent_dst, apparent_edge_ids = edge_data.pop("src"), edge_data.pop("dst"), edge_data.pop("typed_id")
                assert all(src_for_etype == apparent_src) and all(dst_for_etype == apparent_dst) and all(edge_ids_for_etype == apparent_edge_ids), "Edge data assignment failed"
                for col_name in edge_data:
                    graph.edges[subgraph_signature].data[col_name] = edge_data[col_name]

        attach_node_data(graph, nodes)
        attach_edge_data(graph, edges)
        return graph

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
            return "mention"
            # return name.split("@")[0]
        elif "_0x" in name:
            return name.split("_0x")[0]
        else:
            return name

    @classmethod
    def _get_node_name2bucket_mapping(cls, node_id2name, n_buckets):
        return {key: token_hasher(val, n_buckets) for key, val in node_id2name.items()}

    @staticmethod
    def _strip_types_if_needed(value, stripping_flag, stripped_type):
        if not stripping_flag:
            return stripped_type
        else:
            return f"{value}_"

    @classmethod
    def _adjust_types(cls, spec, nodes, edges):
        nodes["type_backup"] = nodes["type"]
        edges["type_backup"] = edges["type"]

        nodes.eval(
            "type = type.map(@strip)",
            local_dict={
                "strip": partial(cls._strip_types_if_needed, stripping_flag=spec["use_node_types"], stripped_type="node_")
            },
            inplace=True
        )
        edges.eval(
            "type = type.map(@strip)",
            local_dict={
                "strip": partial(cls._strip_types_if_needed, stripping_flag=spec["use_edge_types"], stripped_type="edge_")
            },
            inplace=True
        )

    @staticmethod
    def _prepare_node_type_pool(nodes):
        # could be updated gradually to prevent from repeating this every epoch
        mapping, inv_index = compact_property(nodes["type"], return_order=True)
        nodeid2typeid = dict(zip(nodes["id"], nodes["type"].map(mapping.get)))
        return NodeTypePool(nodeid2typeid, inv_index)

    @classmethod
    def _get_partition_ids(cls, spec, partition_label):
        partition_label = cls.partition_columns_names[partition_label]  # get name for the partition mask
        partition = cls.get_partition_index(spec)
        return partition.get_partition_ids(partition_label)

    @classmethod
    def _attach_info_to_label(cls, spec, labels, labels_for, group_by):
        storage = cls.get_storage_instance(spec)
        if labels_for == SGLabelSpec.nodes:
            labels, new_col_name = storage.get_info_for_node_ids(labels["src"], group_by)
            labels.rename({new_col_name: "group", "id": "src"}, axis=1, inplace=True)
        elif labels_for == SGLabelSpec.edges:
            labels, new_col_name = storage.get_info_for_edge_ids(labels["src"], group_by)
            labels.rename({new_col_name: "group", "id": "src"}, axis=1, inplace=True)
        elif labels_for == SGLabelSpec.subgraphs:
            labels, new_col_name = storage.get_info_for_subgraphs(labels["src"], group_by)
            labels.rename({new_col_name: "group"}, axis=1, inplace=True)
        else:
            raise ValueError()
        return labels

    @staticmethod
    def _add_data_to_table(table, data_dict):
        boolean_fields = {"train_mask", "test_mask", "val_mask", "has_label", "any_mask"}
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

    @staticmethod
    def _add_type_nodes(nodes, edges):
        node_new_id = nodes["id"].max() + 1
        edge_new_id = edges["id"].max() + 1

        new_nodes = []
        new_edges = []
        added_type_nodes = {}

        node_slice = nodes[["id", "type"]].values

        for id, type in node_slice:
            if type not in added_type_nodes:
                added_type_nodes[type] = node_new_id
                node_new_id += 1

                new_nodes.append({
                    "id": added_type_nodes[type],
                    "name": type,
                    "type": "type_node",
                })

            new_edges.append({
                "id": edge_new_id,
                "type": "node_type",
                "src": added_type_nodes[type],
                "dst": id,
            })
            edge_new_id += 1

        new_nodes, new_edges = pd.DataFrame(new_nodes), pd.DataFrame(new_edges)

        return nodes.append(new_nodes), edges.append(new_edges)

    @classmethod
    # @lru_cache
    def _k_hop_skip_edges(cls, spec):
        from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping

        storage = cls.get_storage_instance(spec)
        skip_edges = (
                set(special_mapping.keys()) |
                set(special_mapping.values()) |
                set(PythonNodeEdgeDefinitions.reverse_edge_exceptions.values()) |
                set(filter(lambda edge_type: edge_type.endswith("_rev"),
                           storage.get_edge_type_descriptions()))
        )
        return skip_edges

    @classmethod
    def _add_k_hop_edges(cls, spec, nodes, edges):
        g = nx.from_pandas_edgelist(
            edges, source="src", target="dst", create_using=nx.DiGraph, edge_attr="type"
        )

        skip_edges = cls._k_hop_skip_edges(spec)

        def expand_edges(edges, node_id, view, edge_prefix, level=0):
            # edges = []
            if level < spec["k_hops"]:
                if edge_prefix != "":
                    edge_prefix += "|"
                for e in view:
                    next_edge_type = view[e]["type"]
                    if level > 0:
                        new_prefix = f"{level}_hop_connection"
                        # edges.append((node_id, e, new_prefix.rstrip("|")))
                        edges.append({"src": node_id, "dst": e, "type": new_prefix.rstrip("|")})
                    else:
                        new_prefix = edge_prefix + next_edge_type
                    # edges.append((node_id, e, new_prefix.rstrip("|")))
                    if next_edge_type in skip_edges:
                        continue
                    expand_edges(edges, node_id, g[e], new_prefix, level=level + 1)
                    # edges.extend(expand_edges(node_id, g[e], new_prefix, level=level+1))
            return edges

        new_edges = []
        for node in g.nodes:
            expand_edges(new_edges, node, g[node], "", level=0)

        new_edges_df = pd.DataFrame.from_records(new_edges)
        new_edges_id = edges["id"].max() + 1
        new_edges_df["id"] = range(new_edges_id, new_edges_id + len(new_edges_df))
        edges_with_k_hop = edges.append(new_edges_df).drop_duplicates(["src", "dst"])

        return nodes, edges_with_k_hop

    @classmethod
    def _create_graph_from_nodes_and_edges(
            cls, spec, nodes, edges, node_data=None, edge_data=None, n_buckets=200000
    ):
        if node_data is None:
            node_data = {}
        if edge_data is None:
            edge_data = {}

        cls._remove_edges_with_restricted_types(spec, edges)

        if spec["custom_reverse"] is not None:
            edges = cls._add_custom_reverse(spec, edges)
        cls.ensure_connectedness(nodes, edges)

        if spec["k_hops"] > 0:
            nodes, edges = cls._add_k_hop_edges(spec, nodes, edges)

        # if self.type_nodes:
        #     nodes, edges = self._add_type_nodes(nodes, edges)

        node_name_mapping = cls._get_embeddable_names(nodes)
        node_data["embedding_id"] = cls._get_node_name2bucket_mapping(node_name_mapping, n_buckets)
        # node_type_pool = self._prepare_node_type_pool(nodes)

        cls._adjust_types(spec, nodes, edges)

        cls._add_data_to_table(nodes, node_data)
        cls._add_data_to_table(edges, edge_data)

        if len(edges) > 0:
            subgraph = cls._create_hetero_graph(spec, nodes, edges)
            return subgraph
        else:
            return None

    @classmethod
    def get_num_nodes(cls, spec):
        storage = cls.get_storage_instance(spec)
        return storage.get_num_nodes()

    @classmethod
    def get_num_edges(cls, spec):
        storage = cls.get_storage_instance(spec)
        return storage.get_num_edges()

    @classmethod
    def get_partition_size(cls, spec, partition):
        _partition = cls.get_partition_index(spec)
        return _partition.get_partition_size(partition)

    @classmethod
    def get_partition_slice(cls, spec, partition):
        _partition = cls.get_partition_index(spec)
        return _partition.create_exclusive(f"{partition}_mask")

    @classmethod
    def get_proper_partition_column_name(cls, partition_label):
        return cls.partition_columns_names[partition_label]

    @staticmethod
    def _get_df_hash(table):
        return str(pandas.util.hash_pandas_object(table).sum())

    @staticmethod
    def _load_cache_if_exists(spec, cache_key, level=None):
        cache = DatasetCache(spec["data_path"])
        return cache.load_cached(cache_key, level)

    @staticmethod
    def _write_to_cache(spec, obj, cache_key, level=None):
        cache = DatasetCache(spec["data_path"])
        cache.write_to_cache(obj, cache_key, level)

    @classmethod
    def get_labels_for_partition(cls, spec, labels, partition_label, labels_for, group_by=SGPartitionStrategies.package):

        logging.info(f"Getting labels for {partition_label} partition")

        cache_key = f"{cls._get_df_hash(labels)}_{partition_label}_{labels_for}_{group_by.name}"
        cached_result = cls._load_cache_if_exists(spec, cache_key)
        if cached_result is not None:
            return cached_result

        partition_ids = cls._get_partition_ids(spec, partition_label)
        labels_from_partition = labels.query("src in @partition_ids", local_dict={"partition_ids": partition_ids})

        if "group" in labels_from_partition.columns:
            labels_ = labels_from_partition
        else:
            labels_ = cls._attach_info_to_label(spec, labels_from_partition, labels_for, group_by)
            labels_ = labels_.merge(labels, how="left", on="src")

        cls._write_to_cache(spec, labels_, cache_key)
        return labels_

    def get_cache_key(self, how, group):
        return f"{how.name}_{group}"

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

    @classmethod
    # @lru_cache
    def get_graph_types(cls, spec):
        storage = cls.get_storage_instance(spec)
        ntypes = storage.get_node_type_descriptions()
        etypes = storage.get_edge_type_descriptions()

        def only_unique(elements):
            new_list = []
            for ind, element in enumerate(elements):
                if ind != 0 and element == new_list[-1]:
                    continue
                new_list.append(element)
            return new_list

        return only_unique(
            map(
                partial(cls._strip_types_if_needed, stripping_flag=spec["use_node_types"], stripped_type="node_"),
                ntypes
            )
        ), only_unique(
            map(
                partial(cls._strip_types_if_needed, stripping_flag=spec["use_edge_types"], stripped_type="edge_"),
                etypes
            )
        )

    @classmethod
    def inference_mode(self):
        raise NotImplementedError
        # shared_node_types = PythonSharedNodes.shared_node_types
        # type_filter = [ntype for ntype in self._graph_storage.get_node_type_descriptions() if ntype not in shared_node_types]
        # nodes = self._graph_storage.get_nodes(type_filter=type_filter)
        # nodes["train_mask"] = True
        # nodes["test_mask"] = True
        # nodes["val_mask"] = True
        #
        # self.train_partition = self._partition
        # self._partition = PartitionIndex(nodes)
        # self.inference_labels = nodes[["id", "type"]].rename({"id": "src", "type": "dst"}, axis=1)

    @classmethod
    def iterate_subgraphs(cls, spec, how, groups, node_data, edge_data, subgraph_data, n_buckets):
        storage = cls.get_storage_instance(spec)
        iterator = storage.iterate_subgraphs(how, groups)

        for group, nodes, edges in iterator:

            edges_bloom_filter = set()  # BloomFilter(max_elements=len(edges), error_rate=0.01)
            for src, dst in zip(edges["src"], edges["dst"]):
                edges_bloom_filter.add((src, dst))

            # logging.info("Creating DGL graph")
            subgraph = cls._create_graph_from_nodes_and_edges(spec, nodes, edges, node_data, edge_data, n_buckets=n_buckets)
            # logging.info("Finished creating DGL graph")

            if subgraph is not None:
                yield {
                    "group": group,
                    "nodes": nodes,
                    "edges": edges,
                    "subgraph": subgraph,
                    "edges_bloom_filter": edges_bloom_filter,
                    "subgraph_data": subgraph_data
                }

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

        nodes.query("id in @unique_nodes", local_dict={"unique_nodes": unique_nodes}, inplace=True)

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

    @classmethod
    def load_node_names(cls, spec, nodes=None, min_count=None):
        """
        :return: DataFrame that contains mappings from nodes to names that appear more than once in the graph
        """

        names_for = "mention"

        if min_count is None:
            min_count = spec["min_count_for_objectives"]

        if nodes is None:
            logging.info("Loading node names from database")
            storage = cls.get_storage_instance(spec)
            nodes = storage.get_nodes(type_filter=[names_for])

        if "train_mask" in nodes.columns:
            for_training = nodes.query(
                "train_mask == True or test_mask == True or val_mask == True"
            )[['id', 'type_backup', 'name']]\
                .rename({"name": "serialized_name", "type_backup": "type"}, axis=1)
        else:
            # assuming all these nodes are in train, test, or validation sets
            for_training = nodes

        for_training.query(f"type == '{names_for}'", inplace=True)

        cache_key = f"{cls._get_df_hash(for_training)}_{min_count}"
        result = cls._load_cache_if_exists(spec, cache_key)

        if result is None:

            node_names = extract_node_names(for_training, 0)
            node_names = filter_dst_by_freq(node_names, freq=min_count)

            result = node_names
            cls._write_to_cache(spec, result, cache_key)

        return result

    def load_subgraph_function_names(self):
        names_path = os.path.join(self.data_path, "common_name_mappings.json.bz2")
        names = unpersist(names_path)

        function_name2graph_name = dict(zip(names["ast_name"], names["proper_names"]))

        # functions = self.nodes.query(
        #     "id in @functions", local_dict={"functions": set(self.nodes["mentioned_in"])}
        # ).query("type_backup == 'FunctionDef'")
        functions = self._graph_storage.get_nodes(type_filter=['FunctionDef'])

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

        target_nodes = self._graph_storage.get_nodes_with_subwords()

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
            edges = self._graph_storage.get_edges(type_filter=global_edges)

            result = edges[["src", "dst"]]
            self._write_to_cache(result, cache_key)

        return result

    def load_edge_prediction(self):

        edge_types = self._graph_storage.get_edge_type_descriptions()

        # when using this objective remove following edges
        # defined_in_*, executed_*, prev, next, *global_edges
        edge_types_for_prediction = {
            "next", "defined_in_module", "defined_in_class", "defined_in_function",  # , "prev",
        } | {etype for etype in edge_types if etype.startswith("executed_") and not etype.endswith("_rev")}

        cache_key = self._get_df_hash(pd.Series(sorted(list(edge_types_for_prediction))))
        result = self._load_cache_if_exists(cache_key)
        if result is None:
            edges = self._graph_storage.get_edges(type_filter=edge_types_for_prediction)

            # if self.use_ns_groups:
            #     groups = self.get_negative_sample_groups()
            #     valid_nodes = valid_nodes.intersection(set(groups["id"].tolist()))

            result = edges[["id", "type"]]  # edges[["src", "dst", "type"]]
            self._write_to_cache(result, cache_key)

        return result.rename({"id": "src", "type": "dst"}, axis=1)

    @classmethod
    def load_type_prediction(cls, spec):
        from SourceCodeTools.code.data.type_annotation_dataset.type_parser import TypeHierarchyParser
        from SourceCodeTools.code.data.type_annotation_dataset.type_parser import type_is_valid

        type_ann = unpersist(join(spec["data_path"], cls.type_pred_filename))
        type_ann = type_ann.query("dst.apply(@type_is_valid)", local_dict={"type_is_valid": type_is_valid})
        type_ann["dst"] = type_ann["dst"].apply(
            lambda type_: TypeHierarchyParser(type_, normalize=True).assemble(max_level=spec["max_type_ann_level"], simplify_nodes=True)
        )

        cache_key = f"{cls._get_df_hash(type_ann)}_{spec['min_count_for_objectives']}"
        result = cls._load_cache_if_exists(spec, cache_key)

        if result is None:
            storage = cls.get_storage_instance(spec)
            node2id = storage.get_node_types()

            type_ann["src_type"] = type_ann["src"].apply(lambda x: node2id[x])

            type_ann = type_ann[
                type_ann["src_type"].apply(lambda type_: type_ in {"mention"})  # FunctionDef {"arg", "AnnAssign"})
            ]

            counter = Counter(type_ann["dst"])
            allowed = {item for item, count in counter.items() if count >= spec['min_count_for_objectives']}
            type_ann["dst"] = type_ann["dst"].apply(lambda type_: type_ if type_ in allowed else "<unk>")

            # type_ann["dst"] = type_ann["dst"].apply(lambda x: x.strip("\"").strip("'").split("[")[0].split(".")[-1])
            type_ann = filter_dst_by_freq(type_ann, spec["min_count_for_objectives"])
            type_ann = type_ann[["src", "dst"]]
            result = type_ann
            cls._write_to_cache(spec, result, cache_key)

        return result

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
        labels = self._graph_storage.get_nodes_for_classification()
        labels.query("src.map(@in_partition)", local_dict={"in_partition": lambda x: x in self._partition}, inplace=True)

        return labels

    @staticmethod
    def create_subword_masker(nodes, edges):
        """
        :return: SubwordMasker for all nodes that have subwords. Suitable for token prediction objective.
        """
        # TODO
        return SubwordMaskerNoPandas(nodes, edges)

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


class SourceGraphDatasetNoPandas(SourceGraphDataset):
    def __init__(
            self, *args, **kwargs
    ):
        super(SourceGraphDatasetNoPandas, self).__init__(*args, **kwargs)

    @classmethod
    def _remove_edges_with_restricted_types(cls, spec, edges):
        edges.filter(lambda row: row["type"] not in spec["edge_types_to_remove"])

    # @classmethod
    # def _initialize_storage(cls, spec, **kwargs):
    #     # StorageClass = OnDiskGraphStorageWithFastIterationNoPandas
    #     StorageClass = GraphStorageWorkerMetaAdapter
    #
    #     _graph_storage_path = OnDiskGraphStorageWithFastIterationNoPandas.get_storage_file_name(spec["data_path"])
    #     if not OnDiskGraphStorageWithFastIterationNoPandas.verify_imported(_graph_storage_path):
    #         self._graph_storage = OnDiskGraphStorageWithFastIterationNoPandas(_graph_storage_path)
    #         self._graph_storage.import_from_files(spec["data_path"])
    #         self._graph_storage.add_import_completed_flag(_graph_storage_path)
    #     # else:
    #         # self._graph_storage = StorageClass(_graph_storage_path)
    #     self._graph_storage = StorageClass(_graph_storage_path, OnDiskGraphStorageWithFastIterationNoPandas, 4)
    #     self._num_nodes = self._graph_storage.get_num_nodes()

    @staticmethod
    def _add_node_types_to_edges(nodes, edges):

        node_type_map = dict(zip(nodes['id'].values, nodes['type']))

        edges.eval("src_type = src.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges.eval("dst_type = dst.map(@node_type_map.get)", local_dict={"node_type_map": node_type_map}, inplace=True)
        edges = edges.astype({'src_type': 'category', 'dst_type': 'category'}, copy=False)

        return edges

    @classmethod
    def _add_custom_reverse(cls, spec, edges):
        raise NotImplementedError
        to_reverse = edges.query("type in @custom_reverse", local_dict={"custom_reverse": self.custom_reverse})

        to_reverse.eval("type = type.map(@add_rev)", local_dict={"add_rev": lambda type_: type_ + "_rev"}, inplace=True)
        to_reverse.rename({"src": "dst", "dst": "src"}, axis=1, inplace=True)

        new_id = edges["id"].max() + 1
        to_reverse["id"] = range(new_id, new_id + len(to_reverse))

        return edges.append(to_reverse[["src", "dst", "type"]])

    @classmethod
    def _create_hetero_graph(cls, spec, nodes, edges, **kwargs):
        node_data = kwargs.get("node_data", {})
        edge_data = kwargs.get("edge_data", {})

        node2type = dict(zip(nodes["id"], nodes["type"]))
        dense_node_id = OrderedDict()
        typed_edge_ids = defaultdict(OrderedDict)

        all_node_types, all_edge_types = cls.get_graph_types(spec)
        dense_node_types = dict(zip(all_node_types, range(len(all_node_types))))
        dense_edge_types = dict(zip(all_edge_types, range(len(all_edge_types))))

        nxg = nx.DiGraph()

        def add_node(graph, node_id, node_types, dense_ids, node_data):
            if node_id not in dense_ids:
                dense_ids[node_id] = len(dense_ids)
                attrs = {
                    "type": dense_node_types[node_types[src_id]],
                    "nx_id": dense_ids[node_id],
                    "original_id": node_id,
                    "current_type_mask": True
                }
                for field in node_data:
                    attrs[field] = node_data[field][node_id]
                graph.add_node(dense_ids[node_id], **attrs)

        for ind, edge_row in enumerate(edges):
            edge_id, edge_type, src_id, dst_id = edge_row["id"], edge_row["type"], edge_row["src"], edge_row["dst"]
            add_node(nxg, src_id, node2type, dense_node_id, node_data)
            add_node(nxg, dst_id, node2type, dense_node_id, node_data)
            edge_signature = (node2type[src_id], edge_type, node2type[dst_id])

            if src_id not in dense_node_id:
                dense_node_id[src_id] = len(dense_node_id)
                nxg.add_node(dense_node_id[src_id], type=node2type[src_id], original_id=src_id)
            if dst_id not in dense_node_id:
                dense_node_id[dst_id] = len(dense_node_id)
                nxg.add_node(dense_node_id[dst_id], type=node2type[dst_id], original_id=dst_id)

            typed_edge_ids_ = typed_edge_ids[edge_signature]
            typed_edge_ids_[edge_id] = len(typed_edge_ids_)

            edge_attrs = {
                "type": dense_edge_types[edge_type],
                "original_id": edge_id,
                "typed_edge_id": typed_edge_ids_[edge_id],
                "dense_edge_id": ind
            }
            for field in edge_data:
                edge_attrs[field] = edge_data[edge_id]

            nxg.add_edge(dense_node_id[src_id], dense_node_id[dst_id], **edge_attrs)

        return nxg

    # def _create_hetero_graph(self, nodes, edges, **kwargs):
    #     node_data = kwargs.get("node_data", {})
    #     edge_data = kwargs.get("edge_data", {})
    #
    #     def get_torch_type(canonical_type):
    #         torch_types = {
    #             "int32": torch.IntTensor,
    #             "int64": torch.LongTensor,
    #             "bool": torch.BoolTensor,
    #         }
    #         return torch_types[canonical_type]
    #
    #     node2type = dict(zip(nodes["id"], nodes["type"]))
    #     dense_node_id = OrderedDict()
    #     typed_subgraphs = defaultdict(list)
    #     typed_edge_ids = defaultdict(OrderedDict)
    #     node_data_ = dict()
    #     edge_data_ = dict()
    #
    #     for edge_row in edges:
    #         edge_id, edge_type, src_id, dst_id = edge_row["id"], edge_row["type"], edge_row["src"], edge_row["dst"]
    #         edge_signature = (node2type[src_id], edge_type, node2type[dst_id])
    #         if src_id not in dense_node_id:
    #             dense_node_id[src_id] = len(dense_node_id)
    #         if dst_id not in dense_node_id:
    #             dense_node_id[dst_id] = len(dense_node_id)
    #
    #         typed_subgraphs[edge_signature].append((dense_node_id[src_id], dense_node_id[dst_id]))
    #         typed_edge_ids_ = typed_edge_ids[edge_signature]
    #         assert edge_id not in typed_edge_ids_, \
    #             "Found edges without ids. This is likely happening because new edges were added without properly "\
    #             "setting edge ids. The same happens if one removes reverse edges and then adds some of the reverse "\
    #             "edges back."
    #         typed_edge_ids_[edge_id] = len(typed_edge_ids_)
    #
    #     # edge_data_["typed_id"] = typed_edge_ids
    #     edge_data_["original_id"] = dict(zip(edges["id"], edges["id"]))
    #     # node_data_["typed_id"] = dense_node_id
    #     node_data_["original_id"] = dict(zip(dense_node_id.keys(), dense_node_id.keys()))
    #
    #     # need to make metagraphs the same for all graphs. lines below assume node types is set to false always
    #     assert self.use_node_types is False
    #     _, all_edge_types = self.get_graph_types()
    #     for etype in all_edge_types:
    #         signature = ("node_", etype, "node_")
    #         if signature not in typed_subgraphs:
    #             typed_subgraphs[signature] = list()
    #
    #     num_nodes = len(nodes)
    #     graph = dgl.heterograph(typed_subgraphs, num_nodes_dict={ntype: num_nodes for ntype in set(nodes["type"])})
    #
    #     # logging.info(
    #     #     f"Unique triplet types in the graph: {len(typed_subgraphs.keys())}"
    #     # )
    #
    #     def infer_type(dict_):
    #         assert len(dict_) > 0
    #         values = dict_.values()
    #         item_ = next(iter(values))
    #         if isinstance(item_, bool):
    #             return "bool"
    #         elif isinstance(item_, int):
    #             return "int64"
    #
    #     def attach_node_data(graph, node_data):
    #         node_order = list(dense_node_id.keys())
    #         assert len(graph.ntypes) == 1
    #
    #         for col_name in node_data:
    #             graph.nodes["node_"].data[col_name] = get_torch_type(infer_type(node_data[col_name]))(
    #                 list(map(lambda x: node_data[col_name][x], node_order)),
    #             )
    #
    #         graph.nodes["node_"].data["current_type_mask"] = torch.BoolTensor([True] * num_nodes)
    #
    #     def attach_edge_data(graph, edge_data):
    #         for subgraph_signature in graph.canonical_etypes:
    #             if subgraph_signature not in typed_edge_ids:
    #                 continue
    #             edge_order = typed_edge_ids[subgraph_signature].keys()
    #             for col_name in edge_data:
    #                 graph.edges[subgraph_signature].data[col_name] = get_torch_type(infer_type(edge_data[col_name]))(
    #                     list(map(lambda x: edge_data[col_name][x], edge_order)),
    #                 )
    #
    #     attach_node_data(graph, node_data)
    #     attach_edge_data(graph, edge_data)
    #     attach_node_data(graph, node_data_)
    #     attach_edge_data(graph, edge_data_)
    #     return graph

    @classmethod
    def _adjust_types(cls, spec, nodes, edges):
        nodes["type_backup"] = nodes["type"]
        edges["type_backup"] = edges["type"]
        nodes["type"] = map(
            partial(cls._strip_types_if_needed, stripping_flag=spec["use_node_types"], stripped_type="node_"),
            nodes["type"]
        )

        edges["type"] = map(
            partial(cls._strip_types_if_needed, stripping_flag=spec["use_edge_types"], stripped_type="edge_"),
            edges["type"]
        )

    @staticmethod
    def _add_type_nodes(nodes, edges):
        node_new_id = max(nodes["id"]) + 1
        edge_new_id = max(edges["id"]) + 1

        new_nodes = []
        new_edges = []
        added_type_nodes = {}

        node_slice = nodes[["id", "type"]]

        for id, type in node_slice:
            if type not in added_type_nodes:
                added_type_nodes[type] = node_new_id
                node_new_id += 1

                new_nodes.append({
                    "id": added_type_nodes[type],
                    "name": type,
                    "type": "type_node",
                })

            new_edges.append({
                "id": edge_new_id,
                "type": "node_type",
                "src": added_type_nodes[type],
                "dst": id,
            })
            edge_new_id += 1

        nodes.chunks.append(new_nodes)
        edges.chunks.append(new_edges)

        return nodes, edges

    @classmethod
    def _add_k_hop_edges(cls, spec, nodes, edges):
        g = nx.from_pandas_edgelist(
            edges, source="src", target="dst", create_using=nx.DiGraph, edge_attr="type"
        )

        skip_edges = cls._k_hop_skip_edges(spec)

        new_edge_id = max(edges["id"]) + 1

        def expand_edges(edges, node_id, view, edge_prefix, level=0):
            nonlocal new_edge_id

            if level < spec["k_hops"]:
                if edge_prefix != "":
                    edge_prefix += "|"
                for e in view:
                    next_edge_type = view[e]["type"]
                    if level > 0:
                        new_prefix = f"{level}_hop_connection"
                        # edges.append((node_id, e, new_prefix.rstrip("|")))
                        edges.append({"id": new_edge_id, "src": node_id, "dst": e, "type": new_prefix.rstrip("|")})
                        new_edge_id += 1
                    else:
                        new_prefix = edge_prefix + next_edge_type
                    # edges.append((node_id, e, new_prefix.rstrip("|")))
                    if next_edge_type in skip_edges:
                        continue
                    expand_edges(edges, node_id, g[e], new_prefix, level=level + 1)
                    # edges.extend(expand_edges(node_id, g[e], new_prefix, level=level+1))
            return edges

        new_edges = []
        for node in g.nodes:
            expand_edges(new_edges, node, g[node], "", level=0)

        edges.chunks.append(new_edges)

        edges.filter(lambda row: row["src"] != row["dst"])

        return nodes, edges

    @classmethod
    def _create_graph_from_nodes_and_edges(
            cls, spec, nodes, edges, node_data=None, edge_data=None, n_buckets=200000
    ):
        if node_data is None:
            node_data = {}
        if edge_data is None:
            edge_data = {}

        cls._remove_edges_with_restricted_types(spec, edges)

        # if self.custom_reverse is not None:
        #     edges = self._add_custom_reverse(edges)
        # self.ensure_connectedness(nodes, edges)

        if spec["k_hops"] > 0:
            nodes, edges = cls._add_k_hop_edges(spec, nodes, edges)

        if spec["type_nodes"]:
            nodes, edges = cls._add_type_nodes(nodes, edges)

        node_data["embedding_id"] = cls._get_node_name2bucket_mapping(
            cls._get_embeddable_names(nodes),
            n_buckets
        )

        cls._adjust_types(spec, nodes, edges)

        if len(edges) > 0:
            subgraph = cls._create_hetero_graph(spec, nodes, edges, node_data=node_data, edge_data=edge_data)
            return subgraph
        else:
            return None


def read_or_create_gnn_dataset(args, model_base, force_new=False, restore_state=False):
    if restore_state and not force_new:
        args = load_config(join(model_base, "dataset.config"))
        dataset = SourceGraphDatasetNoPandas.create_dataset_specification(
            storage_class=OnDiskGraphStorageWithFastIterationNoPandas,
            storage_kwargs={"path": join(args["data_path"], "dataset.db")},
            **args
        )
    else:
        dataset = SourceGraphDatasetNoPandas.create_dataset_specification(
            storage_class=OnDiskGraphStorageWithFastIterationNoPandas,
            storage_kwargs={"path": join(args["data_path"], "dataset.db")},
            **args
        )

        # save dataset state for recovery
        save_config(args, join(model_base, "dataset.config"))
    return dataset


class ProxyDataset(SourceGraphDataset):
    def __init__(self, *args, **kwargs):
        super(ProxyDataset, self).__init__(*args, **kwargs)

    def _initialize_storage(self, **kwargs):
        storage_kwargs = kwargs.get("storage_kwargs")
        if storage_kwargs is None:
            raise ValueError("Storage arguments are not provided")

        self._graph_storage = InMemoryGraphStorage(**storage_kwargs)

    def _initialize_partition_index(self, partition):
        # if self.type_nodes:
        #     nodes, edges = self._add_type_nodes(self._graph_storage._nodes, self._graph_storage._edges)
        # else:
        #     nodes, edges = self._graph_storage._nodes, self._graph_storage._edges
        all_ids = set(self._graph_storage.get_nodes()["id"]) | set(self._graph_storage.get_edges()["id"])
        partition = pd.DataFrame.from_dict({"id": list(all_ids)}, orient="columns")
        partition["train_mask"] = True
        partition["test_mask"] = True
        partition["val_mask"] = True
        self._partition = PartitionIndex(partition)

    def get_num_nodes(self):
        return self._graph_storage.get_num_nodes()

    def get_num_edges(self):
        return self._graph_storage.get_num_edges()

    def create_graph(self):
        return self._create_graph_from_nodes_and_edges(
            self._graph_storage.get_nodes(),
            self._graph_storage.get_edges()
        )


def test_dataset():
    import sys

    data_path = sys.argv[1]
    partition = sys.argv[2]
    # nodes_path = sys.argv[1]
    # edges_path = sys.argv[2]

    dataset = SourceGraphDatasetNoPandas(
        data_path, partition,
        use_node_types=False,
        use_edge_types=True,
        no_global_edges=True,
        remove_reverse=False,
        # custom_reverse=["global_mention"],
        tokenizer_path="/Users/LTV/Downloads/NitroShare/v2_subsample_no_spacy_v3/with_ast/sentencepiece_bpe.model",
        # type_nodes=True,
        # k_hops=2
    )
    # node_names = unpersist("/Users/LTV/Downloads/NitroShare/v2_subsample_v4_new_ast2_fixed_distinct_types/with_ast/node_names.json.bz2")
    # node_names = dataset.load_node_names(min_count=5)
    # token_names = dataset.load_token_prediction()
    # global_edges = dataset.load_global_edges_prediction()
    # type_labels = dataset.load_type_prediction()
    # node_classes = dataset.load_node_classes()
    # mapping = compact_property(node_names['dst'])
    # node_names['dst'] = node_names['dst'].apply(mapping.get)
    # from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
    # from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedder
    # from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedderWithBpeSubwords


    # from SourceCodeTools.code.data.dataset.DataLoader import SGNodesDataLoader
    # dataloader = SGNodesDataLoader(dataset, labels_for="nodes", number_of_hops=3, batch_size=512, labels=node_names,
    #     masker_fn=dataset.create_node_name_masker,
    #     label_loader_class=TargetLoader,
    #     label_loader_params={
    #         "emb_size": 100,
    #         "tokenizer_path": "/Users/LTV/Downloads/NitroShare/v2_subsample_no_spacy_v3/with_ast/sentencepiece_bpe.model"
    #     })
    # train_data = dataloader.get_dataloader(
    #     partition_label="train",
    # )


    from tqdm import tqdm
    for graph in tqdm(dataset.iterate_subgraphs(how=SGPartitionStrategies["file"], groups=None, node_data=None, edge_data=None, subgraph_data=None, n_buckets=100)):
        pass

    # sm = dataset.create_subword_masker()
    print(dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    test_dataset()
