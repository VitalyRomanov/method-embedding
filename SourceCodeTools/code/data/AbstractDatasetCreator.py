import logging
import os
import shelve
import shutil
import tempfile
from abc import abstractmethod
from collections import defaultdict
from copy import copy
from functools import partial
from os.path import join

from tqdm import tqdm

from SourceCodeTools.code.annotator_utils import map_offsets
from SourceCodeTools.code.common import map_columns, read_edges, read_nodes
from SourceCodeTools.code.data.file_utils import get_random_name, unpersist, persist, unpersist_if_present


class AbstractDatasetCreator:
    """
    Merges several environments indexed with Sourcetrail into a single graph.
    """

    merging_specification = {
        "nodes.bz2": {"columns": ['id'], "output_path": "common_nodes.jsonl", "ensure_unique_with": ['type', 'serialized_name']},
        "edges.bz2": {"columns": ['target_node_id', 'source_node_id'], "output_path": "common_edges.jsonl"},
        "source_graph_bodies.bz2": {"columns": ['id'], "output_path": "common_source_graph_bodies.jsonl", "columns_special": [("replacement_list", map_offsets)]},
        "function_variable_pairs.bz2": {"columns": ['src'], "output_path": "common_function_variable_pairs.jsonl"},
        "call_seq.bz2": {"columns": ['src', 'dst'], "output_path": "common_call_seq.jsonl"},

        "nodes_with_ast.bz2": {"columns": ['id', 'mentioned_in'], "output_path": "common_nodes.jsonl", "ensure_unique_with": ['type', 'serialized_name']},
        "edges_with_ast.bz2": {"columns": ['target_node_id', 'source_node_id', 'mentioned_in'], "output_path": "common_edges.jsonl"},
        "offsets.bz2": {"columns": ['node_id'], "output_path": "common_offsets.jsonl", "columns_special": [("mentioned_in", map_offsets)]},
        "filecontent_with_package.bz2": {"columns": [], "output_path": "common_filecontent.jsonl"},
        "name_mappings.bz2": {"columns": [], "output_path": "common_name_mappings.jsonl"},
    }

    files_for_merging = [
        "nodes.bz2", "edges.bz2", "source_graph_bodies.bz2", "function_variable_pairs.bz2", "call_seq.bz2"
    ]
    files_for_merging_with_ast = [
        "nodes_with_ast.bz2", "edges_with_ast.bz2", "source_graph_bodies.bz2", "function_variable_pairs.bz2",
        "call_seq.bz2", "offsets.bz2", "filecontent_with_package.bz2", "name_mappings.bz2"
    ]

    restricted_edges = {}
    restricted_in_types = {}

    type_annotation_edge_types = []

    environments = None
    edge_priority = dict()

    def __init__(
            self, path, lang, bpe_tokenizer, create_subword_instances, connect_subwords, only_with_annotations,
            do_extraction=False, visualize=False, track_offsets=False, remove_type_annotations=False,
            recompute_l2g=False
    ):
        """
        :param path: path to source code dataset
        :param lang: language to use for AST parser (only Python for now)
        :param bpe_tokenizer: path to bpe tokenizer model
        :param create_subword_instances: whether to create nodes that represent subword instances (doubles the
            number of nodes)
        :param connect_subwords: whether to connect subword instances so that the order of subwords is stored
            in the graph. Has effect only when create_subword_instances=True
        :param only_with_annotations: include only packages that have type annotations into the final graph
        :param do_extraction: when True, process source code and extract AT edges. Otherwise, existing files are
            used.
        :param visualize: visualize graph using pygraphviz and store as PDF (infeasible for large graphs)
        :param track_offsets: store offset information and map node occurrences to global graph ids
        :param remove_type_annotations: when True, removes all type annotations from the graph and stores then
            in a file called `type_annotations.bz2`
        :param recompute_l2g: when True, run merging operation again, without extrcting AST nodes and edges second time
        """
        self.indexed_path = path
        self.lang = lang
        self.bpe_tokenizer = bpe_tokenizer
        self.create_subword_instances = create_subword_instances
        self.connect_subwords = connect_subwords
        self.only_with_annotations = only_with_annotations
        self.extract = do_extraction
        self.visualize = visualize
        self.track_offsets = track_offsets
        self.remove_type_annotations = remove_type_annotations
        self.recompute_l2g = recompute_l2g

        self.path = path
        self._prepare_environments()

        self._init_cache()

    def _init_cache(self):
        # TODO this is wrong, use standard utilities
        rnd_name = get_random_name()

        self.tmp_dir = os.path.join(tempfile.gettempdir(), rnd_name)
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)

        self.local2global_cache_filename = os.path.join(self.tmp_dir, "local2global_cache.db")
        self.local2global_cache = shelve.open(self.local2global_cache_filename)

    def __del__(self):
        self.local2global_cache.close()
        shutil.rmtree(self.tmp_dir)
        # os.remove(self.local2global_cache_filename) # TODO nofile on linux, need to check

    def handle_parallel_edges(self, edges_path):
        logging.info("Handle parallel edges")
        last_id = 0

        temp_edges = join(os.path.dirname(edges_path), "temp_" + os.path.basename(edges_path))

        for ind, edges in enumerate(read_edges(edges_path, as_chunks=True)):
            edges["id"] = range(last_id, len(edges) + last_id)

            edge_bank = defaultdict(list)
            for id_, type_, src, dst in edges[["id", "type", "source_node_id", "target_node_id"]].values:
                edge_bank[(src, dst)].append((id_, type_))

            ids_to_remove = set()
            for key, parallel_edges in edge_bank.items():
                if len(parallel_edges) > 1:
                    parallel_edges = sorted(parallel_edges, key=lambda x: self.edge_priority.get(x[1], 3))
                    ids_to_remove.update(pe[0] for pe in parallel_edges[1:])

            edges = edges[
                edges["id"].apply(lambda id_: id_ not in ids_to_remove)
            ]

            edges["id"] = range(last_id, len(edges) + last_id)
            last_id = len(edges) + last_id

            kwargs = self.get_writing_mode(temp_edges.endswith("csv"), first_written=ind != 0)
            persist(edges, temp_edges, **kwargs)

        os.remove(edges_path)
        os.rename(temp_edges, edges_path)

    def post_pruning(self, nodes_path, edges_path):
        logging.info("Post pruning")

        restricted_nodes = set()

        for nodes in read_nodes(nodes_path, as_chunks=True):
            restricted_nodes.update(
                nodes[
                    nodes["type"].apply(lambda type_: type_ in self.restricted_in_types)
                ]["id"]
            )

        temp_edges = join(os.path.dirname(edges_path), "temp_" + os.path.basename(edges_path))

        for ind, edges in enumerate(read_edges(edges_path, as_chunks=True)):
            edges = edges[
                edges["type"].apply(lambda type_: type_ not in self.restricted_edges)
            ]

            edges = edges[
                edges["target_node_id"].apply(lambda type_: type_ not in restricted_nodes)
            ]

            kwargs = self.get_writing_mode(temp_edges.endswith("csv"), first_written=ind != 0)
            persist(edges, temp_edges, **kwargs)

        os.remove(edges_path)
        os.rename(temp_edges, edges_path)

    def compact_mapping_for_l2g(self, global_nodes, filename):
        if len(global_nodes) > 0:
            self.update_l2g_file(
                mapping=self.create_compact_mapping(global_nodes), filename=filename
            )

    @staticmethod
    def create_compact_mapping(node_ids):
        return dict(zip(node_ids, range(len(node_ids))))

    def update_l2g_file(self, mapping, filename):
        for env_path in tqdm(self.environments, desc=f"Fixing {filename}"):
            filepath = os.path.join(env_path, filename)
            if not os.path.isfile(filepath):
                continue
            l2g = unpersist(filepath)
            l2g["global_id"] = l2g["global_id"].apply(lambda id_: mapping.get(id_, None))
            persist(l2g, filepath)

    def get_local2global(self, path):
        if path in self.local2global_cache:
            return self.local2global_cache[path]
        else:
            local2global_df = unpersist_if_present(path)
            if local2global_df is None:
                return None
            else:
                local2global = dict(zip(local2global_df['id'], local2global_df['global_id']))
                self.local2global_cache[path] = local2global
                return local2global

    @staticmethod
    def persist_if_not_none(table, dir, name):
        if table is not None:
            path = os.path.join(dir, name)
            persist(table, path)

    def write_type_annotation_flag(self, edges, output_dir):
        if len(self.type_annotation_edge_types) > 0:
            query_str = " or ".join(f"type == '{edge_type}'" for edge_type in self.type_annotation_edge_types)
            if len(edges.query(query_str)) > 0:
                with open(os.path.join(output_dir, "has_annotations"), "w") as has_annotations:
                    pass

    def write_local(self, dir, local2global=None, local2global_with_ast=None, **kwargs):

        if not self.recompute_l2g:
            for var_name, var_ in kwargs.items():
                self.persist_if_not_none(var_, dir, var_name + ".bz2")

        self.persist_if_not_none(local2global, dir, "local2global.bz2")
        self.persist_if_not_none(local2global_with_ast, dir, "local2global_with_ast.bz2")

    def merge_files(self, env_path, filename, map_filename, columns_to_map, original, columns_special=None):
        input_table_path = join(env_path, filename)
        local2global = self.get_local2global(join(env_path, map_filename))
        if os.path.isfile(input_table_path) and local2global is not None:
            input_table = unpersist(input_table_path)
            if self.only_with_annotations:
                if not os.path.isfile(join(env_path, "has_annotations")):
                    return original
            new_table = map_columns(input_table, local2global, columns_to_map, columns_special=columns_special)
            if original is None:
                return new_table
            else:
                return original.append(new_table)
        else:
            return original

    def read_mapped_local(self, env_path, filename, map_filename, columns_to_map, columns_special=None):
        input_table_path = join(env_path, filename)
        local2global = self.get_local2global(join(env_path, map_filename))
        if os.path.isfile(input_table_path) and local2global is not None:
            if self.only_with_annotations:
                if not os.path.isfile(join(env_path, "has_annotations")):
                    return None
            input_table = unpersist(input_table_path)
            new_table = map_columns(input_table, local2global, columns_to_map, columns_special=columns_special)
            return new_table
        else:
            return None

    def get_writing_mode(self, is_csv, first_written):
        kwargs = {}
        if first_written is True:
            kwargs["mode"] = "a"
            if is_csv:
                kwargs["header"] = False
        return kwargs

    def create_global_file(
            self, local_file, local2global_file, columns, output_path, message, ensure_unique_with=None,
            columns_special=None
    ):
        assert output_path.endswith("json") or output_path.endswith("csv")

        if ensure_unique_with is not None:
            unique_values = set()
        else:
            unique_values = None

        first_written = False

        for ind, env_path in tqdm(
                enumerate(self.environments), desc=message, leave=True,
                dynamic_ncols=True, total=len(self.environments)
        ):
            mapped_local = self.read_mapped_local(
                env_path, local_file, local2global_file, columns, columns_special=columns_special
            )

            if mapped_local is not None:
                if unique_values is not None:
                    unique_verify = list(zip(*(mapped_local[col_name] for col_name in ensure_unique_with)))

                    mapped_local = mapped_local.loc[
                        map(lambda x: x not in unique_values, unique_verify)
                    ]
                    unique_values.update(unique_verify)

                kwargs = self.get_writing_mode(output_path.endswith("csv"), first_written)

                persist(mapped_local, output_path, **kwargs)
                first_written = True


    # def create_global_file(
    #         self, local_file, local2global_file, columns, output_path, message, ensure_unique_with=None,
    #         columns_special=None
    # ):
    #     global_table = None
    #     for ind, env_path in tqdm(
    #             enumerate(self.environments), desc=message, leave=True,
    #             dynamic_ncols=True, total=len(self.environments)
    #     ):
    #         global_table = self.merge_files(
    #             env_path, local_file, local2global_file, columns, global_table, columns_special=columns_special
    #         )
    #
    #     if ensure_unique_with is not None:
    #         global_table = global_table.drop_duplicates(subset=ensure_unique_with)
    #
    #     if global_table is not None:
    #         global_table.reset_index(drop=True, inplace=True)
    #         assert len(global_table) == len(global_table.index.unique())
    #
    #         persist(global_table, output_path)

    def filter_orphaned_nodes(self, nodes_path, edges_path):
        logging.info("Filter orphaned nodes")
        active_nodes = set()

        for edges in read_edges(edges_path, as_chunks=True):
            active_nodes.update(edges['source_node_id'])
            active_nodes.update(edges['target_node_id'])

        temp_nodes = join(os.path.dirname(nodes_path), "temp_" + os.path.basename(nodes_path))

        for ind, nodes in enumerate(read_nodes(nodes_path, as_chunks=True)):
            nodes = nodes[
                nodes['id'].apply(lambda id_: id_ in active_nodes)
            ]

            kwargs = self.get_writing_mode(temp_nodes.endswith("csv"), first_written=ind != 0)
            persist(nodes, temp_nodes, **kwargs)

        os.remove(nodes_path)
        os.rename(temp_nodes, nodes_path)

    def join_files(self, files, local2global_filename, output_dir):
        for file in files:
            params = copy(self.merging_specification[file])
            params["output_path"] = join(output_dir, params.pop("output_path"))
            self.create_global_file(file, local2global_filename, message=f"Merging {file}", **params)

    def merge_graph_without_ast(self, output_path):
        self.join_files(self.files_for_merging, "local2global.bz2", output_path)

        get_path = partial(join, output_path)

        nodes_path = get_path("common_nodes.json")
        edges_path = get_path("common_edges.json")

        self.filter_orphaned_nodes(
            nodes_path,
            edges_path,
        )
        node_names = self.extract_node_names(
            nodes_path, min_count=2
        )
        if node_names is not None:
            persist(node_names, get_path("node_names.json"))

        self.handle_parallel_edges(edges_path)

        if self.visualize:
            self.visualize_func(
                read_nodes(nodes_path),
                read_edges(edges_path),
                get_path("visualization.pdf")
            )

    def merge_graph_with_ast(self, output_path):

        self.join_files(self.files_for_merging_with_ast, "local2global_with_ast.bz2", output_path)

        get_path = partial(join, output_path)

        nodes_path = get_path("common_nodes.json")
        edges_path = get_path("common_edges.json")

        if self.remove_type_annotations:
            self.filter_type_edges(nodes_path, edges_path)

        self.handle_parallel_edges(edges_path)

        self.post_pruning(nodes_path, edges_path)

        self.filter_orphaned_nodes(
            nodes_path,
            edges_path,
        )
        # persist(global_nodes, get_path("common_nodes.json"))
        node_names = self.extract_node_names(
            nodes_path, min_count=2
        )
        if node_names is not None:
            persist(node_names, get_path("node_names.json"))

        if self.visualize:
            self.visualize_func(
                read_nodes(nodes_path),
                read_edges(edges_path),
                get_path("visualization.pdf")
            )

    @abstractmethod
    def create_output_dirs(self, output_path):
        pass

    @abstractmethod
    def _prepare_environments(self):
        pass

    @staticmethod
    @abstractmethod
    def filter_type_edges(nodes, edges):
        pass

    @staticmethod
    @abstractmethod
    def extract_node_names(nodes, min_count):
        pass

    @abstractmethod
    def do_extraction(self):
        pass

    @abstractmethod
    def merge(self, output_directory):
        pass

        if self.extract:
            logging.info("Extracting...")
            self.do_extraction()

        no_ast_path, with_ast_path = self.create_output_dirs(output_directory)

        if not self.only_with_annotations:
            self.merge_graph_without_ast(no_ast_path)

        self.merge_graph_with_ast(with_ast_path)

    @abstractmethod
    def visualize_func(self, nodes, edges, output_path):
        pass