import shelve
import shutil
import tempfile
from os.path import join
from tqdm import tqdm

from SourceCodeTools.code.data.sourcetrail.common import map_offsets
from SourceCodeTools.code.data.sourcetrail.sourcetrail_filter_type_edges import filter_type_edges
from SourceCodeTools.code.data.sourcetrail.sourcetrail_map_id_columns import map_columns
from SourceCodeTools.code.data.sourcetrail.sourcetrail_merge_graphs import get_global_node_info, merge_global_with_local
from SourceCodeTools.code.data.sourcetrail.sourcetrail_node_local2global import get_local2global
from SourceCodeTools.code.data.sourcetrail.sourcetrail_node_name_merge import merge_names
from SourceCodeTools.code.data.sourcetrail.sourcetrail_decode_edge_types import decode_edge_types
from SourceCodeTools.code.data.sourcetrail.sourcetrail_filter_ambiguous_edges import filter_ambiguous_edges
from SourceCodeTools.code.data.sourcetrail.sourcetrail_parse_bodies2 import process_bodies
from SourceCodeTools.code.data.sourcetrail.sourcetrail_call_seq_extractor import extract_call_seq
from SourceCodeTools.code.data.sourcetrail.sourcetrail_add_reverse_edges import add_reverse_edges
from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges2 import get_ast_from_modules
from SourceCodeTools.code.data.sourcetrail.sourcetrail_extract_variable_names import extract_var_names
from SourceCodeTools.code.data.sourcetrail.sourcetrail_extract_node_names import extract_node_names

from SourceCodeTools.code.data.sourcetrail.file_utils import *


class DatasetCreator:
    def __init__(
            self, path, lang,
            bpe_tokenizer, create_subword_instances,
            connect_subwords, only_with_annotations,
            do_extraction=False, visualize=False, track_offsets=False, remove_type_annotations=False
    ):
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

        paths = (os.path.join(path, dir) for dir in os.listdir(path))
        self.environments = sorted(list(filter(lambda path: os.path.isdir(path), paths)), key=lambda x: x.lower())

        self.init_cache()

        from SourceCodeTools.code.data.sourcetrail.common import UNRESOLVED_SYMBOL
        self.unsolved_symbol = UNRESOLVED_SYMBOL

    def init_cache(self):
        char_ranges = [chr(i) for i in range(ord("a"), ord("a")+26)] + [chr(i) for i in range(ord("A"), ord("A")+26)] + [chr(i) for i in range(ord("0"), ord("0")+10)]
        from random import sample
        rnd_name = "".join(sample(char_ranges, k=10))

        self.tmp_dir = os.path.join(tempfile.gettempdir(), rnd_name)
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)

        self.local2global_cache_filename = os.path.join(self.tmp_dir,"local2global_cache.db")
        self.local2global_cache = shelve.open(self.local2global_cache_filename)

    def __del__(self):
        self.local2global_cache.close()
        shutil.rmtree(self.tmp_dir)
        # os.remove(self.local2global_cache_filename) # TODO nofile on linux, need to check

    def merge(self, output_directory):

        if self.extract:
            logging.info("Extracting...")
            self.do_extraction()

        no_ast_path, with_ast_path = self.create_output_dirs(output_directory)

        if not self.only_with_annotations:
            self.create_global_file("nodes.bz2", "local2global.bz2", ['id'],
                                    join(no_ast_path, "common_nodes.bz2"), message="Merging nodes", ensure_unique_with=['type', 'serialized_name'])
            self.create_global_file("edges.bz2", "local2global.bz2", ['target_node_id', 'source_node_id'],
                                    join(no_ast_path, "common_edges.bz2"), message="Merging edges")
            self.create_global_file("source_graph_bodies.bz2", "local2global.bz2", ['id'],
                                    join(no_ast_path, "common_source_graph_bodies.bz2"), "Merging bodies", columns_special=[("replacement_list", map_offsets)])
            self.create_global_file("function_variable_pairs.bz2", "local2global.bz2", ['src'],
                                    join(no_ast_path, "common_function_variable_pairs.bz2"), "Merging variables")
            self.create_global_file("call_seq.bz2", "local2global.bz2", ['src', 'dst'],
                                    join(no_ast_path, "common_call_seq.bz2"), "Merging call seq")

            global_nodes = self.filter_orphaned_nodes(
                unpersist(join(no_ast_path, "common_nodes.bz2")), no_ast_path
            )
            persist(global_nodes, join(no_ast_path, "common_nodes.bz2"))
            node_names = extract_node_names(
                global_nodes, min_count=2
            )
            if node_names is not None:
                persist(node_names, join(no_ast_path, "node_names.bz2"))

            self.handle_parallel_edges(join(no_ast_path, "common_edges.bz2"))

            if self.visualize:
                self.visualize_func(
                    unpersist(join(no_ast_path, "common_nodes.bz2")),
                    unpersist(join(no_ast_path, "common_edges.bz2")),
                    join(no_ast_path, "visualization.pdf")
                )

        self.create_global_file("nodes_with_ast.bz2", "local2global_with_ast.bz2", ['id', 'mentioned_in'],
                                join(with_ast_path, "common_nodes.bz2"), message="Merging nodes with ast", ensure_unique_with=['type', 'serialized_name'])
        self.create_global_file("edges_with_ast.bz2", "local2global_with_ast.bz2", ['target_node_id', 'source_node_id', 'mentioned_in'],
                                join(with_ast_path, "common_edges.bz2"), "Merging edges with ast")
        self.create_global_file("source_graph_bodies.bz2", "local2global_with_ast.bz2", ['id'],
                                join(with_ast_path, "common_source_graph_bodies.bz2"), "Merging bodies with ast", columns_special=[("replacement_list", map_offsets)])
        self.create_global_file("function_variable_pairs.bz2", "local2global_with_ast.bz2", ['src'],
                                join(with_ast_path, "common_function_variable_pairs.bz2"), "Merging variables with ast")
        self.create_global_file("call_seq.bz2", "local2global_with_ast.bz2", ['src', 'dst'],
                                join(with_ast_path, "common_call_seq.bz2"), "Merging call seq with ast")
        self.create_global_file("offsets.bz2", "local2global_with_ast.bz2", ['node_id'],
                                join(with_ast_path, "common_offsets.bz2"), "Merging offsets with ast",
                                columns_special=[("mentioned_in", map_offsets)])
        self.create_global_file("filecontent_with_package.bz2", "local2global_with_ast.bz2", [],
                                join(with_ast_path, "common_filecontent.bz2"), "Merging filecontents")

        global_nodes = self.filter_orphaned_nodes(
            unpersist(join(with_ast_path, "common_nodes.bz2")), with_ast_path
        )
        persist(global_nodes, join(with_ast_path, "common_nodes.bz2"))
        node_names = extract_node_names(
            global_nodes, min_count=2
        )
        if node_names is not None:
            persist(node_names, join(with_ast_path, "node_names.bz2"))

        if self.remove_type_annotations:
            no_annotations, annotations = filter_type_edges(
                unpersist(join(with_ast_path, "common_nodes.bz2")),
                unpersist(join(with_ast_path, "common_edges.bz2"))
            )
            persist(no_annotations, join(with_ast_path, "common_edges.bz2"))
            if annotations is not None:
                persist(annotations, join(with_ast_path, "type_annotations.bz2"))

        self.handle_parallel_edges(join(with_ast_path, "common_edges.bz2"))

        self.post_pruning(join(with_ast_path, "common_nodes.bz2"), join(with_ast_path, "common_edges.bz2"))

        if self.visualize:
            self.visualize_func(
                unpersist(join(with_ast_path, "common_nodes.bz2")),
                unpersist(join(with_ast_path, "common_edges.bz2")),
                join(with_ast_path, "visualization.pdf")
            )

    def handle_parallel_edges(self, path):
        edges = unpersist(path)
        edges["id"] = list(range(len(edges)))

        edge_priority = {
            "next": -1, "prev": -1, "global_mention": -1, "global_mention_rev": -1,
            "calls": 0,
            "called_by": 0,
            "defines": 1,
            "defined_in": 1,
            "inheritance": 1,
            "inherited_by": 1,
            "imports": 1,
            "imported_by": 1,
            "uses": 2,
            "used_by": 2,
            "uses_type": 2,
            "type_used_by": 2, "mention_scope": 10, "mention_scope_rev": 10, "defined_in_function": 4, "defined_in_function_rev": 4, "defined_in_class": 5, "defined_in_class_rev": 5, "defined_in_module": 6, "defined_in_module_rev": 6
        }

        edge_bank = {}
        for id, type, src, dst in edges[["id", "type", "source_node_id", "target_node_id"]].values:
            key = (src, dst)
            if key in edge_bank:
                edge_bank[key].append((id, type))
            else:
                edge_bank[key] = [(id, type)]

        ids_to_remove = set()
        for key, parallel_edges in edge_bank.items():
            if len(parallel_edges) > 1:
                parallel_edges = sorted(parallel_edges, key=lambda x: edge_priority.get(x[1],3))
                ids_to_remove.update(pe[0] for pe in parallel_edges[1:])

        edges = edges[
            edges["id"].apply(lambda id_: id_ not in ids_to_remove)
        ]

        edges["id"] = range(len(edges))

        persist(edges, path)

    def post_pruning(self, npath, epath):
        nodes = unpersist(npath)
        edges = unpersist(epath)

        restricted_edges = {"global_mention_rev"}
        restricted_in_types = {
            "Op", "Constant", "#attr#", "#keyword#",
            'CtlFlow', 'JoinedStr', 'Name', 'ast_Literal',
            'subword', 'type_annotation'
        }

        restricted_nodes = set(nodes[
            nodes["type"].apply(lambda type_: type_ in restricted_in_types)
        ]["id"].tolist())

        edges = edges[
            edges["type"].apply(lambda type_: type_ not in restricted_edges)
        ]

        edges = edges[
            edges["target_node_id"].apply(lambda type_: type_ not in restricted_nodes)
        ]

        persist(edges, epath)

    def do_extraction(self):
        global_nodes = None
        global_nodes_with_ast = None

        for env_path in self.environments:
            logging.info(f"Found {os.path.basename(env_path)}")

            if not self.is_indexed(env_path):
                logging.info("Package not indexed")
                continue

            nodes, edges, source_location, occurrence, filecontent, element_component = \
                self.read_sourcetrail_files(env_path)

            if nodes is None:
                logging.info("Index is empty")
                continue

            edges = filter_ambiguous_edges(edges, element_component)

            nodes, edges = self.filter_unsolved_symbols(nodes, edges)

            bodies = process_bodies(nodes, edges, source_location, occurrence, filecontent, self.lang)
            call_seq = extract_call_seq(nodes, edges, source_location, occurrence)

            edges = add_reverse_edges(edges)

            # if bodies is not None:
            ast_nodes, ast_edges, offsets = get_ast_from_modules(
                nodes, edges, source_location, occurrence, filecontent,
                self.bpe_tokenizer, self.create_subword_instances, self.connect_subwords, self.lang,
                track_offsets=self.track_offsets
            )

            if offsets is not None:
                offsets["package"] = os.path.basename(env_path)

            nodes_with_ast = nodes.append(ast_nodes)
            edges_with_ast = edges.append(ast_edges)

            if bodies is not None:
                vars = extract_var_names(nodes, bodies, self.lang)
            else:
                vars = None

            global_nodes = self.merge_with_global(global_nodes, nodes)
            global_nodes_with_ast = self.merge_with_global(global_nodes_with_ast, nodes_with_ast)

            local2global = get_local2global(
                global_nodes=global_nodes, local_nodes=nodes
            )
            local2global_with_ast = get_local2global(
                global_nodes=global_nodes_with_ast, local_nodes=nodes_with_ast
            )

            self.write_local(env_path, nodes, edges, bodies, call_seq, vars,
                             nodes_with_ast, edges_with_ast, offsets,
                             local2global, local2global_with_ast)

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

    def create_output_dirs(self, output_path):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        no_ast_path = join(output_path, "no_ast")
        with_ast_path = join(output_path, "with_ast")

        if not self.only_with_annotations:
            if not os.path.isdir(no_ast_path):
                os.mkdir(no_ast_path)
        if not os.path.isdir(with_ast_path):
            os.mkdir(with_ast_path)

        return no_ast_path, with_ast_path

    def is_indexed(self, path):
        basename = os.path.basename(path)
        if os.path.isfile(os.path.join(path, f"{basename}.srctrldb")):
            return True
        else:
            return False

    def get_csv_name(self, name, path):
        return os.path.join(path, filenames[name])

    def filter_unsolved_symbols(self, nodes, edges):
        unsolved = set(nodes.query(f"serialized_name == '{self.unsolved_symbol}'")["id"].tolist())
        if len(unsolved) > 0:
            nodes.query("id not in @unsolved", local_dict={"unsolved": unsolved}, inplace=True)
            edges.query("source_node_id not in @unsolved", local_dict={"unsolved": unsolved}, inplace=True)
            edges.query("target_node_id not in @unsolved", local_dict={"unsolved": unsolved}, inplace=True)
        return nodes, edges

    def read_sourcetrail_files(self, env_path):
        nodes = merge_names(self.get_csv_name("nodes_csv", env_path), exit_if_empty=False)
        edges = decode_edge_types(self.get_csv_name("edges_csv", env_path), exit_if_empty=False)
        source_location = unpersist_if_present(self.get_csv_name("source_location", env_path))
        occurrence = unpersist_if_present(self.get_csv_name("occurrence", env_path))
        filecontent = unpersist_if_present(self.get_csv_name("filecontent", env_path))
        element_component = read_element_component(env_path)

        if nodes is None or edges is None or source_location is None or \
                occurrence is None or filecontent is None:
            # it is fine if element_component is None
            return None, None, None, None, None, None
        else:
            return nodes, edges, source_location, occurrence, filecontent, element_component

    def write_local(self, dir, nodes, edges, bodies, call_seq, vars,
                    nodes_with_ast, edges_with_ast, offsets,
                    local2global, local2global_with_ast):
        write_nodes(nodes, dir)
        write_edges(edges, dir)
        if bodies is not None:
            write_processed_bodies(bodies, dir)
        if call_seq is not None:
            persist(call_seq, join(dir, filenames['call_seq']))
        if vars is not None:
            persist(vars, join(dir, filenames['function_variable_pairs']))
        persist(nodes_with_ast, join(dir, "nodes_with_ast.bz2"))
        persist(edges_with_ast, join(dir, "edges_with_ast.bz2"))
        if offsets is not None:
            persist(offsets, join(dir, "offsets.bz2"))
        if len(edges_with_ast.query("type == 'annotation_for' or type == 'returned_by'")) > 0:
            with open(join(dir, "has_annotations"), "w") as has_annotations:
                pass
        persist(local2global, join(dir, "local2global.bz2"))
        persist(local2global_with_ast, join(dir, "local2global_with_ast.bz2"))

        # add package name to filecontent
        filecontent = read_filecontent(dir)
        filecontent["package"] = os.path.basename(dir)
        persist(filecontent, join(dir, "filecontent_with_package.bz2"))

    def get_global_node_info(self, global_nodes):
        """
        :param global_nodes: nodes from a global merged graph
        :return: Set of existing nodes represented with (type, node_name), minimal available free id
        """
        if global_nodes is None:
            existing_nodes, next_valid_id = set(), 0
        else:
            existing_nodes, next_valid_id = get_global_node_info(global_nodes)
        return existing_nodes, next_valid_id

    def merge_with_global(self, global_nodes, local_nodes):
        """
        Merge nodes obtained from the source code with the previously existing nodes.
        :param global_nodes: Nodes from a global inter-package graph
        :param local_nodes: Nodes from a local file-level graph
        :return: Updated version of the global inter-package graph
        """
        existing_nodes, next_valid_id = self.get_global_node_info(global_nodes)
        new_nodes = merge_global_with_local(existing_nodes, next_valid_id, local_nodes)

        if global_nodes is None:
            global_nodes = new_nodes
        else:
            global_nodes = global_nodes.append(new_nodes)

        return global_nodes

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

    def create_global_file(self, local_file, local2global_file, columns, output_path, message, ensure_unique_with=None, columns_special=None):
        global_table = None
        for ind, env_path in tqdm(
                enumerate(self.environments), desc=message, leave=True,
                dynamic_ncols=True, total=len(self.environments)
        ):
            global_table = self.merge_files(
                env_path, local_file, local2global_file, columns, global_table, columns_special=columns_special
            )

        if ensure_unique_with is not None:
            global_table = global_table.drop_duplicates(subset=ensure_unique_with)

        if global_table is not None:
            global_table.reset_index(drop=True, inplace=True)
            assert len(global_table) == len(global_table.index.unique())

            persist(global_table, output_path)

    def filter_orphaned_nodes(self, global_nodes, output_dir):
        edges = unpersist(join(output_dir, "common_edges.bz2"))
        active_nodes = set(edges['source_node_id'].tolist() + edges['target_node_id'].tolist())
        global_nodes = global_nodes[
            global_nodes['id'].apply(lambda id_: id_ in active_nodes)
        ]
        return global_nodes

    def visualize_func(self, nodes, edges, output_path):
        from SourceCodeTools.code.data.sourcetrail.sourcetrail_draw_graph import visualize
        visualize(nodes, edges, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Merge indexed environments into a single graph')
    parser.add_argument('indexed_environments',
                        help='Path to environments indexed by sourcetrail')
    parser.add_argument('output_directory',
                        help='')
    parser.add_argument('--language', "-l", dest="language", default="python",
                        help='Path to environments indexed by sourcetrail')
    parser.add_argument('--bpe_tokenizer', '-bpe', dest='bpe_tokenizer', type=str,
                        help='')
    parser.add_argument('--create_subword_instances', action='store_true', default=False, help="")
    parser.add_argument('--connect_subwords', action='store_true', default=False,
                        help="Takes effect only when `create_subword_instances` is False")
    parser.add_argument('--only_with_annotations', action='store_true', default=False, help="")
    parser.add_argument('--do_extraction', action='store_true', default=False, help="")
    parser.add_argument('--visualize', action='store_true', default=False, help="")
    parser.add_argument('--track_offsets', action='store_true', default=False, help="")
    parser.add_argument('--remove_type_annotations', action='store_true', default=False, help="")


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

    dataset = DatasetCreator(args.indexed_environments, args.language,
                             args.bpe_tokenizer, args.create_subword_instances,
                             args.connect_subwords, args.only_with_annotations,
                             args.do_extraction, args.visualize, args.track_offsets, args.remove_type_annotations)
    dataset.merge(args.output_directory)
