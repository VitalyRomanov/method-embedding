import logging
import os
from os.path import join

from SourceCodeTools.cli_arguments import DatasetCreatorArguments
from SourceCodeTools.code.annotator_utils import map_offsets
from SourceCodeTools.code.common import read_nodes
from SourceCodeTools.code.data.AbstractDatasetCreator import AbstractDatasetCreator
from SourceCodeTools.code.data.ast_graph.filter_type_edges import filter_type_edges_with_chunks
from SourceCodeTools.code.data.file_utils import filenames, unpersist_if_present, read_element_component
from SourceCodeTools.code.data.sourcetrail.sourcetrail_filter_type_edges import filter_type_edges
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


class DatasetCreator(AbstractDatasetCreator):
    """
    Merges several environments indexed with Sourcetrail into a single graph.
    """

    merging_specification = {
        "nodes.bz2": {"columns_to_map": ['id'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name']},
        "edges.bz2": {"columns_to_map": ['target_node_id', 'source_node_id'], "output_path": "common_edges.json"},
        "bodies.bz2": {"columns_to_map": ['id'], "output_path": "common_bodies.json", "columns_special": [("replacement_list", map_offsets)]},
        "function_variable_pairs.bz2": {"columns_to_map": ['src'], "output_path": "common_function_variable_pairs.json"},
        "call_seq.bz2": {"columns_to_map": ['src', 'dst'], "output_path": "common_call_seq.json"},

        "nodes_with_ast.bz2": {"columns_to_map": ['id', 'mentioned_in'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name']},
        "edges_with_ast.bz2": {"columns_to_map": ['target_node_id', 'source_node_id', 'mentioned_in'], "output_path": "common_edges.json"},
        "offsets.bz2": {"columns_to_map": ['node_id'], "output_path": "common_offsets.json", "columns_special": [("mentioned_in", map_offsets)]},
        "filecontent_with_package.bz2": {"columns_to_map": [], "output_path": "common_filecontent.json"},
        "name_mappings.bz2": {"columns_to_map": [], "output_path": "common_name_mappings.json"},
    }

    files_for_merging = [
        "nodes.bz2", "edges.bz2", "bodies.bz2", "function_variable_pairs.bz2", "call_seq.bz2"
    ]
    files_for_merging_with_ast = [
        "nodes_with_ast.bz2", "edges_with_ast.bz2", "bodies.bz2", "function_variable_pairs.bz2",
        "call_seq.bz2", "offsets.bz2", "filecontent_with_package.bz2", "name_mappings.bz2"
    ]

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
        "type_used_by": 2,
        "mention_scope": 10,
        "mention_scope_rev": 10,
        "defined_in_function": 4,
        "defined_in_function_rev": 4,
        "defined_in_class": 5,
        "defined_in_class_rev": 5,
        "defined_in_module": 6,
        "defined_in_module_rev": 6
    }

    restricted_edges = {"global_mention_rev"}
    restricted_in_types = {
        "Op", "Constant", "#attr#", "#keyword#",
        'CtlFlow', 'JoinedStr', 'Name', 'ast_Literal',
        'subword', 'type_annotation'
    }

    type_annotation_edge_types = ['annotation_for', 'returned_by']

    def __init__(
            self, path, lang,
            bpe_tokenizer, create_subword_instances,
            connect_subwords, only_with_annotations,
            do_extraction=False, visualize=False, track_offsets=False, remove_type_annotations=False,
            recompute_l2g=False
    ):
        super().__init__(
            path, lang, bpe_tokenizer, create_subword_instances, connect_subwords, only_with_annotations,
            do_extraction, visualize, track_offsets, remove_type_annotations, recompute_l2g
        )

        from SourceCodeTools.code.data.sourcetrail.common import UNRESOLVED_SYMBOL
        self.unsolved_symbol = UNRESOLVED_SYMBOL

    def _prepare_environments(self):
        paths = (os.path.join(self.path, dir) for dir in os.listdir(self.path))
        self.environments = sorted(list(filter(lambda path: os.path.isdir(path), paths)), key=lambda x: x.lower())

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

    @staticmethod
    def is_indexed(path):
        basename = os.path.basename(path)
        if os.path.isfile(os.path.join(path, f"{basename}.srctrldb")):
            return True
        else:
            return False

    @staticmethod
    def get_csv_name(name, path):
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

    def do_extraction(self):
        global_nodes = set()
        global_nodes_with_ast = set()

        for env_path in self.environments:
            package_name = os.path.basename(env_path)

            logging.info(f"Found {package_name}")

            if not self.is_indexed(env_path):
                logging.info("Package not indexed")
                continue

            if not self.recompute_l2g:

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
                ast_nodes, ast_edges, offsets, name_mappings = get_ast_from_modules(
                    nodes, edges, source_location, occurrence, filecontent,
                    self.bpe_tokenizer, self.create_subword_instances, self.connect_subwords, self.lang,
                    track_offsets=self.track_offsets, package_name=package_name
                )

                if offsets is not None:
                    offsets["package"] = package_name
                filecontent["package"] = package_name
                # edges["package"] = os.path.basename(env_path)

                # need this check in situations when module has a single file and this file cannot be parsed
                nodes_with_ast = nodes.append(ast_nodes) if ast_nodes is not None else nodes
                edges_with_ast = edges.append(ast_edges) if ast_edges is not None else edges

                if bodies is not None:
                    vars = extract_var_names(nodes, bodies, self.lang)
                else:
                    vars = None
            else:
                nodes = unpersist_if_present(join(env_path, "nodes.bz2"))
                nodes_with_ast = unpersist_if_present(join(env_path, "nodes_with_ast.bz2"))

                if nodes is None or nodes_with_ast is None:
                    continue

                edges = bodies = call_seq = vars = edges_with_ast = offsets = name_mappings = filecontent = None

            # global_nodes = self.merge_with_global(global_nodes, nodes)
            # global_nodes_with_ast = self.merge_with_global(global_nodes_with_ast, nodes_with_ast)

            local2global = get_local2global(
                global_nodes=global_nodes, local_nodes=nodes
            )
            local2global_with_ast = get_local2global(
                global_nodes=global_nodes_with_ast, local_nodes=nodes_with_ast
            )

            global_nodes.update(local2global["global_id"])
            global_nodes_with_ast.update(local2global_with_ast["global_id"])

            self.write_type_annotation_flag(edges_with_ast, env_path)

            self.write_local(
                env_path, global_nodes=nodes, global_edges=edges, ast_nodes=ast_nodes, ast_edges=ast_edges, bodies=bodies, call_seq=call_seq, function_variable_pairs=vars,
                nodes_with_ast=nodes_with_ast, edges_with_ast=edges_with_ast, offsets=offsets,
                local2global=local2global, local2global_with_ast=local2global_with_ast,
                name_mappings=name_mappings, filecontent_with_package=filecontent
            )

        self.compact_mapping_for_l2g(global_nodes, "local2global.bz2")
        self.compact_mapping_for_l2g(global_nodes_with_ast, "local2global_with_ast.bz2")

    @staticmethod
    def extract_node_names(nodes_path, min_count):
        logging.info("Extract node names")
        return extract_node_names(read_nodes(nodes_path), min_count=min_count)

    def filter_type_edges(self, nodes_path, edges_path):
        logging.info("Filter type edges")
        filter_type_edges_with_chunks(nodes_path, edges_path, kwarg_fn=self.get_writing_mode)

    def merge(self, output_directory):

        if self.extract:
            logging.info("Extracting...")
            self.do_extraction()

        no_ast_path, with_ast_path = self.create_output_dirs(output_directory)

        if not self.only_with_annotations:
            self.merge_graph_without_ast(no_ast_path)

        self.merge_graph_with_ast(with_ast_path)

    def visualize_func(self, nodes, edges, output_path, **kwargs):
        from SourceCodeTools.code.data.sourcetrail.sourcetrail_draw_graph import visualize
        visualize(nodes, edges, output_path, **kwargs)


if __name__ == "__main__":

    args = DatasetCreatorArguments().parse()

    if args.recompute_l2g:
        args.do_extraction = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

    dataset = DatasetCreator(
        args.indexed_environments, args.language, args.bpe_tokenizer, args.create_subword_instances,
        args.connect_subwords, args.only_with_annotations, args.do_extraction, args.visualize, args.track_offsets,
        args.remove_type_annotations, args.recompute_l2g
    )
    dataset.merge(args.output_directory)
