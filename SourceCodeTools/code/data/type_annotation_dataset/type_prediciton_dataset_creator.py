import logging
import os
from collections import defaultdict
from os.path import join

from tqdm import tqdm

from SourceCodeTools.cli_arguments import DatasetCreatorArgumentParser
from SourceCodeTools.code.annotator_utils import map_offsets
from SourceCodeTools.code.common import read_nodes
from SourceCodeTools.code.data.AbstractDatasetCreator import AbstractDatasetCreator
from SourceCodeTools.code.data.ast_graph.filter_type_edges import filter_type_edges_with_chunks
from SourceCodeTools.code.data.file_utils import filenames, unpersist_if_present, read_element_component, persist
from SourceCodeTools.code.data.sourcetrail.DatasetCreator2 import DatasetCreator
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


class TypePredDatasetCreator(DatasetCreator):
    """
    Merges several environments indexed with Sourcetrail into a single graph.
    """

    merging_specification = {
        "nodes.bz2": {"columns_to_map": ['id'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name'], "columns_to_filter": [], "columns_to_filter_special": []},
        "edges.bz2": {"columns_to_map": ['target_node_id', 'source_node_id'], "output_path": "common_edges.json", "columns_to_filter": [], "columns_to_filter_special": []},
        "bodies.bz2": {"columns_to_map": ['id'], "output_path": "common_bodies.json", "columns_special": [("replacement_list", map_offsets)], "columns_to_filter": [], "columns_to_filter_special": []},
        "function_variable_pairs.bz2": {"columns_to_map": ['src'], "output_path": "common_function_variable_pairs.json", "columns_to_filter": [], "columns_to_filter_special": []},
        "call_seq.bz2": {"columns_to_map": ['src', 'dst'], "output_path": "common_call_seq.json", "columns_to_filter": [], "columns_to_filter_special": []},

        "nodes_with_ast.bz2": {"columns_to_map": ['id', 'mentioned_in'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name'], "columns_to_filter": ["id"], "columns_to_filter_special": []},
        "edges_with_ast.bz2": {"columns_to_map": ['target_node_id', 'source_node_id', 'mentioned_in'], "output_path": "common_edges.json", "columns_to_filter": ["target_node_id", "source_node_id"], "columns_to_filter_special": []},
        "offsets.bz2": {"columns_to_map": ['node_id'], "output_path": "common_offsets.json", "columns_special": [("mentioned_in", map_offsets)], "columns_to_filter": ["node_id"], "columns_to_filter_special": []},
        "filecontent_with_package.bz2": {"columns_to_map": [], "output_path": "common_filecontent.json", "columns_to_filter": [], "columns_to_filter_special": []},
        "name_mappings.bz2": {"columns_to_map": [], "output_path": "common_name_mappings.json", "columns_to_filter": [], "columns_to_filter_special": []},
    }

    files_for_merging = [
        "nodes.bz2", "edges.bz2", "bodies.bz2", "function_variable_pairs.bz2", "call_seq.bz2"
    ]
    files_for_merging_with_ast = [
        "edges_with_ast.bz2", "nodes_with_ast.bz2", "bodies.bz2", "function_variable_pairs.bz2",
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
    restricted_nodes = None

    type_annotation_edge_types = ['annotation_for', 'returned_by']

    def __init__(
            self, path, lang,
            bpe_tokenizer, create_subword_instances,
            connect_subwords, only_with_annotations,
            do_extraction=False, visualize=False, track_offsets=False, remove_type_annotations=False,
            recompute_l2g=False
    ):
        logging.warning("Argument `only_with_annotations` is ignored")
        logging.warning("Argument `remove_type_annotations` is ignored")
        remove_type_annotations = True
        only_with_annotations = True
        super().__init__(
            path, lang, bpe_tokenizer, create_subword_instances, connect_subwords, only_with_annotations,
            do_extraction, visualize, track_offsets, remove_type_annotations, recompute_l2g
        )

        from SourceCodeTools.code.data.sourcetrail.common import UNRESOLVED_SYMBOL
        self.unsolved_symbol = UNRESOLVED_SYMBOL

    def create_global_file(
            self, local_file, local2global_file, columns_to_map, output_path, message, ensure_unique_with=None,
            columns_special=None, **kwargs
    ):
        assert output_path.endswith("json") or output_path.endswith("csv")

        if ensure_unique_with is not None:
            unique_values = set()
        else:
            unique_values = None

        first_written = False

        if self.restricted_nodes is None:
            self.restricted_nodes = defaultdict(set)

        for ind, env_path in tqdm(
                enumerate(self.environments), desc=message, leave=True,
                dynamic_ncols=True, total=len(self.environments)
        ):
            mapped_local = self.read_mapped_local(
                env_path, local_file, local2global_file, columns_to_map, columns_special=columns_special
            )

            if mapped_local is not None:
                if env_path not in self.restricted_nodes:
                    assert local_file == "edges_with_ast.bz2"
                    mentions_with_types = set(mapped_local.query("type == 'annotation_for' or type == 'returned_by'")["mentioned_in"])
                    mention_edges_with_na = mapped_local.query(f"mentioned_in.map(@mentions_with_types) or mentioned_in.isna()", local_dict={"mentions_with_types": lambda x: x in mentions_with_types})
                    mention_edges = mention_edges_with_na.query(f"not mentioned_in.isna()")
                    mention_edge_ids = set(mention_edges["id"])
                    mention_nodes = set(mention_edges['source_node_id']) | set(mention_edges['target_node_id'])
                    mention_edges = mention_edges.append(mention_edges_with_na.query(
                        "target_node_id.map(@mention_nodes) and type.map(@not_ends_with_rev) and mentioned_in.isna()",  # and id.map(@mention_edge_ids)",
                        local_dict={
                            "mention_nodes": lambda x: x in mention_nodes,
                            "mention_edge_ids": lambda x: x not in mention_edge_ids,
                            "not_ends_with_rev": lambda x: not x.endswith("rev"),
                        }
                    )).drop_duplicates()
                    mention_nodes = set(mention_edges['source_node_id']) | set(mention_edges['target_node_id'])
                    self.restricted_nodes[env_path].update(mention_nodes)

                for col in self.merging_specification[local_file]["columns_to_filter"]:
                    mapped_local = mapped_local.query(
                        f"{col}.map(@in_nodes)", local_dict={"in_nodes": lambda x: x in self.restricted_nodes[env_path]}
                    )

                for col in self.merging_specification[local_file]["columns_to_filter_special"]:
                    def filter_(offsets):
                        return [(s, e, v) for s, e, v in offsets if v in self.restricted_nodes[env_path]]

                    mapped_local[col] = mapped_local[col].apply(filter_)

                if unique_values is not None:
                    unique_verify = list(zip(*(mapped_local[col_name] for col_name in ensure_unique_with)))

                    mapped_local = mapped_local.loc[
                        map(lambda x: x not in unique_values, unique_verify)
                    ]
                    unique_values.update(unique_verify)

                kwargs = self.get_writing_mode(output_path.endswith("csv"), first_written)

                persist(mapped_local, output_path, **kwargs)
                first_written = True


if __name__ == "__main__":

    args = DatasetCreatorArgumentParser().parse()

    if args.recompute_l2g:
        args.do_extraction = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

    dataset = TypePredDatasetCreator(
        args.indexed_environments, args.language, args.bpe_tokenizer, args.create_subword_instances,
        args.connect_subwords, args.only_with_annotations, args.do_extraction, args.visualize, args.track_offsets,
        args.remove_type_annotations, args.recompute_l2g
    )
    dataset.merge(args.output_directory)
