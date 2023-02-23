import logging
import os
from collections import defaultdict
from itertools import chain
from os.path import join
from pathlib import Path
from random import random

import networkx as nx
import pandas as pd
from tqdm import tqdm

from SourceCodeTools.cli_arguments import DatasetCreatorArgumentParser
from SourceCodeTools.code.annotator_utils import map_offsets, adjust_offsets2
from SourceCodeTools.code.common import SQLTable
from SourceCodeTools.code.data.file_utils import unpersist_if_present, persist, \
    write_mapping_to_json, read_mapping_from_json
from SourceCodeTools.code.data.sourcetrail.DatasetCreator2 import DatasetCreator
from SourceCodeTools.code.data.sourcetrail.sourcetrail_node_local2global import get_local2global
from SourceCodeTools.code.data.sourcetrail.sourcetrail_filter_ambiguous_edges import filter_ambiguous_edges
from SourceCodeTools.code.data.sourcetrail.sourcetrail_parse_bodies2 import process_bodies
from SourceCodeTools.code.data.sourcetrail.sourcetrail_call_seq_extractor import extract_call_seq
from SourceCodeTools.code.data.sourcetrail.sourcetrail_add_reverse_edges import add_reverse_edges
from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges2 import get_ast_from_modules
from SourceCodeTools.code.data.sourcetrail.sourcetrail_extract_variable_names import extract_var_names
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping
from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_dataset import process_body as extract_type_annotations
from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.entity.entity_render import render_single


class HierarchicalDatasetCreator(DatasetCreator):
    """
    Merges several environments indexed with Sourcetrail into a single graph.
    """

    merging_specification = {
        "global_nodes.bz2": {"columns_to_map": ['id'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name']},
        "global_edges.bz2": {"columns_to_map": ['target_node_id', 'source_node_id'], "output_path": "common_edges.json"},
        "bodies.bz2": {"columns_to_map": ['id'], "output_path": "common_bodies.json", "columns_special": [("replacement_list", map_offsets)]},
        "function_variable_pairs.bz2": {"columns_to_map": ['src'], "output_path": "common_function_variable_pairs.json"},
        "call_seq.bz2": {"columns_to_map": ['src', 'dst'], "output_path": "common_call_seq.json"},

        "ast_nodes.bz2": {"columns_to_map": ['id', 'mentioned_in'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name']},
        "ast_edges.bz2": {"columns_to_map": ['target_node_id', 'source_node_id', 'mentioned_in'], "output_path": "common_edges.json"},
        "offsets.bz2": {"columns_to_map": ['node_id'], "output_path": "common_offsets.json", "columns_special": [("mentioned_in", map_offsets)]},
        "filecontent_with_package.bz2": {"columns_to_map": [], "output_path": "common_filecontent.json"},
        "name_mappings.bz2": {"columns_to_map": [], "output_path": "common_name_mappings.json"},
    }

    files_for_merging_with_ast = [
        "global_nodes.bz2", "global_edges.bz2", "ast_nodes.bz2", "ast_edges.bz2", "bodies.bz2", "function_variable_pairs.bz2",
        "call_seq.bz2", "offsets.bz2", "filecontent_with_package.bz2", "name_mappings.bz2"
    ]

    def __init__(self, *args, **kwargs):
        super(HierarchicalDatasetCreator, self).__init__(*args, **kwargs)

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
                ast_nodes = unpersist_if_present(join(env_path, "ast_nodes.bz2"))
                nodes = unpersist_if_present(join(env_path, "global_nodes.bz2"))

                if ast_nodes is None:
                    nodes = unpersist_if_present(join(env_path, "nodes.bz2"))
                    nodes_with_ast = unpersist_if_present(join(env_path, "nodes_with_ast.bz2"))

                    if nodes is None or nodes_with_ast is None:
                        continue

                    global_edge_types = set(special_mapping.keys()) | set(special_mapping.values())

                    edges_with_ast = unpersist_if_present(join(env_path, "edges_with_ast.bz2"))
                    edges = unpersist_if_present(join(env_path, "edges.bz2"))

                    ast_edges = edges_with_ast.query("type not in @global_edge_types", local_dict={"global_edge_types": global_edge_types})
                    ast_nodes = nodes_with_ast
                    persist(ast_edges, join(env_path, "ast_edges.bz2"))
                    persist(nodes_with_ast, join(env_path, "ast_nodes.bz2"))
                    persist(nodes, join(env_path, "global_nodes.bz2"))
                    persist(edges, join(env_path, "global_edges.bz2"))

                nodes_with_ast = nodes.append(ast_nodes) if ast_nodes is not None else nodes

                edges = bodies = call_seq = vars = edges_with_ast = offsets = name_mappings = filecontent = ast_nodes = ast_edges = None

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

    def read_all_local_files(self, env_path):
        files = {}

        for filename in self.files_for_merging_with_ast:
            file_ = self.read_mapped_local(
                env_path, filename, "local2global_with_ast.bz2", **self.merging_specification[filename]
            )
            if file_ is not None:
                files[filename] = file_

        if "name_mappings.bz2" not in files:
            return None

        if len(files) > 0:
            assert len(set(files["ast_edges.bz2"]["id"]).intersection(set(files["global_edges.bz2"]["id"]))) == 0  # should be disjoint
            num_global_edges = len(files["global_edges.bz2"])
            num_ast_edges = len(files["ast_edges.bz2"])
            files["global_edges.bz2"]["id"] = range(num_global_edges)
            files["ast_edges.bz2"]["id"] = range(num_global_edges, num_global_edges + num_ast_edges)

            files["ast_edges"] = SQLTable(files["ast_edges.bz2"], ":memory:", "ast_edges")
            files["ast_nodes"] = SQLTable(files["ast_nodes.bz2"], ":memory:", "ast_nodes")
            files["global_edges"] = SQLTable(files["global_edges.bz2"], ":memory:", "global_edges")
            files["global_nodes"] = SQLTable(files["global_nodes.bz2"], ":memory:", "global_nodes")
        else:
            files = None
        return files

    def get_mention_hierarchy_graph(self, files):
        mentions = files["ast_edges.bz2"][
            ['source_node_id', 'target_node_id', 'mentioned_in']
        ]
        assert any(mentions["mentioned_in"].isna()) is False
        mention_ids = set(mentions["mentioned_in"])
        mention_hierarchy_edges = pd.concat([
            mentions[["mentioned_in", "target_node_id"]].rename({"mentioned_in": "src", "target_node_id": "dst"},
                                                                axis=1).query("dst in @mentions",
                                                                              local_dict={"mentions": mention_ids}),
            mentions[["mentioned_in", "source_node_id"]].rename({"mentioned_in": "src", "source_node_id": "dst"},
                                                                axis=1).query("dst in @mentions",
                                                                              local_dict={"mentions": mention_ids})
        ])
        mention_hierarchy_edges.drop_duplicates(inplace=True)
        no_def_candidates = mention_hierarchy_edges.query("src == dst")
        mention_hierarchy_edges = mention_hierarchy_edges.query("src != dst")

        no_def_candidates = set(no_def_candidates["src"])
        no_def = no_def_candidates - (set(mention_hierarchy_edges["src"]) | set(mention_hierarchy_edges["dst"]))

        mention_hierarchy_graph = nx.from_edgelist(mention_hierarchy_edges.values, create_using=nx.DiGraph)

        root_nodes = set(mention_hierarchy_edges["src"]) - set(mention_hierarchy_edges["dst"])
        root_nodes.update(no_def)

        return mention_hierarchy_graph, root_nodes

    def useful_mappings(self, files):

        def get_global_to_local_ids(files):
            global_to_local_ids = dict()
            local_ids_to_global = dict()
            global_name2id = dict(
                zip(files["global_nodes.bz2"]["serialized_name"], files["global_nodes.bz2"]["id"]))
            ast_name2id = dict(zip(files["ast_nodes.bz2"]["serialized_name"], files["ast_nodes.bz2"]["id"]))

            for ast_name, global_name in zip(files["name_mappings.bz2"]["ast_name"],
                                             files["name_mappings.bz2"]["proper_names"]):
                global_to_local_ids[global_name2id[global_name]] = ast_name2id[ast_name]
                local_ids_to_global[ast_name2id[ast_name]] = global_name2id[global_name]
            return global_to_local_ids, local_ids_to_global

        mappings = {}

        mappings["global_node_2_type"] = dict(zip(files["global_nodes.bz2"]["id"], files["global_nodes.bz2"]["type"]))
        mappings["filecontent"] = dict(
            zip(files["filecontent_with_package.bz2"]["id"], files["filecontent_with_package.bz2"]["content"]))
        # mappings["bodies"] = dict(zip(files["bodies.bz2"]["id"], files["bodies.bz2"].to_dict(orient="records")))

        mappings["global_offsets"] = dict()
        for offset in chain(*files["offsets.bz2"]["mentioned_in"]):
            mappings["global_offsets"][offset[2]] = offset

        mappings["offsets"] = defaultdict(list)
        for ind, offset in files["offsets.bz2"].iterrows():
            for _, _, m in offset["mentioned_in"]:
                mappings["offsets"][m].append((offset["start"], offset["end"], offset["node_id"]))

        mappings["ast_node2file_id"] = dict(
            zip(files["ast_edges.bz2"]["source_node_id"], files["ast_edges.bz2"]["file_id"]))
        mappings["ast_node2file_id"].update(
            dict(zip(files["ast_edges.bz2"]["target_node_id"], files["ast_edges.bz2"]["file_id"])))

        global_to_local_ids, local_ids_to_global = get_global_to_local_ids(files)
        mappings["global_to_local_ids"] = global_to_local_ids
        mappings["local_ids_to_global"] = local_ids_to_global

        mappings["node2name"] = dict(zip(files["ast_nodes.bz2"]["id"], files["ast_nodes.bz2"]["serialized_name"]))

        return mappings

    def get_text_and_offsets(self, file_id, file_text, global_id, mention_id, global_type, files, mappings):
        offsets = None
        text = None

        def get_offsets_for_file(file_id):
            offsets = files["offsets.bz2"].query(f"file_id == {file_id}")[
                ["start", "end", "node_id"]].values.tolist()
            return offsets

        def get_span_for_class_or_method(mention_id):
            start = end = None
            if mention_id in mappings["global_offsets"]:
                start, end, _ = mappings["global_offsets"][mention_id]
            elif mappings["local_ids_to_global"].get(mention_id, None) in mappings["global_offsets"]:
                start, end, _ = mappings["global_offsets"][global_id]
            return start, end

        def get_offsets_for_class_or_method(mention_id, start, end):
            offsets = None
            global_id = mappings["local_ids_to_global"].get(mention_id, None)
            if global_id in mappings["offsets"]:
                offsets = mappings["offsets"][global_id]
            elif mention_id in mappings["offsets"]:
                offsets = mappings["offsets"][mention_id]

            if offsets is not None:
                offsets = [o for o in offsets if o[0] >= start and o[1] <= end]
                offsets = adjust_offsets2(offsets, -start)
                for s, _, _ in offsets:
                    assert s >= 0, "Offset became negative"

            return offsets

        if global_type == "module":
            offsets = get_offsets_for_file(file_id)
            text = file_text
        else:
            start, end = get_span_for_class_or_method(mention_id)
            if start is not None:
                text = file_text[start: end]
                offsets = get_offsets_for_class_or_method(mention_id, start, end)

        return text, offsets

    def write_mention(
            self, files, mappings, file_id, file_text, location_path, mention_id,
            mention_hierarchy_graph, nlp
    ):
        mention_path = location_path.joinpath(str(mention_id))
        if mention_id in mention_hierarchy_graph:
            nested_mentions = mention_hierarchy_graph[mention_id]
        else:
            nested_mentions = []

        global_id = mappings["local_ids_to_global"].get(mention_id, None)
        global_type = mappings["global_node_2_type"].get(global_id, None)

        metadata = {
            "global_id": global_id,
            "global_type": global_type,
        }

        if global_type is not None:

            text, offsets = self.get_text_and_offsets(
                file_id, file_text, global_id, mention_id, global_type, files, mappings
            )

            if text is not None:
                metadata["original_text"] = text
                metadata["original_offsets"] = offsets

                entry = extract_type_annotations(nlp, text, offsets, require_labels=False)
                if entry is not None:
                    metadata["normalized_text"] = entry["text"]
                    metadata["normalized_offsets"] = entry["replacements"]
                    metadata["type_annotations"] = entry["ents"]
                    if len(entry["cats"]) == 1:
                        metadata["returns"] = entry["cats"]
                    if len(entry["docstrings"]) == 1:
                        metadata["docstring"] = entry["docstrings"][0]

        edges_written = 0
        written_edge_ids = set()

        for m in nested_mentions:
            _edges_written, _written_edge_ids = self.write_mention(
                files, mappings, file_id, file_text, mention_path, m, mention_hierarchy_graph, nlp
            )
            edges_written += _edges_written
            written_edge_ids.update(_written_edge_ids)
            assert len(written_edge_ids) == edges_written

        def get_mention_nodes_and_edges(mention_id):
            # mention_edges = files["ast_edges.bz2"].query(f"mentioned_in == {mention_id}")
            mention_edges = files["ast_edges"].query(
                f"select * from ast_edges where mentioned_in = {mention_id}")
            mention_node_ids = set(mention_edges["source_node_id"]) | set(mention_edges["target_node_id"])
            # mention_nodes = files["ast_nodes.bz2"].query("id in @mention_ids", local_dict={"mention_ids": mention_node_ids})
            mention_nodes = files["ast_nodes"].query(
                f"select * from ast_nodes where id in ({','.join(map(str, mention_node_ids))})")
            # mention_global_nodes = files["global_nodes.bz2"].query("id in @mention_ids", local_dict={"mention_ids": mention_node_ids})
            mention_global_nodes = files["global_nodes"].query(
                f"select * from global_nodes where id in ({','.join(map(str, mention_node_ids))})")
            mention_nodes = mention_nodes.append(mention_global_nodes)
            mention_nodes.drop_duplicates(subset=["id", "type", "serialized_name"], inplace=True)
            return mention_nodes, mention_edges

        mention_nodes, mention_edges = get_mention_nodes_and_edges(mention_id)
        mention_path.mkdir(exist_ok=True, parents=True)

        write_mapping_to_json(metadata, mention_path.joinpath("metadata.json"))

        def filter_type_annotations(edges):
            edges = SQLTable(edges, ":memory:", "edges")
            # type_ann = edges.query("type == 'annotation_for' or type == 'annotation_for_rev' or type == 'returned_by' or type == 'returned_by_rev'")
            # no_type_ann = edges.query("type != 'annotation_for' and type != 'annotation_for_rev' and type != 'returned_by' and type != 'returned_by_rev'")
            type_ann = edges.query(
                "select * from edges where type = 'annotation_for' or type = 'annotation_for_rev' or type = 'returned_by' or type = 'returned_by_rev'")
            no_type_ann = edges.query(
                "select * from edges where type != 'annotation_for' and type != 'annotation_for_rev' and type != 'returned_by' and type != 'returned_by_rev'")

            type_ann = type_ann.query("type == 'annotation_for' or type == 'returned_by'")
            # type_ann["source_node_id"] = type_ann["source_node_id"].apply(node2name.get)
            # type_ann = type_ann.rename({"source_node_id": "dst", "target_node_id": "src"}, axis=1)
            return no_type_ann, type_ann

        def get_type_annotation_labels(type_ann_edges):
            type_ann_edges["source_node_id"] = type_ann_edges["source_node_id"].apply(mappings["node2name"].get)
            type_ann_edges = type_ann_edges.rename({"source_node_id": "dst", "target_node_id": "src"}, axis=1)
            return type_ann_edges[["src", "dst"]]

        if self.remove_type_annotations:
            # mention_edges, type_ann_edges = filter_type_annotations(mention_edges)
            mention_edges, type_ann_edges = mention_edges, None
        else:
            type_ann_edges = None

        edge_offsets = mention_edges[["id", "offset_start", "offset_end"]].dropna()
        node_strings = mention_nodes[["id", "string"]].dropna()

        if type_ann_edges is not None and len(type_ann_edges) > 0:
            persist(type_ann_edges[["id", "type", "source_node_id", "target_node_id"]],
                    mention_path.joinpath("type_ann_edges.parquet"))
            persist(get_type_annotation_labels(type_ann_edges), mention_path.joinpath("type_ann_labels.parquet"))
        if len(edge_offsets) > 0:
            persist(edge_offsets, mention_path.joinpath("edge_offsets.parquet"))
        if len(node_strings) > 0:
            persist(node_strings, mention_path.joinpath("node_strings.parquet"))

        persist(mention_edges[["id", "type", "source_node_id", "target_node_id"]],
                mention_path.joinpath("edges.parquet"))
        persist(mention_nodes[["id", "type", "serialized_name"]], mention_path.joinpath("nodes.parquet"))

        edges_written += len(mention_edges["id"])
        written_edge_ids.update(mention_edges["id"])

        if self.visualize is True and len(nested_mentions) == 0 and len(mention_edges) < 200:
            self.visualize_func(
                mention_nodes, mention_edges, mention_path.joinpath("visualization.pdf"), show_reverse=True
            )
            # if "normalized_text" in metadata:
            #     render_single(metadata["normalized_text"], metadata["normalized_offsets"], mention_path.joinpath("visualization.html"))

        return edges_written, written_edge_ids

    def assign_partitions(self, dataset_location):
        for package_path in tqdm(
                dataset_location.iterdir(), desc="Assigning partitions", leave=True,
                dynamic_ncols=True, total=len(self.environments)
        ):
            if not package_path.is_dir():
                continue

            def is_int(value):
                try:
                    int(value)
                    return True
                except:
                    return False

            def write_partition_to_metadata(path, partition):
                metadata_path = path.joinpath("metadata.json")
                if metadata_path.is_file():
                    metadata = read_mapping_from_json(metadata_path)
                else:
                    metadata = dict()
                metadata["partition"] = partition
                write_mapping_to_json(metadata, metadata_path)

            for file_path in package_path.iterdir():
                if not file_path.is_dir() or not is_int(file_path.name):
                    continue

                if random() > 0.8:
                    # train
                    write_partition_to_metadata(file_path, "train")
                else:
                    if random() > 0.5:
                        # test
                        write_partition_to_metadata(file_path, "test")
                    else:
                        # val
                        write_partition_to_metadata(file_path, "val")

    def create_dataset_structure(self, output_directory):
        nlp = create_tokenizer("spacy")

        # for ind, env_path in tqdm(
        #         enumerate(self.environments), desc="Creating dataset hierarchy", leave=True,
        #         dynamic_ncols=True, total=len(self.environments)
        # ):
        for ind, env_path in enumerate(self.environments):
            package_name = Path(env_path).name
            logging.info(f"Found {package_name}")
            package_path = output_directory.joinpath(package_name)
            if package_path.joinpath("nodes.parquet").is_file():
                logging.info(f"Hierarchy is already present")
                continue

            files = self.read_all_local_files(env_path)
            if files is None:
                logging.info(f"Cannot use package, check all required files")
                continue

            mention_hierarchy_graph, root_nodes = self.get_mention_hierarchy_graph(files)
            mappings = self.useful_mappings(files)

            edges_written = 0
            written_edge_ids = set()

            for root in tqdm(root_nodes, desc=f"Creating hierarchy for {package_name}", dynamic_ncols=True):
                file_id = mappings["ast_node2file_id"][root]
                file_id_path = package_path.joinpath(str(file_id))
                file_text = mappings["filecontent"][file_id]

                _edges_written, _written_edge_ids = self.write_mention(
                    files, mappings, file_id, file_text, file_id_path, root, mention_hierarchy_graph, nlp
                )
                edges_written += _edges_written
                written_edge_ids.update(_written_edge_ids)
                assert edges_written == len(written_edge_ids)

            remaining_edges = files["ast_edges.bz2"].query("id not in @written_edges", local_dict={"written_edges": written_edge_ids})
            assert len(remaining_edges) == 0

            # persist(files["global_edges.bz2"], package_path.joinpath("edges.parquet"))
            # persist(files["global_nodes.bz2"], package_path.joinpath("nodes.parquet"))
            persist(files["global_edges.bz2"], package_path.joinpath("edges.parquet"))
            persist(files["global_nodes.bz2"], package_path.joinpath("nodes.parquet"))

    def merge(self, output_directory):

        if self.extract:
            logging.info("Extracting...")
            self.do_extraction()

        # no_ast_path, with_ast_path = self.create_output_dirs(output_directory)

        # if not self.only_with_annotations:
        #     self.merge_graph_without_ast(no_ast_path)

        output_directory = Path(output_directory)

        self.create_dataset_structure(output_directory)
        self.assign_partitions(output_directory)


if __name__ == "__main__":

    args = DatasetCreatorArgumentParser().parse()

    if args.recompute_l2g:
        args.do_extraction = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

    dataset = HierarchicalDatasetCreator(
        args.indexed_environments, args.language, args.bpe_tokenizer, args.create_subword_instances,
        args.connect_subwords, args.only_with_annotations, args.do_extraction, args.visualize, args.track_offsets,
        args.remove_type_annotations, args.recompute_l2g
    )
    dataset.merge(args.output_directory)
