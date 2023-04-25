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
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping, node_types
from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_dataset import process_body as extract_type_annotations
from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.entity.entity_render import render_single


global_node_types = set(node_types.values())


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
            return [tuple(offset) for offset in offsets]

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
                # offsets = adjust_offsets2(offsets, -start)
                # for s, _, _ in offsets:
                #     assert s >= 0, "Offset became negative"

            return offsets

        enclosing_span = None
        if global_type == "module":
            offsets = get_offsets_for_file(file_id)
            text = file_text
            offsets = [offset for offset in offsets if offset[0] < len(text) and offset[1] < len(text)]
        else:
            start, end = get_span_for_class_or_method(mention_id)
            if start is not None:
                text = file_text[start: end]
                offsets = get_offsets_for_class_or_method(mention_id, start, end)
                enclosing_span = (start, end)

        return text, offsets, enclosing_span

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

        entry = {}
        edges_written = 0
        written_edge_ids = set()
        written_offsets = set()

        if global_type is None:
            return edges_written, written_edge_ids, written_offsets

        # offsets here have original spans
        # adjust them to the text later
        text, offsets, enclosing_span = self.get_text_and_offsets(
            file_id, file_text, global_id, mention_id, global_type, files, mappings
        )

        if text is None:
            return edges_written, written_edge_ids, written_offsets

        for m in nested_mentions:
            _edges_written, _written_edge_ids, _written_offsets = self.write_mention(
                files, mappings, file_id, file_text, mention_path, m, mention_hierarchy_graph, nlp
            )
            edges_written += _edges_written
            written_edge_ids.update(_written_edge_ids)
            written_offsets.update(_written_offsets)
            assert len(written_edge_ids) == edges_written

        offsets = list(set(offsets) - written_offsets)
        written_offsets.update(offsets)

        metadata["original_text"] = text
        metadata["original_offsets"] = offsets
        metadata["enclosing_span"] = enclosing_span

        def get_mention_nodes_and_edges(mention_id):
            mention_edges = files["ast_edges"].query(f"select * from ast_edges where mentioned_in = {mention_id}")
            mention_node_ids = set(mention_edges["source_node_id"]) | set(mention_edges["target_node_id"])
            mention_nodes = files["ast_nodes"].query(f"select * from ast_nodes where id in ({','.join(map(str, mention_node_ids))})")
            mention_global_nodes = files["global_nodes"].query(f"select * from global_nodes where id in ({','.join(map(str, mention_node_ids))})")
            mention_nodes = mention_nodes.append(mention_global_nodes)
            mention_nodes.drop_duplicates(subset=["id", "type", "serialized_name"], inplace=True)
            return mention_nodes, mention_edges

        mention_nodes, mention_edges = get_mention_nodes_and_edges(mention_id)

        def get_edge_offsets(edge_offsets, enclosing_span):
            if enclosing_span is not None:
                for edge in edge_offsets:
                    assert edge["offset_start"] >= enclosing_span[0] and edge["offset_end"] <= enclosing_span[1]
            offsets = []
            for edge in edge_offsets:
                offsets.append((edge["offset_start"], edge["offset_end"], f"#edge_{edge['id']}"))
            return offsets

        try:
            edge_offsets = get_edge_offsets(
                mention_edges[["id", "offset_start", "offset_end"]]\
                    .dropna()\
                    .astype({"offset_start": "int64", "offset_end": "int64"}).to_dict(orient="records"),
                enclosing_span
            )
            metadata["original_edge_offsets"] = edge_offsets
        except AssertionError:
            # when getting edges for some mentions, they seem to be out of enclosing_span
            # the reason for this is unclear. skip such cases for now
            return edges_written, written_edge_ids, written_offsets

        mention_path.mkdir(exist_ok=True, parents=True)

        def map_nodes_to_embeddable_names(nodes):
            nodes = nodes.to_dict(orient="records")
            node_to_emb_name = {}
            for node in nodes:
                if node["type"] in {
                    "ctx", "type_node", "type_annotation", "Name", "#attr#", "#keyword#", "Op", "Constant", "JoinedStr",
                    "CtlFlow", "astliteral", "subword", "mention"
                } or node["type"] in global_node_types:
                    node_to_emb_name[node["id"]] = node["serialized_name"].split("@")[0]
                else:
                    node_to_emb_name[node["id"]] = node["type"]
            return node_to_emb_name

        emb_names = map_nodes_to_embeddable_names(mention_nodes)

        if enclosing_span is not None:
            entry_offsets = adjust_offsets2(offsets + edge_offsets, -enclosing_span[0])
        else:
            entry_offsets = offsets + edge_offsets

        entry_offsets = [offset for offset in entry_offsets if offset[0] < len(text) and offset[1] < len(text)]

        entry_ = extract_type_annotations(nlp, text, entry_offsets, require_labels=False, remove_docstring=True)
        if entry_ is not None:
            entry["enclosing_span"] = enclosing_span
            entry["normalized_text"] = entry_["text"]
            entry["normalized_node_offsets"] = entry_["replacements"]
            entry["normalized_edge_offsets"] = [(e[0], e[1], int(e[2][6:])) for e in entry_["ents"] if e[2].startswith("#edge_")]
            entry["type_annotations"] = [e for e in entry_["ents"] if not e[2].startswith("#edge_")]
            entry["node_names"] = emb_names
            if len(entry_["cats"]) == 1:
                entry["returns"] = entry_["cats"]
            if len(entry_["docstrings"]) == 1:
                entry["docstring"] = entry_["docstrings"][0]
        else:
            return edges_written, written_edge_ids, written_offsets

        write_mapping_to_json(metadata, mention_path.joinpath("metadata.json"))

        entry["edge_id"] = mention_edges["id"].tolist()
        entry["src_id"] = mention_edges["source_node_id"].tolist()
        entry["dst_id"] = mention_edges["target_node_id"].tolist()
        entry["edge_type"] = mention_edges["type"].tolist()

        write_mapping_to_json(entry, mention_path.joinpath("entry.json"))

        write_mapping_to_json(
            mention_nodes[["id", "type", "serialized_name"]].rename({"serialized_name": "name"}, axis=1).to_dict(orient="records"),
            mention_path.joinpath("nodes.json")
        )

        edges_written += len(mention_edges["id"])
        written_edge_ids.update(mention_edges["id"])

        if self.visualize is True and len(nested_mentions) == 0 and len(mention_edges) < 200:
            self.visualize_func(
                mention_nodes, mention_edges, mention_path.joinpath("visualization.pdf"), show_reverse=True
            )
            render_single(entry["normalized_text"], entry["normalized_node_offsets"], mention_path.joinpath("visualization.html"))

        return edges_written, written_edge_ids, written_offsets

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

        for ind, env_path in enumerate(self.environments):
            package_name = Path(env_path).name
            logging.info(f"Found {package_name}")
            package_path = output_directory.joinpath(package_name)
            if package_path.joinpath("nodes.json").is_file():
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

                _edges_written, _written_edge_ids, _written_offsets = self.write_mention(
                    files, mappings, file_id, file_text, file_id_path, root, mention_hierarchy_graph, nlp
                )
                edges_written += _edges_written
                written_edge_ids.update(_written_edge_ids)
                assert edges_written == len(written_edge_ids)

            remaining_edges = files["ast_edges.bz2"].query("id not in @written_edges", local_dict={"written_edges": written_edge_ids})
            logging.warning(f"{len(remaining_edges)} edges skipped")

            global_nodes = files["global_nodes.bz2"][["id", "type", "serialized_name"]].rename(
                {"serialized_name": "name"}, axis=1)

            edges_json = defaultdict(list)
            for row_ind, edge in files["global_edges.bz2"].iterrows():
                edges_json["edge_id"].append(edge["id"])
                edges_json["src_id"].append(edge["source_node_id"])
                edges_json["dst_id"].append(edge["target_node_id"])
                edges_json["edge_type"].append(edge["type"])
            edges_json["node_names"] = dict(zip(global_nodes["id"], global_nodes["name"]))
            write_mapping_to_json(edges_json, package_path.joinpath("entry.json"))

            write_mapping_to_json(
                global_nodes.to_dict(
                    orient="records"),
                package_path.joinpath("nodes.json")
            )

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
