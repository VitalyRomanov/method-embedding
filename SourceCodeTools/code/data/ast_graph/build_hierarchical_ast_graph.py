import logging
import os.path
import shutil
from ast import literal_eval
from collections import defaultdict
from os.path import join
from pathlib import Path
from random import random

import networkx as nx
import numpy as np
import pandas as pd

from tqdm import tqdm

from SourceCodeTools.cli_arguments import AstDatasetCreatorArgumentParser
from SourceCodeTools.code.common import read_nodes, SQLTable
from SourceCodeTools.code.data.AbstractDatasetCreator import AbstractDatasetCreator
from SourceCodeTools.code.data.ast_graph.build_ast_graph import build_ast_only_graph2
from SourceCodeTools.code.data.ast_graph.extract_node_names import extract_node_names
from SourceCodeTools.code.data.ast_graph.filter_type_edges import filter_type_edges_with_chunks
from SourceCodeTools.code.data.file_utils import persist, unpersist, unpersist_if_present, write_mapping_to_json, \
    read_mapping_from_json
from SourceCodeTools.code.data.ast_graph.draw_graph import visualize
from SourceCodeTools.code.annotator_utils import map_offsets
from SourceCodeTools.code.data.ast_graph.local2global import get_local2global
from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.entity.entity_render import render_single


class HierarchicalAstDatasetCreator(AbstractDatasetCreator):

    merging_specification = {
        "source_graph_bodies.bz2": {"columns_to_map": ['id'], "output_path": "common_source_graph_bodies.json", "columns_special": [("replacement_list", map_offsets)]},
        "function_variable_pairs.bz2": {"columns_to_map": ['src'], "output_path": "common_function_variable_pairs.json"},
        "call_seq.bz2": {"columns_to_map": ['src', 'dst'], "output_path": "common_call_seq.json"},

        "nodes_with_ast.bz2": {"columns_to_map": ['id', 'mentioned_in'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name']},
        "edges_with_ast.bz2": {"columns_to_map": ['target_node_id', 'source_node_id', 'mentioned_in'], "output_path": "common_edges.json"},
        "offsets.bz2": {"columns_to_map": ['node_id', 'mentioned_in'], "output_path": "common_offsets.json"},
        "filecontent_with_package.bz2": {"columns_to_map": [], "output_path": "common_filecontent.json"},
        "name_mappings.bz2": {"columns_to_map": [], "output_path": "common_name_mappings.json"},
    }

    files_for_merging_with_ast = [
        "nodes_with_ast.bz2", "edges_with_ast.bz2",
        "offsets.bz2", "filecontent_with_package.bz2"
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
            self, path, lang, bpe_tokenizer, create_subword_instances, connect_subwords, only_with_annotations,
            do_extraction=False, visualize=False, track_offsets=False, remove_type_annotations=False,
            recompute_l2g=False, chunksize=1000, keep_frac=1.0, seed=None, create_mention_instances=True,
            graph_format_version="v3.5"
    ):
        self.chunksize = chunksize
        self.keep_frac = keep_frac
        self.seed = seed
        self.graph_format_version = graph_format_version
        self.create_mention_instances = create_mention_instances
        super().__init__(
            path, lang, bpe_tokenizer, create_subword_instances, connect_subwords, only_with_annotations,
            do_extraction, visualize, track_offsets, remove_type_annotations, recompute_l2g
        )

    def __del__(self):
        # TODO use /tmp and add flag for overriding temp folder location
        if hasattr(self, "temp_path") and os.path.isdir(self.temp_path):
            shutil.rmtree(self.temp_path)
        # pass

    def _prepare_environments(self):
        from time import time_ns
        dataset_location = os.path.dirname(self.path)
        temp_path = os.path.join(dataset_location, f"temp_graph_builder_{time_ns()}")
        self.temp_path = temp_path
        if os.path.isdir(temp_path):
            raise FileExistsError(f"Directory exists: {temp_path}")
        os.makedirs(temp_path, exist_ok=True)

        rnd_state = np.random.RandomState(self.seed)

        for ind, chunk in enumerate(pd.read_csv(self.path, chunksize=self.chunksize)):
            chunk = chunk.sample(frac=self.keep_frac, random_state=rnd_state)
            chunk_path = os.path.join(temp_path, f"chunk_{ind}")
            os.mkdir(chunk_path)
            persist(chunk, os.path.join(chunk_path, "source_code.bz2"))

        paths = (os.path.join(temp_path, dir) for dir in os.listdir(temp_path))
        self.environments = sorted(list(filter(lambda path: os.path.isdir(path), paths)), key=lambda x: x.lower())

    @staticmethod
    def extract_node_names(nodes_path, min_count):
        logging.info("Extract node names")
        return extract_node_names(read_nodes(nodes_path), min_count=min_count)

    def filter_type_edges(self, nodes_path, edges_path):
        logging.info("Filter type edges")
        filter_type_edges_with_chunks(nodes_path, edges_path, kwarg_fn=self.get_writing_mode)

    def do_extraction(self):
        global_nodes_with_ast = set()

        for env_path in self.environments:
            logging.info(f"Found {os.path.basename(env_path)}")

            if not self.recompute_l2g:

                source_code = unpersist(join(env_path, "source_code.bz2"))

                nodes_with_ast, edges_with_ast, offsets = build_ast_only_graph2(
                    zip(source_code["package"], source_code["id"], source_code["filecontent"]), self.bpe_tokenizer,
                    create_subword_instances=self.create_subword_instances, connect_subwords=self.connect_subwords,
                    lang=self.lang, track_offsets=self.track_offsets, mention_instances=self.create_mention_instances,
                    graph_variety=self.graph_format_version
                )

            else:
                nodes_with_ast = unpersist_if_present(join(env_path, "nodes_with_ast.bz2"))

                if nodes_with_ast is None:
                    continue

                edges_with_ast = offsets = source_code = None

            local2global_with_ast = get_local2global(
                global_nodes=global_nodes_with_ast, local_nodes=nodes_with_ast
            )

            global_nodes_with_ast.update(local2global_with_ast["global_id"])

            self.write_type_annotation_flag(edges_with_ast, env_path)

            self.write_local(
                env_path,
                local2global_with_ast=local2global_with_ast,
                nodes_with_ast=nodes_with_ast, edges_with_ast=edges_with_ast, offsets=offsets,
                filecontent_with_package=source_code,
            )

        self.compact_mapping_for_l2g(global_nodes_with_ast, "local2global_with_ast.bz2")

    def create_output_dirs(self, output_path):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        with_ast_path = join(output_path, "with_ast")

        if not os.path.isdir(with_ast_path):
            os.mkdir(with_ast_path)

        return with_ast_path

    def read_all_local_files(self, env_path):
        files = {}

        for filename in self.files_for_merging_with_ast:
            file_ = self.read_mapped_local(
                env_path, filename, "local2global_with_ast.bz2", **self.merging_specification[filename]
            )
            if file_ is not None:
                files[filename] = file_

        if len(files) > 0:
            files["ast_edges"] = SQLTable(files["edges_with_ast.bz2"], ":memory:", "ast_edges")
            files["ast_nodes"] = SQLTable(files["nodes_with_ast.bz2"], ":memory:", "ast_nodes")
        else:
            files = None
        return files

    def get_mention_hierarchy_graph(self, files):
        mentions = files["edges_with_ast.bz2"][
            ['source_node_id', 'target_node_id', 'mentioned_in']
        ]
        mentions.dropna(axis=0, inplace=True)
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
        mappings = {}

        mappings["filecontent"] = dict(
            zip(files["filecontent_with_package.bz2"]["id"], files["filecontent_with_package.bz2"]["filecontent"]))

        mappings["offsets"] = defaultdict(list)
        for ind, offset in files["offsets.bz2"].iterrows():
            mappings["offsets"][offset["mentioned_in"]].append((offset["start"], offset["end"], offset["node_id"]))

        mappings["ast_node2file_id"] = dict(
            zip(files["edges_with_ast.bz2"]["source_node_id"], files["edges_with_ast.bz2"]["file_id"]))
        mappings["ast_node2file_id"].update(
            dict(zip(files["edges_with_ast.bz2"]["target_node_id"], files["edges_with_ast.bz2"]["file_id"])))

        mappings["node2name"] = dict(zip(files["nodes_with_ast.bz2"]["id"], files["nodes_with_ast.bz2"]["serialized_name"]))

        return mappings

    def get_text_and_offsets(self, file_id, file_text, mention_id, files, mappings):
        if mappings["node2name"][mention_id].startswith("Module"):
            text = mappings["filecontent"][file_id]
        else:
            text = None
        offsets = mappings["offsets"][mention_id]
        enclosing_span = None
        return text, offsets, enclosing_span

    def write_mention(
            self, files, mappings, file_id, file_text, location_path, mention_id,
            mention_hierarchy_graph, nlp, new_edge_id
    ):
        mention_path = location_path.joinpath(str(mention_id))
        if mention_id in mention_hierarchy_graph:
            nested_mentions = mention_hierarchy_graph[mention_id]
        else:
            nested_mentions = []

        entry = {}
        edges_written = 0
        written_edge_ids = set()
        written_offsets = set()

        # offsets here have original spans
        # adjust them to the text later
        text, offsets, enclosing_span = self.get_text_and_offsets(
            file_id, file_text, mention_id, files, mappings
        )

        for m in nested_mentions:
            _edges_written, _written_edge_ids, _written_offsets = self.write_mention(
                files, mappings, file_id, file_text, mention_path, m, mention_hierarchy_graph, nlp, new_edge_id
            )
            edges_written += _edges_written
            written_edge_ids.update(_written_edge_ids)
            written_offsets.update(_written_offsets)
            new_edge_id = max(new_edge_id, (max(written_edge_ids) + 1) if len(written_edge_ids) > 0 else 0)
            assert len(written_edge_ids) == edges_written

        offsets = list(set(offsets) - written_offsets)
        written_offsets.update(offsets)

        entry["normalized_text"] = text
        entry["enclosing_span"] = enclosing_span
        entry["normalized_node_offsets"] = offsets

        def get_mention_nodes_and_edges(mention_id):
            mention_edges = files["ast_edges"].query(f"select * from ast_edges where mentioned_in = {mention_id}")
            mention_node_ids = set(mention_edges["source_node_id"]) | set(mention_edges["target_node_id"])
            mention_node_ids_str = f"{','.join(map(str,mention_node_ids))}"
            mention_edges = mention_edges.append(
                files["ast_edges"].query(f"select * from ast_edges where mentioned_in is null and (source_node_id in ({mention_node_ids_str}) or target_node_id in ({mention_node_ids_str}))")
            )
            mention_node_ids = set(mention_edges["source_node_id"]) | set(mention_edges["target_node_id"])
            mention_nodes = files["ast_nodes"].query(f"select * from ast_nodes where id in ({','.join(map(str, mention_node_ids))})")
            mention_nodes.drop_duplicates(subset=["id", "type", "serialized_name"], inplace=True)
            return mention_nodes, mention_edges

        mention_nodes, mention_edges = get_mention_nodes_and_edges(mention_id)
        mention_edges["id"] = range(new_edge_id, new_edge_id + len(mention_edges))

        def get_edge_offsets(edge_offsets, enclosing_span):
            if enclosing_span is not None:
                for edge in edge_offsets:
                    assert edge["offset_start"] >= enclosing_span[0] and edge["offset_end"] <= enclosing_span[1]
            offsets = []
            for edge in edge_offsets:
                offsets.append((edge["offset_start"], edge["offset_end"], edge['id']))
            return offsets

        try:
            edge_offsets = get_edge_offsets(
                mention_edges[["id", "offset_start", "offset_end"]]\
                    .dropna()\
                    .astype({"offset_start": "int64", "offset_end": "int64"}).to_dict(orient="records"),
                enclosing_span
            )
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
                }:
                    node_to_emb_name[node["id"]] = node["serialized_name"].split("@")[0]
                else:
                    node_to_emb_name[node["id"]] = node["type"]
            return node_to_emb_name

        emb_names = map_nodes_to_embeddable_names(mention_nodes)

        entry["normalized_edge_offsets"] = edge_offsets
        entry["node_names"] = emb_names

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

            files = self.read_all_local_files(env_path)
            if files is None:
                logging.info(f"Cannot use package, check all required files")
                continue

            mention_hierarchy_graph, root_nodes = self.get_mention_hierarchy_graph(files)
            mappings = self.useful_mappings(files)

            edges_written = 0
            written_edge_ids = set()

            partition_proper_name = {
                "train": "train",
                "eval": "test",
                "dev": "val"
            }

            for root in tqdm(root_nodes, desc=f"Creating hierarchy for {package_name}", dynamic_ncols=True):
                package_name = files["ast_edges"].query(f"select * from ast_edges where target_node_id = {root}")["package"][0]
                file_id = mappings["ast_node2file_id"][root]
                package_path = output_directory.joinpath(f"{package_name}")
                file_id_path = package_path.joinpath(str(file_id))
                metadata_path = file_id_path.joinpath("metadata.json")
                if metadata_path.is_file():
                    logging.info("File already processed")
                    continue
                file_text = mappings["filecontent"][file_id]

                _edges_written, _written_edge_ids, _written_offsets = self.write_mention(
                    files, mappings, file_id, file_text, file_id_path, root, mention_hierarchy_graph, nlp,
                    new_edge_id=(max(written_edge_ids) + 1) if len(written_edge_ids) > 0 else 0
                )
                edges_written += _edges_written
                written_edge_ids.update(_written_edge_ids)
                metadata = files["filecontent_with_package.bz2"].query(f"id == {file_id}")[
                    ["fn_path", "comment", "label", "partition", "original_function", "original_span", "misuse_span"]
                ].to_dict(orient="records")[0]
                for key in metadata:
                    try:
                        metadata[key] = literal_eval(metadata[key])
                    except:
                        pass
                assert len(metadata) > 0
                metadata["partition"] = partition_proper_name[metadata["partition"]]
                write_mapping_to_json(metadata, metadata_path)

    def merge(self, output_directory):

        if self.extract:
            logging.info("Extracting...")
            self.do_extraction()

        output_directory = Path(output_directory)

        self.create_dataset_structure(output_directory)

    def visualize_func(self, nodes, edges, output_path, **kwargs):
        visualize(nodes, edges, output_path, **kwargs)


if __name__ == "__main__":

    args = AstDatasetCreatorArgumentParser().parse()

    if args.recompute_l2g:
        args.do_extraction = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

    dataset = HierarchicalAstDatasetCreator(
        args.source_code, args.language, args.bpe_tokenizer, args.create_subword_instances,
        args.connect_subwords, args.only_with_annotations, args.do_extraction, args.visualize, args.track_offsets,
        args.remove_type_annotations, args.recompute_l2g, args.chunksize, args.keep_frac, args.seed,
        args.use_mention_instances, args.graph_format_version
    )
    dataset.merge(args.output_directory)
