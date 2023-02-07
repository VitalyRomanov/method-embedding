import hashlib
import logging
import os.path
import shutil
import sys
from os.path import join

import numpy as np
import pandas as pd

from tqdm import tqdm

from SourceCodeTools.cli_arguments import AstDatasetCreatorArguments
from SourceCodeTools.code.ast import has_valid_syntax
from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.AbstractDatasetCreator import AbstractDatasetCreator
from SourceCodeTools.code.data.ast_graph.extract_node_names import extract_node_names
from SourceCodeTools.code.data.ast_graph.filter_type_edges import filter_type_edges, filter_type_edges_with_chunks
from SourceCodeTools.code.data.file_utils import persist, unpersist, unpersist_if_present
from SourceCodeTools.code.data.ast_graph.draw_graph import visualize
from SourceCodeTools.code.ast.python_ast2 import AstGraphGenerator, GNode, PythonSharedNodes
from SourceCodeTools.code.annotator_utils import adjust_offsets2, map_offsets, to_offsets, get_cum_lens
from SourceCodeTools.code.data.ast_graph.local2global import get_local2global
from SourceCodeTools.nlp.string_tools import get_byte_to_char_map


class MentionTokenizer:
    def __init__(self, bpe_tokenizer_path, create_subword_instances, connect_subwords):
        from SourceCodeTools.nlp.embed.bpe import make_tokenizer
        from SourceCodeTools.nlp.embed.bpe import load_bpe_model

        self.bpe = make_tokenizer(load_bpe_model(bpe_tokenizer_path)) \
            if bpe_tokenizer_path else None
        self.create_subword_instances = create_subword_instances
        self.connect_subwords = connect_subwords

    def replace_mentions_with_subwords(self, edges):
        """
        Process edges and tokenize certain node types
        :param edges: List of edges
        :return: List of edges, including new edges for subword tokenization
        """

        if self.create_subword_instances:
            def produce_subw_edges(subwords, dst):
                return self.produce_subword_edges_with_instances(subwords, dst)
        else:
            def produce_subw_edges(subwords, dst):
                return self.produce_subword_edges(subwords, dst, self.connect_subwords)

        new_edges = []
        for edge in edges:

            if edge['type'] == "local_mention":

                dst = edge['dst']

                if self.bpe is not None:
                    if hasattr(dst, "name_scope") and dst.name_scope == "local":
                        # TODO
                        # this rule seems to be irrelevant now
                        subwords = self.bpe(dst.name.split("@")[0])
                    else:
                        subwords = self.bpe(edge['src'].name)

                    new_edges.extend(produce_subw_edges(subwords, dst))
                else:
                    new_edges.append(edge)

            elif self.bpe is not None and edge["type"] == "__global_name":
                # should not have global edges
                # subwords = self.bpe(edge['src'].name)
                # new_edges.extend(produce_subw_edges(subwords, edge['dst']))
                pass
            elif self.bpe is not None and edge['src'].type in PythonSharedNodes.tokenizable_types_and_annotations:
                new_edges.append(edge)
                if edge['type'] != "global_mention_rev":
                    # should not have global edges here
                    pass

                dst = edge['src']
                subwords = self.bpe(dst.name)
                new_edges.extend(produce_subw_edges(subwords, dst))
            else:
                new_edges.append(edge)

        return new_edges

    @staticmethod
    def connect_prev_next_subwords(edges, current, prev_subw, next_subw):
        if next_subw is not None:
            edges.append({
                'src': current,
                'dst': next_subw,
                'type': 'next_subword',
                'offsets': None
            })
        if prev_subw is not None:
            edges.append({
                'src': current,
                'dst': prev_subw,
                'type': 'prev_subword',
                'offsets': None
            })

    def produce_subword_edges(self, subwords, dst, connect_subwords=False):
        new_edges = []

        subwords = list(map(lambda x: GNode(name=x, type="subword"), subwords))
        for ind, subword in enumerate(subwords):
            new_edges.append({
                'src': subword,
                'dst': dst,
                'type': 'subword',
                'offsets': None
            })
            if connect_subwords:
                self.connect_prev_next_subwords(new_edges, subword, subwords[ind - 1] if ind > 0 else None,
                                                subwords[ind + 1] if ind < len(subwords) - 1 else None)
        return new_edges

    def produce_subword_edges_with_instances(self, subwords, dst, connect_subwords=True):
        new_edges = []

        subwords = list(map(lambda x: GNode(name=x, type="subword"), subwords))
        instances = list(map(lambda x: GNode(name=x.name + "@" + dst.name, type="subword_instance"), subwords))
        for ind, subword in enumerate(subwords):
            subword_instance = instances[ind]
            new_edges.append({
                'src': subword,
                'dst': subword_instance,
                'type': 'subword_instance',
                'offsets': None
            })
            new_edges.append({
                'src': subword_instance,
                'dst': dst,
                'type': 'subword',
                'offsets': None
            })
            if connect_subwords:
                self.connect_prev_next_subwords(new_edges, subword_instance, instances[ind - 1] if ind > 0 else None,
                                                instances[ind + 1] if ind < len(instances) - 1 else None)
        return new_edges


class NodeIdResolver:
    def __init__(self):
        self.node_ids = {}
        self.new_nodes = []
        self.stashed_nodes = []

        self._resolver_cache = dict()

    def stash_new_nodes(self):
        """
        Put new nodes into temporary storage.
        :return: Nothing
        """
        self.stashed_nodes.extend(self.new_nodes)
        self.new_nodes = []

    def get_node_id(self, type_name):
        return hashlib.md5(type_name.encode('utf-8')).hexdigest()

    def resolve_node_id(self, node, **kwargs):
        """
        Resolve node id from name and type, create new node is no nodes like that found.
        :param node: node
        :param kwargs:
        :return: updated node (return object with the save reference as input)
        """
        if not hasattr(node, "id"):
            node_repr = f"{node.name.strip()}_{node.type.strip()}"

            if node_repr in self.node_ids:
                node_id = self.node_ids[node_repr]
                node.setprop("id", node_id)
            else:
                new_id = self.get_node_id(node_repr)
                self.node_ids[node_repr] = new_id

                if not PythonSharedNodes.is_shared(node) and not node.name == "unresolved_name":
                    assert "0x" in node.name

                self.new_nodes.append(
                    {
                        "id": new_id,
                        "type": node.type,
                        "serialized_name": node.name,
                        "mentioned_in": pd.NA,
                        "string": node.string
                    }
                )
                if hasattr(node, "scope"):
                    self.resolve_node_id(node.scope)
                    self.new_nodes[-1]["mentioned_in"] = node.scope.id
                node.setprop("id", new_id)
        return node

    def prepare_for_write(self, from_stashed=False):
        nodes = self.new_nodes_for_write(from_stashed)[
            ['id', 'type', 'serialized_name', 'mentioned_in', 'string']
        ]

        return nodes

    def new_nodes_for_write(self, from_stashed=False):

        new_nodes = pd.DataFrame(self.new_nodes if not from_stashed else self.stashed_nodes)
        if len(new_nodes) == 0:
            return None

        new_nodes = new_nodes[
            ['id', 'type', 'serialized_name', 'mentioned_in', 'string']
        ].astype({"mentioned_in": "string", "id": "string"})

        return new_nodes

    def adjust_ast_node_types(self, mapping, from_stashed=False):
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        for node in nodes:
            node["type"] = mapping.get(node["type"], node["type"])

    def drop_nodes(self, node_ids_to_drop, from_stashed=False):
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        position = 0
        while position < len(nodes):
            if nodes[position]["id"] in node_ids_to_drop:
                nodes.pop(position)
            else:
                position += 1

    def map_mentioned_in_to_global(self, mapping, from_stashed=False):
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        for node in nodes:
            node["mentioned_in"] = mapping.get(node["mentioned_in"], node["mentioned_in"])


def get_ast_generator_class(base_class):
    class AstProcessor(base_class):
        def get_edges(self, as_dataframe=True):
            edges = []
            all_edges, top_node_name = self.parse(self.root)
            edges.extend(all_edges)

            if as_dataframe:
                df = pd.DataFrame(edges)
                df = df.astype({col: "Int32" for col in df.columns if col not in {"src", "dst", "type"}})

                body = "\n".join(self.source)
                cum_lens = get_cum_lens(body, as_bytes=True)
                byte2char = get_byte_to_char_map(body)

                def format_offsets(edges: pd.DataFrame):
                    edges["start_line__end_line__start_column__end_column"] = list(
                        zip(edges["line"], edges["end_line"], edges["col_offset"], edges["end_col_offset"])
                    )

                    def into_offset(range):
                        try:
                            return to_offsets(body, [(*range, None)], cum_lens=cum_lens, b2c=byte2char, as_bytes=True)[-1][:2]
                        except:
                            return None

                    edges["offsets"] = edges["start_line__end_line__start_column__end_column"].apply(into_offset)
                    edges.drop(
                        axis=1,
                        labels=[
                            "start_line__end_line__start_column__end_column",
                            "line",
                            "end_line",
                            "col_offset",
                            "end_col_offset"
                        ], inplace=True
                    )

                format_offsets(df)
                return df
            else:
                body = "\n".join(self.source)
                cum_lens = get_cum_lens(body, as_bytes=True)
                byte2char = get_byte_to_char_map(body)

                def format_offsets(edge):
                    def into_offset(range):
                        try:
                            return to_offsets(body, [(*range, None)], cum_lens=cum_lens, b2c=byte2char, as_bytes=True)[-1][:2]
                        except:
                            return None

                    if "line" in edge:
                        edge["offsets"] = into_offset(
                            (edge["line"], edge["end_line"], edge["col_offset"], edge["end_col_offset"])
                        )
                        edge.pop("line")
                        edge.pop("end_line")
                        edge.pop("col_offset")
                        edge.pop("end_col_offset")
                    else:
                        edge["offsets"] = None
                    if "var_line" in edge:
                        edge["var_offsets"] = into_offset(
                            (edge["var_line"], edge["var_end_line"], edge["var_col_offset"], edge["var_end_col_offset"])
                        )
                        edge.pop("var_line")
                        edge.pop("var_end_line")
                        edge.pop("var_col_offset")
                        edge.pop("var_end_col_offset")

                for edge in edges:
                    format_offsets(edge)

                return edges

    return AstProcessor


def standardize_new_edges(edges, node_resolver, mention_tokenizer):
    """
    Tokenize relevant node names, assign id to every node, collapse edge representation to id-based
    :param edges: list of edges
    :param node_resolver: helper class that tracks node ids
    :param mention_tokenizer: helper class that performs tokenization of relevant nodes
    :return:
    """

    edges = mention_tokenizer.replace_mentions_with_subwords(edges)

    resolve_node_id = lambda node: node_resolver.resolve_node_id(node)
    extract_id = lambda node: node.id

    for edge in edges:
        edge["src"] = resolve_node_id(edge["src"])
        edge["dst"] = resolve_node_id(edge["dst"])
        if "scope" in edge:
            edge["scope"] = resolve_node_id(edge["scope"])

    for edge in edges:
        edge["src"] = extract_id(edge["src"])
        edge["dst"] = extract_id(edge["dst"])
        if "scope" in edge:
            edge["scope"] = extract_id(edge["scope"])
        else:
            edge["scope"] = pd.NA
        edge["file_id"] = pd.NA

    return edges


def process_code_without_index(
        source_code, node_resolver, mention_tokenizer, track_offsets=False, mention_instances=False,
        reverse_edges=True, base_class=None
):
    if base_class is None:
        base_class = AstGraphGenerator

    AstGeneratorClass = get_ast_generator_class(base_class)

    try:
        ast_processor = AstGeneratorClass(
            source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances
        )
    except:
        return None, None, None
    try: # TODO recursion error does not appear consistently. The issue is probably with library versions...
        edges = ast_processor.get_edges(as_dataframe=False)
    except RecursionError:
        return None, None, None

    if len(edges) == 0:
        return None, None, None

    # tokenize names, replace nodes with their ids
    edges = standardize_new_edges(edges, node_resolver, mention_tokenizer)

    if track_offsets:
        def get_valid_offsets(edges):
            """
            :param edges: Dictionary that represents edge. Information is tored in edges but is related to source node
            :return: Information about location of this edge (offset) in the source file in fromat (start, end, node_id)
            """
            return [(edge["offsets"][0], edge["offsets"][1], (edge["src"], edge["dst"], edge["type"]), edge["scope"]) for edge in edges
                    if edge["offsets"] is not None]

        def get_node_offsets(offsets):
            return [(offset[0], offset[1], offset[2][0], offset[3]) for offset in offsets]

        def offsets_to_edge_mapping(offsets):
            return {offset[2]: (offset[0], offset[1]) for offset in offsets}

        def attach_offsets_to_edges(edges, offsets_edge_mapping):
            for edge in edges:
                repr = (edge["src"], edge["dst"], edge["type"])
                if repr in offsets_edge_mapping:
                    offset = offsets_edge_mapping[repr]
                    edge["offset_start"] = offset[0]
                    edge["offset_end"] = offset[1]

        # recover ast offsets for the current file
        valid_offsets = get_valid_offsets(edges)
        ast_offsets = get_node_offsets(valid_offsets)
        attach_offsets_to_edges(edges, offsets_to_edge_mapping(valid_offsets))
    else:
        ast_offsets = None

    return edges, ast_offsets


def ast_graph_for_single_example(
        source_code, bpe_tokenizer_path, create_subword_instances=False, connect_subwords=False, track_offsets=False,
        mention_instances=False, reverse_edges=True, ast_generator_base_class=None
):
    node_resolver = NodeIdResolver()
    mention_tokenizer = MentionTokenizer(bpe_tokenizer_path, create_subword_instances, connect_subwords)
    all_ast_edges = []
    all_offsets = []

    source_code_ = source_code.lstrip()
    initial_strip = source_code[:len(source_code) - len(source_code_)]

    if not has_valid_syntax(source_code):
        raise SyntaxError()

    edges, ast_offsets = process_code_without_index(
        source_code, node_resolver, mention_tokenizer, track_offsets=track_offsets, base_class=ast_generator_base_class,
        mention_instances=mention_instances, reverse_edges=reverse_edges
    )

    if ast_offsets is not None:
        adjust_offsets2(ast_offsets, len(initial_strip))

    if edges is None:
        raise ValueError("No graph can be generated from the source code")

    # afterprocessing

    # for edge in edges:
    #     edge["file_id"] = source_code_id

    # finish afterprocessing

    all_ast_edges.extend(edges)

    def format_offsets(ast_offsets, target):
        """
        Format offset as a record and add to the common storage for offsets
        :param ast_offsets:
        :param target: List where all other offsets are stored.
        :return: Nothing
        """
        if ast_offsets is not None:
            for offset in ast_offsets:
                target.append({
                    "file_id": 0,  # source_code_id,
                    "start": offset[0],
                    "end": offset[1],
                    "node_id": offset[2],
                    "mentioned_in": offset[3],
                    "string": source_code[offset[0]: offset[1]],
                    "package": "0",  # package
                })

    format_offsets(ast_offsets, target=all_offsets)

    node_resolver.stash_new_nodes()

    all_ast_nodes = node_resolver.new_nodes_for_write(from_stashed=True)

    if all_ast_nodes is None:
        return None, None, None

    def prepare_edges(all_ast_edges):
        all_ast_edges = pd.DataFrame(all_ast_edges)
        all_ast_edges.drop_duplicates(["type", "src", "dst"], inplace=True)
        all_ast_edges = all_ast_edges.query("src != dst")
        all_ast_edges["id"] = 0

        column_order = ["id", "type", "src", "dst", "file_id", "scope"]
        if "offset_start" in all_ast_edges.columns:
            column_order.append("offset_start")
            column_order.append("offset_end")

        all_ast_edges = all_ast_edges[column_order] \
            .rename({'src': 'source_node_id', 'dst': 'target_node_id', 'scope': 'mentioned_in'}, axis=1) \
            .astype({'file_id': 'Int32', "mentioned_in": 'string'})

        all_ast_edges["id"] = range(len(all_ast_edges))
        return all_ast_edges

    all_ast_edges = prepare_edges(all_ast_edges)

    if len(all_offsets) > 0:
        all_offsets = pd.DataFrame(all_offsets)
    else:
        all_offsets = None

    node2id = dict(zip(all_ast_nodes["id"], range(len(all_ast_nodes))))

    def map_columns_to_int(table, dense_columns, sparse_columns):
        types = {column: "int64" for column in dense_columns}
        types.update({column: "Int64" for column in sparse_columns})

        for column, dtype in types.items():
            table[column] = table[column].apply(node2id.get).astype(dtype)

    map_columns_to_int(all_ast_nodes, dense_columns=["id"], sparse_columns=["mentioned_in"])
    map_columns_to_int(
        all_ast_edges,
        dense_columns=["source_node_id", "target_node_id"],
        sparse_columns=["mentioned_in"]
    )
    if all_offsets is not None:
        map_columns_to_int(all_offsets, dense_columns=["node_id"], sparse_columns=["mentioned_in"])

    return all_ast_nodes, all_ast_edges, all_offsets


def build_ast_only_graph(
        source_codes, bpe_tokenizer_path, create_subword_instances, connect_subwords, lang, track_offsets=False
):
    node_resolver = NodeIdResolver()
    mention_tokenizer = MentionTokenizer(bpe_tokenizer_path, create_subword_instances, connect_subwords)
    all_ast_edges = []
    all_offsets = []

    for package, source_code_id, source_code in tqdm(source_codes, desc="Processing modules"):
        source_code_ = source_code.lstrip()
        initial_strip = source_code[:len(source_code) - len(source_code_)]

        if not has_valid_syntax(source_code):
            continue

        edges, ast_offsets = process_code_without_index(
            source_code, node_resolver, mention_tokenizer, track_offsets=track_offsets
        )

        if ast_offsets is not None:
            adjust_offsets2(ast_offsets, len(initial_strip))

        if edges is None:
            continue

        # afterprocessing

        for edge in edges:
            edge["file_id"] = source_code_id

        # finish afterprocessing

        all_ast_edges.extend(edges)

        def format_offsets(ast_offsets, target):
            """
            Format offset as a record and add to the common storage for offsets
            :param ast_offsets:
            :param target: List where all other offsets are stored.
            :return: Nothing
            """
            if ast_offsets is not None:
                for offset in ast_offsets:
                    target.append({
                        "file_id": source_code_id,
                        "start": offset[0],
                        "end": offset[1],
                        "node_id": offset[2],
                        "mentioned_in": offset[3],
                        "string": source_code[offset[0]: offset[1]],
                        "package": package
                    })

        format_offsets(ast_offsets, target=all_offsets)

        node_resolver.stash_new_nodes()

    all_ast_nodes = node_resolver.new_nodes_for_write(from_stashed=True)

    if all_ast_nodes is None:
        return None, None, None

    def prepare_edges(all_ast_edges):
        all_ast_edges = pd.DataFrame(all_ast_edges)
        all_ast_edges.drop_duplicates(["type", "src", "dst"], inplace=True)
        all_ast_edges = all_ast_edges.query("src != dst")
        all_ast_edges["id"] = 0

        column_order = ["id", "type", "src", "dst", "file_id", "scope"]
        if "offset_start" in all_ast_edges.columns:
            column_order.append("offset_start")
            column_order.append("offset_end")

        all_ast_edges = all_ast_edges[column_order] \
            .rename({'src': 'source_node_id', 'dst': 'target_node_id', 'scope': 'mentioned_in'}, axis=1) \
            .astype({'file_id': 'Int32', "mentioned_in": 'string'})

        all_ast_edges["id"] = range(len(all_ast_edges))
        return all_ast_edges

    all_ast_edges = prepare_edges(all_ast_edges)

    if len(all_offsets) > 0:
        all_offsets = pd.DataFrame(all_offsets)
    else:
        all_offsets = None

    node2id = dict(zip(all_ast_nodes["id"], range(len(all_ast_nodes))))

    def map_columns_to_int(table, dense_columns, sparse_columns):
        types = {column: "int64" for column in dense_columns}
        types.update({column: "Int64" for column in sparse_columns})

        for column, dtype in types.items():
            table[column] = table[column].apply(node2id.get).astype(dtype)

    map_columns_to_int(all_ast_nodes, dense_columns=["id"], sparse_columns=["mentioned_in"])
    map_columns_to_int(
        all_ast_edges,
        dense_columns=["source_node_id", "target_node_id"],
        sparse_columns=["mentioned_in"]
    )
    map_columns_to_int(all_offsets, dense_columns=["node_id"], sparse_columns=["mentioned_in"])

    return all_ast_nodes, all_ast_edges, all_offsets


pd.options.mode.chained_assignment = None  # default='warn'


def create_test_data(output_dir):
    # [(id, source), (id, source)]
    test_code = pd.DataFrame.from_records([
        {"id": 1, "filecontent": "import numpy\nnumpy.array([1,2,3])", "package": "any_name_1"},
        {"id": 2, "filecontent": "from numpy.submodule import fn1 as f1, fn2 as f2\n", "package": "can use the same name here any_name_1"},
        {"id": 3, "filecontent": """try:
   a = b
except Exception as e:
   a = c
else:
   a = d
finally:
   print(a)""", "package": "a"}
    ])
    persist(test_code, os.path.join(output_dir, "source_code.bz2"))


def build_ast_graph_from_modules():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source_code", type=str, help="Path to DataFrame pickle (written with pandas.to_pickle, use `bz2` format).")
    parser.add_argument("output_path")
    parser.add_argument("--bpe_tokenizer", type=str, help="Path to sentencepiece model. When provided, names will be subtokenized.")
    parser.add_argument("--visualize", action="store_true", help="Visualize graph. Do not use on large graphs.")
    parser.add_argument("--create_test_data", action="store_true", help="Visualize graph. Do not use on large graphs.")
    args = parser.parse_args()

    if args.create_test_data:
        print(f"Creating test data in {args.output_path}")
        create_test_data(args.output_path)
        sys.exit()

    source_code = unpersist(args.source_code)

    output_dir = args.output_path

    nodes, edges, offsets = build_ast_only_graph(
        zip(source_code["package"], source_code["id"], source_code["filecontent"]), args.bpe_tokenizer,
        create_subword_instances=False, connect_subwords=False, lang="py", track_offsets=True
    )

    print(f"Writing output to {output_dir}")
    persist(source_code, os.path.join(output_dir, "common_filecontent.bz2"))
    persist(nodes, os.path.join(output_dir, "common_nodes.bz2"))
    persist(edges, os.path.join(output_dir, "common_edges.bz2"))
    persist(offsets, os.path.join(output_dir, "common_offsets.bz2"))

    if args.visualize:
        visualize(nodes, edges, os.path.join(output_dir, "visualization.pdf"))


class AstDatasetCreator(AbstractDatasetCreator):

    merging_specification = {
        "source_graph_bodies.bz2": {"columns": ['id'], "output_path": "common_source_graph_bodies.json", "columns_special": [("replacement_list", map_offsets)]},
        "function_variable_pairs.bz2": {"columns": ['src'], "output_path": "common_function_variable_pairs.json"},
        "call_seq.bz2": {"columns": ['src', 'dst'], "output_path": "common_call_seq.json"},

        "nodes_with_ast.bz2": {"columns": ['id', 'mentioned_in'], "output_path": "common_nodes.json", "ensure_unique_with": ['type', 'serialized_name']},
        "edges_with_ast.bz2": {"columns": ['target_node_id', 'source_node_id', 'mentioned_in'], "output_path": "common_edges.json"},
        "offsets.bz2": {"columns": ['node_id', 'mentioned_in'], "output_path": "common_offsets.json"},
        "filecontent_with_package.bz2": {"columns": [], "output_path": "common_filecontent.json"},
        "name_mappings.bz2": {"columns": [], "output_path": "common_name_mappings.json"},
    }

    files_for_merging_with_ast = [
        "nodes_with_ast.bz2", "edges_with_ast.bz2", "source_graph_bodies.bz2", "function_variable_pairs.bz2",
        "call_seq.bz2", "offsets.bz2", "filecontent_with_package.bz2"
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
            recompute_l2g=False, chunksize=10000, keep_frac=1.0, seed=None
    ):
        self.chunksize = chunksize
        self.keep_frac = keep_frac
        self.seed = seed
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
        dataset_location = os.path.dirname(self.path)
        temp_path = os.path.join(dataset_location, "temp_graph_builder")
        self.temp_path = temp_path
        if os.path.isdir(temp_path):
            raise FileExistsError(f"Directory exists: {temp_path}")
        os.mkdir(temp_path)

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

                nodes_with_ast, edges_with_ast, offsets = build_ast_only_graph(
                    zip(source_code["package"], source_code["id"], source_code["filecontent"]), self.bpe_tokenizer,
                    create_subword_instances=self.create_subword_instances, connect_subwords=self.connect_subwords,
                    lang=self.lang, track_offsets=self.track_offsets
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

    def merge(self, output_directory):

        if self.extract:
            logging.info("Extracting...")
            self.do_extraction()

        with_ast_path = self.create_output_dirs(output_directory)

        self.merge_graph_with_ast(with_ast_path)

    def visualize_func(self, nodes, edges, output_path):
        visualize(nodes, edges, output_path)


if __name__ == "__main__":

    args = AstDatasetCreatorArguments().parse()

    if args.recompute_l2g:
        args.do_extraction = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

    dataset = AstDatasetCreator(
        args.source_code, args.language, args.bpe_tokenizer, args.create_subword_instances,
        args.connect_subwords, args.only_with_annotations, args.do_extraction, args.visualize, args.track_offsets,
        args.remove_type_annotations, args.recompute_l2g, args.chunksize, args.keep_frac, args.seed
    )
    dataset.merge(args.output_directory)