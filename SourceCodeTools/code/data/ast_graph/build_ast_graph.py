import hashlib
import os.path
import sys

import pandas as pd

from tqdm import tqdm

from SourceCodeTools.code.ast import has_valid_syntax
from SourceCodeTools.code.data.file_utils import persist, unpersist
from SourceCodeTools.code.data.sourcetrail.sourcetrail_draw_graph import visualize
from SourceCodeTools.code.ast.python_ast2 import AstGraphGenerator, GNode, PythonSharedNodes
from SourceCodeTools.code.annotator_utils import adjust_offsets2
from SourceCodeTools.code.annotator_utils import to_offsets, get_cum_lens
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
            # if edge['src'].type in {"#attr#", "Name"}:
            #     if hasattr(edge['src'], "global_id"):
            #         new_edges.extend(self.global_mention_edges_from_node(edge['src']))
            # elif edge['dst'].type == "mention":
            #     if hasattr(edge['dst'], "global_id"):
            #         new_edges.extend(self.global_mention_edges_from_node(edge['dst']))

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
                    # new_edges.append(make_reverse_edge(edge))

                dst = edge['src']
                subwords = self.bpe(dst.name)
                new_edges.extend(produce_subw_edges(subwords, dst))
            # elif self.bpe is not None and edge['dst'].type in {"Global"} and edge['src'].type != "Constant":
            #     # this brach is disabled because it does not seem to make sense
            #     # Globals can be referred by Name nodes, but they are already processed in the branch above
            #     new_edges.append(edge)
            #     new_edges.append(make_reverse_edge(edge))
            #
            #     dst = edge['src']
            #     subwords = self.bpe(dst.name)
            #     new_edges.extend(produce_subw_edges(subwords, dst))
            else:
                new_edges.append(edge)

        return new_edges

    # @staticmethod
    # def global_mention_edges_from_node(node):
    #     global_edges = []
    #     if type(node.global_id) is int:
    #         id_type = [(node.global_id, node.global_type)]
    #     else:
    #         id_type = zip(node.global_id, node.global_type)
    #
    #     for gid, gtype in id_type:
    #         global_mention = {
    #             "src": GNode(name=None, type=gtype, id=gid),
    #             "dst": node,
    #             "type": "global_mention",
    #             "offsets": None
    #         }
    #         global_edges.append(global_mention)
    #         global_edges.append(make_reverse_edge(global_mention))
    #     return global_edges

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
                self.connect_prev_next_subwords(new_edges, subword, subwords[ind - 1] if ind > 0 else None,
                                                subwords[ind + 1] if ind < len(subwords) - 1 else None)
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


class AstProcessor(AstGraphGenerator):
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


def process_code_without_index(source_code, node_resolver, mention_tokenizer, track_offsets=False):
    try:
        ast_processor = AstProcessor(source_code)
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
            return [(edge["offsets"][0], edge["offsets"][1], edge["src"], edge["scope"]) for edge in edges if edge["offsets"] is not None]

        # recover ast offsets for the current file
        ast_offsets = get_valid_offsets(edges)
    else:
        ast_offsets = None

    return edges, ast_offsets


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
            :param global_and_ast_offsets:
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

        all_ast_edges = all_ast_edges[["id", "type", "src", "dst", "file_id", "scope"]] \
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

    all_ast_nodes["id"] = all_ast_nodes["id"].apply(node2id.get)
    all_ast_nodes["mentioned_in"] = all_ast_nodes["mentioned_in"].apply(node2id.get)
    all_ast_edges["source_node_id"] = all_ast_edges["source_node_id"].apply(node2id.get)
    all_ast_edges["target_node_id"] = all_ast_edges["target_node_id"].apply(node2id.get)
    all_ast_edges["mentioned_in"] = all_ast_edges["mentioned_in"].apply(node2id.get)
    all_offsets["node_id"] = all_offsets["node_id"].apply(node2id.get)
    all_offsets["mentioned_in"] = all_offsets["mentioned_in"].apply(node2id.get)

    return all_ast_nodes, all_ast_edges, all_offsets


pd.options.mode.chained_assignment = None  # default='warn'


def create_test_data(output_dir):
    # [(id, source), (id, source)]
    test_code = pd.DataFrame.from_records([
        {"id": 1, "filecontent": "import numpy\nnumpy.array([1,2,3])", "package": "any_name_1"},
        {"id": 2, "filecontent": "from numpy import *\n", "package": "can use the same name here any_name_1"},
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


if __name__ == "__main__":

    build_ast_graph_from_modules()