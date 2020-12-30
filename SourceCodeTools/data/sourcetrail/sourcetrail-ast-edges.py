import ast
import os
import re
import sys
from copy import copy
from csv import QUOTE_NONNUMERIC

import pandas as pd
from nltk import RegexpTokenizer

from SourceCodeTools.data.sourcetrail.sourcetrail_types import node_types, edge_types
from SourceCodeTools.graph.python_ast import AstGraphGenerator
from SourceCodeTools.graph.python_ast import GNode
from SourceCodeTools.proc.entity.annotator.annotator_utils import to_offsets, overlap, resolve_self_collision

pd.options.mode.chained_assignment = None


def create_subword_tokenizer(lang, vs):
    from pathlib import Path
    from bpemb.util import sentencepiece_load, http_get
    import re

    def _load_file(file, archive=False):
        cache_dir = Path.home() / Path(".cache/bpemb")
        archive_suffix = ".tar.gz"
        base_url = "https://nlp.h-its.org/bpemb/"
        cached_file = Path(cache_dir) / file
        if cached_file.exists():
            return cached_file
        suffix = archive_suffix if archive else ""
        file_url = base_url + file + suffix
        print("downloading", file_url)
        return http_get(file_url, cached_file, ignore_tardir=True)
    model_file = "{lang}/{lang}.wiki.bpe.vs{vs}.model".format(lang=lang, vs=vs)
    model_file = _load_file(model_file)
    spm = sentencepiece_load(model_file)
    return lambda text: spm.EncodeAsPieces(re.sub(r"\d", "0", text.lower()))


class NodeResolver:
    def __init__(self, nodes):

        self.nodeid2name = dict(zip(nodes['id'].tolist(), nodes['serialized_name'].tolist()))
        self.nodeid2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))

        self.valid_new_node = nodes['id'].max() + 1
        self.node_ids = {}
        self.new_nodes = []

        self.old_nodes = nodes.copy()
        self.old_nodes['mentioned_in'] = -1

    def get_new_node_id(self):
        new_id = self.valid_new_node
        self.valid_new_node += 1
        return new_id

    def resolve(self, node, srctrl2original):

        decorated = "@" in node.name
        assert len([c for c in node.name if c == "@"]) <= 1

        if decorated:
            name_, decorator = node.name.split("@")
        else:
            name_ = copy(node.name)

        if name_ in srctrl2original:
            node_id = int(name_.split("_")[1])
            if decorated:
                real_name = srctrl2original[name_]
                real_name += "@" + decorator
                type_ = "mention"
                new_node = GNode(name=real_name, type=type_, name_scope="local")
            else:
                real_name = self.nodeid2name[node_id]
                type_ = self.nodeid2type[node_id]
                new_node = GNode(name=real_name, type=type_, id=node_id, name_scope="global")
            return new_node

        replacements = dict()
        for name in re.finditer("srctrlnd_[0-9]+", name_):
            if isinstance(name, re.Match):
                name = name.group()
            elif isinstance(name, str):
                pass
            else:
                print("Unknown type")
            if name.startswith("srctrlnd_"):
                node_id = name.split("_")[1]
                replacements[name] = {
                    "name": self.nodeid2name[int(node_id)],
                    "id": node_id
                }

        real_name = name_
        for r, v in replacements.items():
            real_name = name_.replace(r, v["name"])

        if decorated:
            real_name += "@" + decorator

        return GNode(name=real_name, type=node.type)

    def resolve_node_id(self, node, function_id):
        if not hasattr(node, "id"):
            node_repr = (node.name, node.type)

            if node_repr in self.node_ids:
                node.setprop("id", self.node_ids[node_repr])
            else:
                new_id = self.get_new_node_id()
                self.node_ids[node_repr] = new_id
                self.new_nodes.append(
                    {"id": new_id, "type": node.type, "serialized_name": node.name, "mentioned_in": function_id})

                node.setprop("id", new_id)
        return node


def get_ast_nodes(edges):
    nodes = []
    for ind, row in edges.iterrows():
        if "line" not in row or pd.isna(row["line"]):
            continue

        nodes.append((
            row['line'],
            row['end_line'],
            row['col_offset'],
            row['end_col_offset'],
            row['src']
        ))

    return nodes


def get_byte_to_char_map(unicode_string):
    """
    Generates a dictionary mapping character offsets to byte offsets for unicode_string.
    """
    response = {}
    byte_offset = 0
    for char_offset, character in enumerate(unicode_string):
        response[byte_offset] = char_offset
        print(character, byte_offset, char_offset)
        byte_offset += len(character.encode('utf-8'))
    response[byte_offset] = len(unicode_string)
    return response


def adjust_offsets(offsets, amount):
    return [(offset[0] - amount, offset[1] - amount, offset[2]) for offset in offsets]


def format_replacement_offsets(offsets):
    return [(offset[0], offset[0], offset[1], offset[2], offset[3]) for offset in offsets]


def keep_node(node_string):
    if "(" in node_string or ")" in node_string or "[" in node_string or "]" in node_string or \
            "{" in node_string or "}" in node_string or " " in node_string or "," in node_string:
            return False
    return True


def filter_nodes(offsets, body):
    """
    Prevents overlaps between nodes extracted from AST and nodes provided by Sourcetrail
    """
    return [offset for offset in offsets if keep_node(body[offset[0]:offset[1]])]


def join_offsets(offsets_1, offsets_2):
    joined = []
    while offsets_1 or offsets_2:
        if len(offsets_1) == 0:
            joined.append(offsets_2.pop(0))
        elif len(offsets_2) == 0:
            joined.append(offsets_1.pop(0))
        elif offsets_1[0] == offsets_2[0]:
            joined.append(offsets_1.pop(0))
            offsets_2.pop(0)
        elif overlap(offsets_1[0], offsets_2[0]):
            # Exception: ('Should not overlap:', (360, 382, 611771), (368, 375, 611758))
            # >>> body[360:382]
            # Out[1]: 'ZIIwSfl(JicFbMT, item)'
            # >>> body[368:375]
            # Out[3]: 'JicFbMT'
            #
            # it appears some nodes can overlap. preserve the smallest one
            len_1 = offsets_1[0][1] - offsets_1[0][0]
            len_2 = offsets_2[0][1] - offsets_2[0][0]
            if len_1 < len_2:
                joined.append(offsets_1.pop(0))
                offsets_2.pop(0)
            elif len_2 > len_1:
                joined.append(offsets_2.pop(0))
                offsets_1.pop(0)
            else:
                # print("Should not overlap:", offsets_1[0], offsets_2[0])
                # TODO
                # it seems to be some unreasonable error. skip both versions
                # joined.append(offsets_1.pop(0))
                offsets_1.pop(0)
                offsets_2.pop(0)
                # raise Exception("Should not overlap:", offsets_1[0], offsets_2[0])
        elif offsets_1[0][0] < offsets_2[0][0]:
            joined.append(offsets_1.pop(0))
        elif offsets_1[0][0] > offsets_2[0][0]:
            joined.append(offsets_2.pop(0))
        else:
            raise Exception("Illegal scenario")

    return joined


def random_replacement_lookup(name, replacements, tokenizer):
    if "[" not in name:
        return replacements.get(name, name)
    else:
        tokens = tokenizer.tokenize(name)
        r_tokens = map(lambda x: replacements.get(x, x), tokens)
        corrected_name = "".join(r_tokens)
        return corrected_name


def filter_out_mentions_for_srctrl_nodes(edges):
    # filter mention_scope edges for sourcetrail nodes
    srctrl_decode = lambda x: \
        GNode(name=x.name.split("@")[0], type="Name") if x.name.startswith("srctrlnd") and "@" in x.name else x
    edges['src'] = edges['src'].apply(srctrl_decode)
    edges['dst'] = edges['dst'].apply(srctrl_decode)
    edges = edges.query("src!=dst")
    edges['help'] = edges['dst'].apply(lambda x: x.name.split("_")[0])
    edges = edges.query("help!='srctrlnd' or type!='mention_scope'").drop("help", axis=1)
    return edges


def replace_mentions_with_subwords(edges, bpe):
    edges = edges.to_dict(orient="records")

    new_edges = []
    for edge in edges:
        if edge['type'] == "local_mention":
            dst = edge['dst']
            subwords = bpe(edge['src'])
            for ind, subword in enumerate(subwords):
                new_edges.append({
                    'src': subword,
                    'dst': dst,
                    'type': 'subword',
                    'line': pd.NA,
                    'end_line': pd.NA,
                    'col_offset': pd.NA,
                    'end_col_offset': pd.NA,
                })
                # if ind < len(subwords) - 1:
                #     new_edges.append({
                #         'src': subword,
                #         'dst': subwords[ind + 1],
                #         'type': 'next_subword',
                #         'line': pd.NA,
                #         'end_line': pd.NA,
                #         'col_offset': pd.NA,
                #         'end_col_offset': pd.NA,
                #     })
                # if ind > 0:
                #     new_edges.append({
                #         'src': subword,
                #         'dst': subwords[ind - 1],
                #         'type': 'prev_subword',
                #         'line': pd.NA,
                #         'end_line': pd.NA,
                #         'col_offset': pd.NA,
                #         'end_col_offset': pd.NA,
                #     })
        else:
            new_edges.append(edge)

    return pd.DataFrame(new_edges)


def produce_subword_edges(subwords, dst):
    new_edges = []

    subwords = list(map(lambda x: GNode(name=x, type="subword"), subwords))
    instances = list(map(lambda x: GNode(name=x.name + "@" + dst.name, type="subword_instance"), subwords))
    for ind, subword in enumerate(subwords):
        subword_instance = instances[ind]
        new_edges.append({
            'src': subword,
            'dst': subword_instance,
            'type': 'subword_instance',
            'line': pd.NA,
            'end_line': pd.NA,
            'col_offset': pd.NA,
            'end_col_offset': pd.NA,
        })
        new_edges.append({
            'src': subword_instance,
            'dst': dst,
            'type': 'subword',
            'line': pd.NA,
            'end_line': pd.NA,
            'col_offset': pd.NA,
            'end_col_offset': pd.NA,
        })
        if ind < len(subwords) - 1:
            new_edges.append({
                'src': subword_instance,
                'dst': instances[ind + 1],
                'type': 'next_subword',
                'line': pd.NA,
                'end_line': pd.NA,
                'col_offset': pd.NA,
                'end_col_offset': pd.NA,
            })
        if ind > 0:
            new_edges.append({
                'src': subword_instance,
                'dst': instances[ind - 1],
                'type': 'prev_subword',
                'line': pd.NA,
                'end_line': pd.NA,
                'col_offset': pd.NA,
                'end_col_offset': pd.NA,
            })

    return new_edges


def global_mention_edges(edge):
    edge['type'] = "global_mention"
    return [edge, make_reverse_edge(edge)]


def make_reverse_edge(edge):
    rev_edge = copy(edge)
    rev_edge['type'] = edge['type'] + "_rev"
    rev_edge['src'] = edge['dst']
    rev_edge['dst'] = edge['src']
    return rev_edge


def replace_mentions_with_subword_instances(edges, bpe):
    edges = edges.to_dict(orient="records")

    new_edges = []
    for edge in edges:
        if edge['type'] == "local_mention":
            if hasattr(edge['src'], "id"):
                # this edge connects sourcetrail node need to add couple of links
                # to ensure global information flow
                new_edges.extend(global_mention_edges(edge))
                # edge['type'] = "global_mention"
                # rev_edge = copy(edge)
                # rev_edge['src'] = edge['dst']
                # rev_edge['dst'] = edge['src']
                # rev_edge['type'] = "global_mention_rev"
                # new_edges.append(edge)
                # new_edges.append(rev_edge)

            dst = edge['dst']

            # this is
            if hasattr(dst, "name_scope") and dst.name_scope == "local":
                subwords = bpe(dst.name.split("@")[0])
            else:
                subwords = bpe(edge['src'].name)

            new_edges.extend(produce_subword_edges(subwords, dst))

        elif edge['type'] == "attr":
            new_edges.append(edge)
            new_edges.append(make_reverse_edge(edge))

            dst = edge['src']
            subwords = bpe(dst.name)
            new_edges.extend(produce_subword_edges(subwords, dst))

        elif edge['type'] == "name" or edge['type'] == "names":
            if hasattr(edge['src'], "id"):
                new_edges.extend(global_mention_edges(edge))

            dst = edge['dst']
            subwords = bpe(edge['src'].name.split(".")[-1])
            new_edges.extend(produce_subword_edges(subwords, dst))
        else:
            new_edges.append(edge)

    return pd.DataFrame(new_edges)


def get_srctrl2original_replacements(record):
    random2srctrl = ast.literal_eval(record['random_2_srctrl'])
    random2original = ast.literal_eval(record['random_2_original'])

    return {random2srctrl[key]: random2original[key] for key in random2original}


def append_edges(path, edges):
    edges[['id', 'type', 'src', 'dst']].to_csv(path, mode="a", index=False, header=False)


def write_bodies(path, bodies):
    pd.DataFrame(bodies).to_csv(
        path,
        index=False, quoting=QUOTE_NONNUMERIC
    )


def write_nodes(path, node_resolver):
    with open(path, 'w', encoding='utf8', errors='replace') as f:
        pd.concat([node_resolver.old_nodes, pd.DataFrame(node_resolver.new_nodes)])\
            [['id', 'type', 'serialized_name', 'mentioned_in']].to_csv(f, index=False, quoting=QUOTE_NONNUMERIC)

def add_reverse_edges(edges):
    rev_edges = edges.copy()
    rev_edges['source_node_id'] = edges['target_node_id']
    rev_edges['target_node_id'] = edges['source_node_id']
    rev_edges['type'] = rev_edges['type'].apply(lambda x: x + "_rev")

    return pd.concat([edges, rev_edges], axis=0)

def write_edges_v2(bodies, node_resolver, nodes_with_ast_name, edges_with_ast_name, n_subwords=1000000):
    bodies_with_replacements = []
    with open(os.path.join(os.path.dirname(nodes_with_ast_name), "bodies_with_replacements.csv"), "w") as sink:
        sink.write("id,body,replacement_list\n")

    subword_tokenizer = create_subword_tokenizer(lang="multi", vs=n_subwords)
    tokenizer = RegexpTokenizer("\w+|[^\w\s]")

    for ind_bodies, (_, row) in enumerate(bodies.iterrows()):
        orig_body = row['random_replacements']
        if not isinstance(orig_body, str):
            continue

        srctrl2original = get_srctrl2original_replacements(row)

        c = orig_body.lstrip()
        strip_len = len(orig_body) - len(c)

        try:
            ast.parse(c)
        except SyntaxError as e:
            print(e)
            continue

        g = AstGraphGenerator(c)

        edges = g.get_edges()

        if len(edges) == 0:
            continue

        replacements = ast.literal_eval(row['random_2_srctrl'])
        # replacements_lookup = lambda x: complex_replacement_lookup(x, replacements)
        replacements_lookup = lambda x: \
            GNode(name=random_replacement_lookup(x.name, replacements, tokenizer),
                  type=x.type) if "@" not in x.name else \
            GNode(name=random_replacement_lookup(x.name.split("@")[0], replacements, tokenizer) +
                  "@" + x.name.split("@")[1],
                  type=x.type)

        edges['src'] = edges['src'].apply(replacements_lookup)
        edges['dst'] = edges['dst'].apply(replacements_lookup)

        # edges = filter_out_mentions_for_srctrl_nodes(edges)

        resolve = lambda node: node_resolver.resolve(node, srctrl2original)

        edges['src'] = edges['src'].apply(resolve)
        edges['dst'] = edges['dst'].apply(resolve)

        edges = replace_mentions_with_subword_instances(edges, subword_tokenizer)
        edges['id'] = 0

        resolve_node_id = lambda node: node_resolver.resolve_node_id(node, row['id'])

        edges['src'] = edges['src'].apply(resolve_node_id)
        edges['dst'] = edges['dst'].apply(resolve_node_id)

        extract_id = lambda node: node.id
        edges['src'] = edges['src'].apply(extract_id)
        edges['dst'] = edges['dst'].apply(extract_id)

        ast_nodes = resolve_self_collision(filter_nodes(adjust_offsets(
            to_offsets(c, get_ast_nodes(edges), as_bytes=True), -strip_len), orig_body))

        srctrl_nodes = list(map(
            lambda x: (x[0], x[1], node_resolver.resolve(GNode(name=x[2], type="Name"), srctrl2original).id),
            to_offsets(row['random_replacements'],
                       format_replacement_offsets(ast.literal_eval(row['replacement_list'])))
        ))

        all_offsets = join_offsets(
            sorted(ast_nodes, key=lambda x: x[0]),
            sorted(srctrl_nodes, key=lambda x: x[0])
        )

        bodies_with_replacements.append({
            "id": row['id'],
            "body": row['body'],
            "replacement_list": all_offsets
        })

        append_edges(path=edges_with_ast_name, edges=edges)
        print("\r%d/%d" % (ind_bodies, len(bodies['normalized_body'])), end="")

    print(" " * 30, end="\r")

    write_bodies(path=os.path.join(os.path.dirname(nodes_with_ast_name), "bodies_with_replacements.csv"),
                 bodies=bodies_with_replacements)

    write_nodes(path=nodes_with_ast_name, node_resolver=node_resolver)


def main(argv):
    working_directory = argv[1]
    try:
        n_subwords = int(argv[2])
    except:
        n_subwords = 1000000

    node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
    edge_path = os.path.join(working_directory, "edges.csv")
    bodies_path = os.path.join(working_directory, "source-graph-bodies.csv")

    node = pd.read_csv(node_path, sep=",", dtype={"id": int, "type": int, "serialized_name": str})
    edge = pd.read_csv(edge_path, sep=",", dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})
    bodies = pd.read_csv(bodies_path, sep=",", dtype={"id": int, "body": str, "docstring": str, "normalized_body": str})

    node['type'] = node['type'].apply(lambda type: node_types[type])
    edge['type'] = edge['type'].apply(lambda type: edge_types[type])
    edge = add_reverse_edges(edge)

    node_resolver = NodeResolver(node)
    edges_with_ast_name = os.path.join(working_directory, "edges_with_ast.csv")
    nodes_with_ast_name = os.path.join(working_directory, "nodes_with_ast.csv")

    edge.to_csv(edges_with_ast_name, index=False, quoting=QUOTE_NONNUMERIC)

    # write_edges_v1(bodies, node_resolver, nodes_with_ast_name, edges_with_ast_name)
    write_edges_v2(bodies, node_resolver, nodes_with_ast_name, edges_with_ast_name, n_subwords=n_subwords)


if __name__ == "__main__":
    main(sys.argv)
