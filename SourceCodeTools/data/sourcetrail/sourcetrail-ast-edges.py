from SourceCodeTools.graph.python_ast import AstGraphGenerator
import sys, os
import pandas as pd
from csv import QUOTE_NONNUMERIC
import re
from copy import copy
import ast

from SourceCodeTools.proc.entity.annotator.annotator_utils import to_offsets, overlap


# from node_name_serializer import deserialize_node_name

class NodeResolver:
    def __init__(self, node):

        self.nodeid2name = dict(zip(node['id'].tolist(), node['serialized_name'].tolist()))

        # self.valid_new_type = 260 # fits about 250 new type
        # self.type_maps = {}
        # self.new_types = []

        self.valid_new_node = node['id'].max() + 1
        self.ast_node_type = 3
        self.node_maps = {}
        self.new_nodes = []

    def resolve(self, name_orig):

        name_ = copy(name_orig)

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

        # try to replace into node id
        for r, v in replacements.items():
            name_ = name_.replace(r, v["id"])

        try:
            node_id = int(name_)
            return node_id
        except:
            # can go here if the node is not sourcetrail node or
            # if the node is a composite type
            pass

        # failed to convert into id, create new node
        name_ = copy(name_orig)
        for r, v in replacements.items():
            name_ = name_.replace(r, v["name"])

        name = name_
        if name not in self.node_maps:
            self.node_maps[name] = self.valid_new_node
            self.new_nodes.append({"id": self.valid_new_node, "type": self.ast_node_type, "serialized_name": name})
            self.valid_new_node += 1
        return self.node_maps[name]


# def resolve_edge_type(edge_type):
#     global valid_new_type, type_maps, new_types
#
#     # if edge_type not in type_maps:
#     #     type_maps[edge_type] = valid_new_type
#     #     new_types.append({"type_id": valid_new_type, "type_desc": edge_type})
#     #     valid_new_type += 1
#     #     if valid_new_type == 512: raise Exception("Type overlap!")
#     return edge_type #type_maps[edge_type]


# def resolve_node_names(name_orig):
#     global valid_new_node, ast_node_type, node_maps, new_nodes
#
#     name_ = copy(name_orig)
#
#     replacements = dict()
#     for name in re.finditer("src trlnd_[0-9]+", name_):
#         if isinstance(name, re.Match):
#             name = name.group()
#         elif isinstance(name, str):
#             pass
#         else:
#             print("Unknown type")
#         if name.startswith("srctrlnd_"):
#             node_id = name.split("_")[1]
#             replacements[name] = {
#                 "name": nodeid2name[int(node_id)],
#                 "id": node_id
#             }
#             # if nodeid2name[int(node_id)]=="bokeh.io.output.output_file":
#             #     pass
#
#     # try to replace into node id
#     for r, v in replacements.items():
#         name_ = name_.replace(r, v["id"])
#
#     try:
#         node_id = int(name_)
#         return node_id
#     except:
#         # can go here if the node is not sourcetrail node or
#         # if the node is a composite type
#         pass
#
#     # failed to convert into id, create new node
#     name_ = copy(name_orig)
#     for r, v in replacements.items():
#         name_ = name_.replace(r, v["name"])
#
#     name = name_
#     if name not in node_maps:
#         node_maps[name] = valid_new_node
#         new_nodes.append({"id": valid_new_node, "type": ast_node_type, "serialized_name": name})
#         valid_new_node += 1
#     return node_maps[name]
#
#     # if len(replacements) == 1:
#     #     # this is an existing node
#     #     for r, v in replacements.items():
#     #         name_ = name_.replace(r, v["id"])
#     #     node_id = int(name_)
#     #     return node_id
#     # else:
#     #     # this is a new node, has either 0 or >=2 replacements
#     #     for r, v in replacements.items():
#     #         name_ = name_.replace(r, v["name"])
#     #
#     #     name = name_
#     #     if name not in node_maps:
#     #         node_maps[name] = valid_new_node
#     #         new_nodes.append({"id": valid_new_node, "type": ast_node_type, "serialized_name": name})
#     #         valid_new_node += 1
#     #     return node_maps[name]

def get_sourcetrail_nodes(edges):
    nodes = []
    for ind, row in edges.iterrows():
        if "line" not in row or pd.isna(row["line"]):
            continue

        # if row['src'].startswith("srctrlnd_"):
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
    if "(" in node_string or ")" in node_string or "[" in node_string or "]" in node_string or "{" in node_string or "}" in node_string or " " in node_string or "," in node_string:
            return False
    return True


def filter_nodes(offsets, body):
    """
    Prevents overlaps between nodes extracted from AST and nodes provided by Sourcetrail
    """
    return [offset for offset in offsets if keep_node(body[offset[0]:offset[1]])]


def join_offsets(offsets_1, offsets_2, body=None):
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


def write_edges_v1(bodies, node_resolver, nodes_with_ast_name, edges_with_ast_name):
    for ind_bodies, (_, row) in enumerate(bodies.iterrows()):
        c_ = row['normalized_body']
        if not isinstance(c_, str): continue

        c = c_.lstrip()

        try:
            ast.parse(c)
        except SyntaxError as e:
            print(e)
            continue

        g = AstGraphGenerator(c)

        edges = g.get_edges()

        if len(edges) == 0:
            continue

        # edges['type'] = edges['type'].apply(resolve_edge_type)
        # edges['source_node_id'] = edges['src'].apply(resolve_node_names)
        # edges['target_node_id'] = edges['dst'].apply(resolve_node_names)
        edges['source_node_id'] = edges['src'].apply(node_resolver.resolve)
        edges['target_node_id'] = edges['dst'].apply(node_resolver.resolve)
        edges['id'] = 0

        edges[['id', 'type', 'source_node_id', 'target_node_id']].to_csv(edges_with_ast_name, mode="a", index=False,
                                                                         header=False)
        print("\r%d/%d" % (ind_bodies, len(bodies['normalized_body'])), end="")

    print(" " * 30, end="\r")

    with open(nodes_with_ast_name, 'w', encoding='utf8', errors='replace') as f:
        pd.concat([node, pd.DataFrame(node_resolver.new_nodes)]).to_csv(f, index=False, quoting=QUOTE_NONNUMERIC)


def write_edges_v2(bodies, node_resolver, nodes_with_ast_name, edges_with_ast_name):
    bodies_with_replacements = []

    for ind_bodies, (_, row) in enumerate(bodies.iterrows()):
        orig_body = row['random_replacements']
        if not isinstance(orig_body, str): continue

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

        random_replacements_lookup = lambda x: ast.literal_eval(row['random_2_srctrl']).get(x,x)

        edges['src'] = edges['src'].apply(random_replacements_lookup)
        edges['dst'] = edges['dst'].apply(random_replacements_lookup)

        edges['src'] = edges['src'].apply(node_resolver.resolve)
        edges['dst'] = edges['dst'].apply(node_resolver.resolve)
        edges['id'] = 0

        # if ind_bodies == 475:
        #     print()

        # srctrlnodes = to_offsets(c, get_sourcetrail_nodes(edges))
        # for x in srctrlnodes:
        #     x = (b2c[x[0]], b2c[x[1]], x[2])
        srctrlnodes = filter_nodes(adjust_offsets(
            to_offsets(c, get_sourcetrail_nodes(edges), as_bytes=True)
            , -strip_len), orig_body)

        replacements = list(map(
            lambda x: (x[0], x[1], node_resolver.resolve(x[2])),
            to_offsets(row['random_replacements'],
                       format_replacement_offsets(ast.literal_eval(row['replacement_list'])))))

        all_offsets = join_offsets(
            sorted(srctrlnodes, key=lambda x: x[0]),
            sorted(replacements, key=lambda x: x[0]), orig_body
        )

        bodies_with_replacements.append({
            "id": row['id'],
            "body": row['body'],
            "replacement_list": all_offsets
        })

        # replacements = []
        # for ind, row in edges.iterrows():
        #     if pd.isna(row['line']):
        #         continue
        #     replacements.append((
        #         row['line'],
        #         row['end_line'],
        #         row['col_offset'],
        #         row['end_col_offset'],
        #         row['source_node_id']
        #     ))

        edges[['id', 'type', 'src', 'dst']].to_csv(edges_with_ast_name, mode="a", index=False,
                                                                         header=False)
        print("\r%d/%d" % (ind_bodies, len(bodies['normalized_body'])), end="")

    print(" " * 30, end="\r")

    pd.DataFrame(bodies_with_replacements).to_csv(
        os.path.join(os.path.dirname(nodes_with_ast_name), "bodies_with_replacements.csv"), index=False, quoting=QUOTE_NONNUMERIC
    )

    with open(nodes_with_ast_name, 'w', encoding='utf8', errors='replace') as f:
        pd.concat([node, pd.DataFrame(node_resolver.new_nodes)]).to_csv(f, index=False, quoting=QUOTE_NONNUMERIC)

if __name__ == "__main__":

    working_directory = sys.argv[1]

    node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
    edge_path = os.path.join(working_directory, "edges.csv")
    bodies_path = os.path.join(working_directory, "source-graph-bodies.csv")

    node = pd.read_csv(node_path, sep=",", dtype={"id": int, "type": int, "serialized_name": str})
    edge = pd.read_csv(edge_path, sep=",", dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})
    bodies = pd.read_csv(bodies_path, sep=",", dtype={"id": int, "body": str, "docstring": str, "normalized_body": str})

    node_resolver = NodeResolver(node)
    edges_with_ast_name = os.path.join(working_directory, "edges_with_ast.csv")
    nodes_with_ast_name = os.path.join(working_directory, "nodes_with_ast.csv")
    edge.to_csv(edges_with_ast_name, index=False, quoting=QUOTE_NONNUMERIC)

    # write_edges_v1(bodies, node_resolver, nodes_with_ast_name, edges_with_ast_name)
    write_edges_v2(bodies, node_resolver, nodes_with_ast_name, edges_with_ast_name)

