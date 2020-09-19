from SourceCodeTools.graph.python_ast import AstGraphGenerator
import sys, os
import pandas as pd
from csv import QUOTE_NONNUMERIC
import re
from copy import copy
import ast

from SourceCodeTools.proc.entity.annotator.annotator_utils import to_offsets


# from node_name_serializer import deserialize_node_name

working_directory = sys.argv[1]

node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
edge_path = os.path.join(working_directory, "edges.csv")
bodies_path = os.path.join(working_directory, "source-graph-bodies.csv")

node = pd.read_csv(node_path, sep=",", dtype={"id": int, "type": int, "serialized_name": str})
edge = pd.read_csv(edge_path, sep=",", dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})
bodies = pd.read_csv(bodies_path, sep=",", dtype={"id": int,"body": str,"docstring": str,"normalized_body": str})

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
        for name in re.finditer("srstrlnd_[0-9]+", name_):
            if isinstance(name, re.Match):
                name = name.group()
            elif isinstance(name, str):
                pass
            else:
                print("Unknown type")
            if name.startswith("srstrlnd_"):
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
#     for name in re.finditer("srstrlnd_[0-9]+", name_):
#         if isinstance(name, re.Match):
#             name = name.group()
#         elif isinstance(name, str):
#             pass
#         else:
#             print("Unknown type")
#         if name.startswith("srstrlnd_"):
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
        if pd.isna(row["line"]):
            continue

        if row['src'].startswith("srstrlnd_"):
            nodes.append((
                row['line'],
                row['end_line'],
                row['col_offset'],
                row['end_col_offset'],
                row['src']
            ))

    return nodes



node_resolver = NodeResolver(node)
edges_with_ast_name = os.path.join(working_directory, "edges_with_ast.csv")
edge.to_csv(edges_with_ast_name, index=False, quoting=QUOTE_NONNUMERIC)

for ind, (_, row) in enumerate(bodies.iterrows()):
    c_ = row['normalized_body']
    if not isinstance(c_, str): continue

    c = c_.lstrip()
    strip_len = len(c_) - len(c)

    try:
        ast.parse(c)
    except SyntaxError as e:
        print(e)
        continue

    g = AstGraphGenerator(c)

    edges = g.get_edges()

    srctrlnodes = get_sourcetrail_nodes(edges)

    if len(edges) == 0:
        continue

    # edges['type'] = edges['type'].apply(resolve_edge_type)
    # edges['source_node_id'] = edges['src'].apply(resolve_node_names)
    # edges['target_node_id'] = edges['dst'].apply(resolve_node_names)
    edges['source_node_id'] = edges['src'].apply(node_resolver.resolve)
    edges['target_node_id'] = edges['dst'].apply(node_resolver.resolve)
    edges['id'] = 0

    replacements = []
    for ind, row in edges.iterrows():
        if pd.isna(row['line']):
            continue
        replacements.append((
            row['line'],
            row['end_line'],
            row['col_offset'],
            row['end_col_offset'],
            row['source_node_id']
        ))

    replacements_offsets = to_offsets(c, replacements)

    if strip_len > 0:
        replacements_offsets = list(map(lambda x: (x[0] + strip_len, x[1] + strip_len, x[2]), replacements_offsets))

    edges[['id','type','source_node_id','target_node_id']].to_csv(edges_with_ast_name, mode="a", index=False, header=False)
    print("\r%d/%d" % (ind, len(bodies['normalized_body'])), end="")

print(" " * 30 , end = "\r")
# pd.DataFrame(type_maps).to_csv("new_types.csv", index=False)


with open(os.path.join(working_directory, "nodes_with_ast.csv"), 'w', encoding='utf8', errors='replace') as f:
    pd.concat([node, pd.DataFrame(node_resolver.new_nodes)]).to_csv(f, index=False, quoting=QUOTE_NONNUMERIC)
# pd.concat([node, pd.DataFrame(new_nodes)]).to_csv(os.path.join(working_directory, "nodes_with_ast.csv"), index=False)