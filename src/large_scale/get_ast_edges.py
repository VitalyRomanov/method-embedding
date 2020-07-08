from python_ast import AstGraphGenerator
import sys, os
import pandas as pd
from csv import QUOTE_NONNUMERIC
# from node_name_serializer import deserialize_node_name

working_directory = sys.argv[1]

node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
edge_path = os.path.join(working_directory, "edges.csv")
bodies_path = os.path.join(working_directory, "source-graph-bodies.csv")

node = pd.read_csv(node_path, sep=",", dtype={"id": int, "type": int, "serialized_name": str})
edge = pd.read_csv(edge_path, sep=",", dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})
bodies = pd.read_csv(bodies_path, sep=",", dtype={"id": int,"body": str,"docstring": str,"normalized_body": str})

valid_new_type = 260 # fits about 250 new type
type_maps = {}
new_types = []

valid_new_node = node['id'].max() + 1
ast_node_type = 3
node_maps = {}
new_nodes = []

def resolve_edge_type(edge_type):
    global valid_new_type, type_maps, new_types

    # if edge_type not in type_maps:
    #     type_maps[edge_type] = valid_new_type
    #     new_types.append({"type_id": valid_new_type, "type_desc": edge_type})
    #     valid_new_type += 1
    #     if valid_new_type == 512: raise Exception("Type overlap!")
    return edge_type #type_maps[edge_type]

def resolve_node_names(name):
    global valid_new_node, ast_node_type, node_maps, new_nodes

    if name.startswith("srstrlnd_"):
        try:
            node_id = int(name.split("_")[1])
        except:
            print(name)
            raise Exception()
        return node_id
    else:
        if name not in node_maps:
            node_maps[name] = valid_new_node
            new_nodes.append({"id": valid_new_node, "type": ast_node_type, "serialized_name": name})
            valid_new_node += 1
        return node_maps[name]

edges_with_ast_name = os.path.join(working_directory, "edges_with_ast.csv")
edge.to_csv(edges_with_ast_name, index=False, quoting=QUOTE_NONNUMERIC)

for ind, c in enumerate(bodies['normalized_body']):
    try:
        try:
            c.strip()
        except:
            # print(c)
            continue
        g = AstGraphGenerator(c.strip())
        edges = g.get_edges()

        try:
            edges['type'] = edges['type'].apply(resolve_edge_type)
            edges['source_node_id'] = edges['src'].apply(resolve_node_names)
            edges['target_node_id'] = edges['dst'].apply(resolve_node_names)
            edges['id'] = 0
        except KeyError:
            if len(edges) == 0:
                continue
            else:
                print(edges)
                raise Exception()
        except:
            print(c)
            raise Exception()

        edges[['id','type','source_node_id','target_node_id']].to_csv(edges_with_ast_name, mode="a", index=False, header=False)
        print("\r%d/%d" % (ind, len(bodies['normalized_body'])), end="")
    except SyntaxError:
        # print(c.strip())
        pass
print(" " * 30 , end = "\r")
pd.DataFrame(type_maps).to_csv("new_types.csv", index=False)


with open(os.path.join(working_directory, "nodes_with_ast.csv"), 'w', encoding='utf8', errors='replace') as f:
    pd.concat([node, pd.DataFrame(new_nodes)]).to_csv(f, index=False, quoting=QUOTE_NONNUMERIC)
# pd.concat([node, pd.DataFrame(new_nodes)]).to_csv(os.path.join(working_directory, "nodes_with_ast.csv"), index=False)