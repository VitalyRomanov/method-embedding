#%%
 
import ast
import sys
import pandas
import os

# base_folder = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/"
base_folder = "/home/ltv/data/datasets/source_code/python-source-graph/"

bodies_path = base_folder + "functions.csv" # sys.argv[1]
nodes_path = base_folder + "normalized_sourcetrail_nodes.csv" # sys.argv[2]
edges_path = base_folder + "edges_component_0.csv" # sys.argv[3]
occurrence_path = base_folder + "occurrence.csv"
source_location_path = base_folder + "source_location.csv"

#%%

bodies = pandas.read_csv(bodies_path)
nodes = pandas.read_csv(nodes_path)
edges = pandas.read_csv(edges_path)
# occurrence = pandas.read_csv(occurrence_path)
# source_location = pandas.read_csv(source_location_path)

#%% 

# occurrence.set_index("element_id", inplace=True)
# source_location.set_index("id", inplace=True)
# occurrence.sort_index(inplace=True)
# source_location.sort_index(inplace=True)

#%%

# src_loc_id2data = dict(zip(
#         source_location['id'].values,
#         zip(
#             source_location['file_node_id'].values, 
#             source_location['start_line'].values, 
#             source_location['end_line'].values,
#             source_location['type'].values
#             )
#         ))
#%%

# node2occurence = dict()

# for ind, row in occurrence.iterrows():
#     if row['element_id'] not in node2occurence:
#         node2occurence[row['element_id']] = []

#     node2occurence[row['element_id']].append(row['source_location_id'])

#     if ind % 10000 == 0:
#         print(f"\r{ind}/{occurrence.shape[0]}", end="")

# # for node_id, group in occurrence.groupby('element_id'):
# #     node2occurence[node_id] = group['source_location_id'].values.tolist()

# # node2occurence = dict([(node_id, group['source_location_id'].values.tolist()) for node_id, group in occurrence.groupby('element_id')])


#%%

def process_names(name):
    parts = name.split(".")

    if parts[-1] == "__init__":
        return parts[-2]
    else:
        return parts[-1]

nodes_id2name = dict(zip(nodes['id'].values, nodes['serialized_name'].values))

# nodes_id2name = dict()

# for ind, row in nodes.iterrows():
#     nodes_id2name[row['id']] = row['serialized_name'].

#%%

edges_group = edges[edges['type'] == 8].groupby("source_node_id")


# %%

def get_body(node_id, bodies):
    b = bodies.query(f"id == {node_id}")
    if b.shape[0] == 0:
        return None
    else:
        if pandas.isna(b.iloc[0]['content']):
            return None
        return b.iloc[0]['content']
    # try:
        
    #     return .iloc[0]['content']
    # except KeyError:
    #     return None

# def get_locations(node_id):
#     locations_ids = occurrence.query(f"element_id == {node_id}")['source_location_id'].values
#     if len(locations_ids) == 0: 
#         return []
#     # definition = pandas.concat([source_location.query(f"id == {lid}") for lid in locations_ids], axis=0).query("type == 1")
#     # definition = [src_loc_id2data[lid] for lid in locations_ids]
#     for lid in locations_ids:
#         yield src_loc_id2data[lid]
    
    # return definition

# def get_definition_location(node_id):
#     locations = get_locations(node_id)
#     def_loc = list(filter(lambda x: x[-1] == 1, locations))
#     return def_loc[0] if def_loc else None

# def get_order(node_id, children):

#     def_loc = get_definition_location(node_id)

#     if def_loc is None:
#         return None

#     file_id, s_l, e_l, o_type = def_loc

#     order = []
#     for child in children:
#         for oc in filter(lambda x: x[0] == file_id and x[1] >= s_l and x[2] <= e_l, get_locations(child)):
#             #   and x[1]==x[2]
#             order.append((child, oc))

#     return order

    

#%%

def resolve_call(calls, chld_names):
    resolved = []
    for c in calls:
        for ch in chld_names:
            if c[0] in ch[1]:
                resolved.append((ch[0], c[1]))
                break
        else:
            resolved.append(("?", c[1]))
    return resolved


def get_name_from_ast(n, node_id):
    if hasattr(n.func, "id"):
        return n.func.id
    elif hasattr(n.func, "attr"):
        return n.func.attr
    else:
        return "??"
    # elif hasattr(n.func, "slice"):
    #     return "??"
    # elif hasattr(n.func, "func"):
    #     return "??"
    # elif n.func.__class__.__name__ == "Lambda":
    #     return "??"
    # else:
    #     raise Exception(node_id, n.func, dir(n.func), n.func.lineno, n.func.__class__.__name__)

import ast

sink = open("consequent_calls.txt", "w")

CHARS_IN_LINE = 100

for ind, (node_id, group) in enumerate(edges_group):

    body_str = get_body(node_id, bodies)
    chld = group['target_node_id'].values.tolist()
    chld_names = list(map(lambda x: nodes_id2name[x], chld))
    ch_and_name = list(zip(chld, chld_names))

    if body_str is not None:
        
        try:
            tree = ast.parse(body_str)
        except SyntaxError:
            continue

        calls = \
            [
                (
                    get_name_from_ast(n, node_id), 
                    n.lineno * CHARS_IN_LINE + n.col_offset
                ) for n in ast.walk(tree) if hasattr(n, "func")
            ]
        resolved = resolve_call(calls, ch_and_name)
        resolved = list(filter(lambda x: x[0] != "?", resolved))
        ordered = sorted(resolved, key=lambda x: x[1])
        ordered_nodes = [str(o[0]) for o in ordered]
        if len(ordered_nodes) > 1:
            sink.write("%s\n" % '\t'.join(ordered_nodes))

            # print("\t".join(ordered_nodes))
        # print(node_id, chld, ordered_nodes)
    # def_loc = get_definition_location(node_id)

    # if body_str is None and def_loc is not None or body_str is not None and def_loc is None:
    # if def_loc is not None:
    #     print(node_id, chld, def_loc, get_order(node_id, chld))

    # if ind == 20: break
    if ind %1000 == 0:
        print(f"\r{ind}/200000", end="")
sink.close()

# %%



# %%
