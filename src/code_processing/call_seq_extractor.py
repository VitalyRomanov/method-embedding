#%%
 
# import ast
import sys
# import pandas
import os
# import ast
import pandas as pd

# lang = sys.argv[1]
working_directory = sys.argv[1]
# # base_folder = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/"
# # base_folder = "/home/ltv/data/datasets/source_code/python-source-graph/"
#
# bodies_path = os.path.join(working_directory, "source-graph-functions.csv")
# nodes_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
# edges_path = os.path.join(working_directory, "edges.csv")
# # occurrence_path = os.path.join(working_directory, "occurrence.csv")
# # source_location_path = os.path.join(working_directory, "source_location.csv")
#
# #%%
#
# bodies = pandas.read_csv(bodies_path)
# nodes = pandas.read_csv(nodes_path)
# edges = pandas.read_csv(edges_path)
#
#
# #%%
#
# def process_names(name):
#     parts = name.split(".")
#
#     if parts[-1] == "__init__":
#         return parts[-2]
#     else:
#         return parts[-1]
#
# nodes_id2name = dict(zip(nodes['id'].values, nodes['serialized_name'].values))
#
#
# #%%
#
# edges_group = edges[edges['type'] == 8].groupby("source_node_id")
#
#
# # %%
#
# def get_body(node_id, bodies):
#     b = bodies.query(f"id == {node_id}")
#     if b.shape[0] == 0:
#         return None
#     else:
#         if pandas.isna(b.iloc[0]['content']):
#             return None
#         return b.iloc[0]['content']
#
#
#
# #%%
#
# def resolve_call(calls, chld_names):
#     resolved = []
#     for c in calls:
#         for ch in chld_names:
#             if c[0] in ch[1]:
#                 resolved.append((ch[0], c[1]))
#                 break
#         else:
#             resolved.append(("?", c[1]))
#     return resolved
#
#
# def get_name_from_ast(n, node_id):
#     if hasattr(n.func, "id"):
#         return n.func.id
#     elif hasattr(n.func, "attr"):
#         return n.func.attr
#     else:
#         return "??"
#     # elif hasattr(n.func, "slice"):
#     #     return "??"
#     # elif hasattr(n.func, "func"):
#     #     return "??"
#     # elif n.func.__class__.__name__ == "Lambda":
#     #     return "??"
#     # else:
#     #     raise Exception(node_id, n.func, dir(n.func), n.func.lineno, n.func.__class__.__name__)
#
#
# CHARS_IN_LINE = 100
#
# with open(os.path.join(working_directory, "source-graph-consequent-calls.txt"), "w") as sink:
#
#     for ind, (node_id, group) in enumerate(edges_group):
#
#         body_str = get_body(node_id, bodies)
#         chld = group['target_node_id'].values.tolist()
#         chld_names = list(map(lambda x: nodes_id2name[x], chld))
#         ch_and_name = list(zip(chld, chld_names))
#
#         if body_str is not None:
#
#             try:
#                 tree = ast.parse(body_str.strip())
#             except SyntaxError:
#                 continue
#
#             calls = \
#                 [
#                     (
#                         get_name_from_ast(n, node_id),
#                         n.lineno * CHARS_IN_LINE + n.col_offset
#                     ) for n in ast.walk(tree) if hasattr(n, "func")
#                 ]
#             resolved = resolve_call(calls, ch_and_name)
#             resolved = list(filter(lambda x: x[0] != "?", resolved))
#             ordered = sorted(resolved, key=lambda x: x[1])
#             ordered_nodes = [str(o[0]) for o in ordered]
#             if len(ordered_nodes) > 1:
#                 sink.write("%s\n" % '\t'.join(ordered_nodes))
#
#                 # print("\t".join(ordered_nodes))
#
#         if ind %1000 == 0:
#             print(f"\r{ind}/{bodies.shape[0]}", end="")













source_location_path = os.path.join(working_directory, "source_location.csv")
occurrence_path = os.path.join(working_directory, "occurrence.csv")
node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
edge_path = os.path.join(working_directory, "edges.csv")
# filecontent_path = os.path.join(working_directory, "filecontent.csv")

print("Reading data...", end ="")
source_location = pd.read_csv(source_location_path, sep=",")
occurrence = pd.read_csv(occurrence_path, sep=",")
node = pd.read_csv(node_path, sep=",")
edge = pd.read_csv(edge_path, sep=",").rename(columns={'type':'e_type'})
# filecontent = pd.read_csv(filecontent_path, sep=",")

node_edge = pd.concat([node, edge], sort=False)
node_edge = node_edge.astype({"target_node_id": "Int32", "source_node_id": "Int32"})

print("ok", end ="\n")

assert len(node_edge["id"].unique()) == len(node_edge), f"{len(node_edge['id'].unique())} != {len(node_edge)}"

source_location.rename(columns={'id':'source_location_id', 'type':'occ_type'}, inplace=True)
node_edge.rename(columns={'id':'element_id'}, inplace=True)

occurrences = occurrence.merge(source_location, on='source_location_id',)

nodes = node_edge.merge(occurrences, on='element_id')

occurrence_group = nodes.groupby("file_node_id")

DEFINITION_TYPE = 1

bodies = []
with open(os.path.join(working_directory, "call_seq.csv"), "w") as sink:
    sink.write("src,dst\n")

    for occ_ind, (group_id, group) in enumerate(occurrence_group):


        definitions = group.query(f"occ_type == {DEFINITION_TYPE} and (type == 4096 or type == 8192)")

        if len(definitions):
            # print("\n\n\n")
            # print(definitions)
            for ind, row in definitions.iterrows():
                elements = group.query(f"start_line >= {row.start_line} and end_line <= {row.end_line} and occ_type != {DEFINITION_TYPE} and e_type == 8")

                # print(elements)

                # sources = filecontent.query(f"id == {group_id}").iloc[0]['content'].split("\n")

                # body = "\n".join(sources[row.start_line - 1: row.end_line - 1])
                # print(body)

                elements.sort_values(by=["start_line", "end_column"], inplace=True)

                # print(elements)

                all_calls = elements['target_node_id'].tolist()

                # for start_line_g, line_calls in elements.groupby("start_line"):
                #
                #     line_calls.sort_values(by="end_column", inplace=True)
                #
                #     all_calls.extend(line_calls["target_node_id"].tolist())

                # print(all_calls)
                for i in range(len(all_calls) - 1):
                    sink.write(f"{all_calls[i]},{all_calls[i+1]}\n")

                # print("\n\n\n")
        print(f"\r{occ_ind}/{len(occurrence_group)}", end="")
    print()