#%%
import sys, os
import pandas as pd

# nodes = pd.read_csv(sys.argv[1])
# edges = pd.read_csv(sys.argv[2])
nodes = pd.read_csv("/Volumes/External/datasets/Code/source-graphs/python-source-graph/v2_with_ast/00_sourcetrail_export/common_nodes_with_ast.csv")
edges = pd.read_csv("/Volumes/External/datasets/Code/source-graphs/python-source-graph/v2_with_ast/00_sourcetrail_export/common_edges_with_types_with_ast.csv")
edges['id'] = edges['target_node_id']

i = edges.query("type == -59").index
t = edges.loc[i, 'target_node_id']
s = edges.loc[i, 'source_node_id']

edges.loc[i, 'target_node_id'] = s
edges.loc[i, 'source_node_id'] = t



#%%
nodes = nodes.dropna(axis=0)
nodes['package'] = nodes['serialized_name'].apply(lambda x: x.split(".")[0])

sources_with_names = edges.merge(nodes, on="id")

annotation_edges = edges.query("type == -2 or type == -3")

#%%

def inspect_children(edges, source_id):
    branches = edges.query(f"source_node_id == {source_id}")
    # print(branches['serialized_name'])
    for ind, row in branches.iterrows():
        if len(row['serialized_name'].split(".")) > 1:
            # print(row)
            return row['package']
    for ind, row in branches.iterrows():
        return inspect_children(edges, row['target_node_id'])

for ind, row in annotation_edges.iterrows():
    package_name = inspect_children(sources_with_names, row['target_node_id'])
    print(package_name)