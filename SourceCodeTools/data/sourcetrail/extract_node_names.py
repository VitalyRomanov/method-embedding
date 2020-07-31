    #%%
import math
import sys, os
import pandas as p
from csv import QUOTE_NONNUMERIC

def get_node_name(full_name):
    return full_name.split(".")[-1].split("___")[0]

nodes_path = sys.argv[1]
# nodes_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/v2/00_sourcetrail_export/common_nodes.csv"

data = p.read_csv(nodes_path)
# some cells are empty, probably because of empty strings in AST
data = data.dropna(axis=0)
data = data[data['type'] != 262144]
data['src'] = data['id']
data['dst'] = data['serialized_name'].apply(get_node_name)

#%%

counts = data['dst'].value_counts()

#%%
# print(data.shape)
data['counts'] = data['dst'].apply(lambda x: counts[x])
data = data.query("counts > 1")
# print(data.shape)

#%%

data[['src', 'dst']].to_csv(os.path.join(os.path.dirname(nodes_path), "node_names.csv"), index=False, quoting=QUOTE_NONNUMERIC)