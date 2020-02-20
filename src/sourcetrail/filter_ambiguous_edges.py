#%%
import sys
import pandas as pd
import os

# TODO:
# 1. Add CSV header automatically

# working_directory = "/Users/LTV/Documents/sourcetrail/numpy/" #sys.argv[1]
# working_directory = "/Volumes/External/datasets/Code/source-graphs/python-source-graph" #sys.argv[1]
working_directory = sys.argv[1]

edges_path = os.path.join(working_directory, "edges.csv") #sys.argv[1]
element_component_path = os.path.join(working_directory, "element_component.csv") #sys.argv[2]

if not os.path.isfile(edges_path):
    raise Exception("File not found: %s" % edges_path)
if not os.path.isfile(element_component_path):
    raise Exception("File not found: %s" % element_component_path)

#%%

# as of january 2020, it seems that all elements in element_component table are ambiguous
ambiguous_edges = set()
for chunk in pd.read_csv(element_component_path, sep=",", chunksize=1000000):
    ambiguous_edges |= set(chunk["element_id"].values.tolist())


# ambiguous_edges = set(pd.read_csv(element_component_path, sep=",")["element_id"].values.tolist())

#%%


non_ambiguous_path = os.path.join(working_directory, "non-ambiguous_edges.csv")
ambiguous_path = os.path.join(working_directory, "ambiguous_edges.csv")

# with open(ambiguous_path, "w") as ambiguous:
#     ambiguous.write("id,type,source_node_id,target_node_id\n")
#     with open(non_ambiguous_path, "w") as non_ambiguous:
#         non_ambiguous.write("id,type,source_node_id,target_node_id\n")
#         for chunk in pd.read_csv(edges_path, chunksize=100):
#             for row_ind, row in chunk.iterrows():
#                 if row.id in ambiguous_edges:
#                     ambiguous.write("%d,%d,%d,%d\n" % (row.id, row.type, row.source_node_id, row.target_node_id))
#                 else:
#                     non_ambiguous.write("%d,%d,%d,%d\n" % (row.id, row.type, row.source_node_id, row.target_node_id))

with open(non_ambiguous_path, "w") as non_ambiguous:
    non_ambiguous.write("id,type,source_node_id,target_node_id\n")
with open(ambiguous_path, "w") as ambiguous:
    ambiguous.write("id,type,source_node_id,target_node_id\n")
            
for chunk_ind, chunk in enumerate(pd.read_csv(edges_path, chunksize=100000)):
    amb_ind = chunk["id"].apply(lambda x: x in ambiguous_edges).values
    chunk[~amb_ind].to_csv(non_ambiguous_path, header=False, index=False, mode='a')
    chunk[amb_ind].to_csv(ambiguous_path, header=False, index=False, mode='a')
    


# %%
