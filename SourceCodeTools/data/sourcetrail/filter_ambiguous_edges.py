#%%
import sys
import pandas as pd
import os

working_directory = sys.argv[1]

edges_path = os.path.join(working_directory, "edges.csv")
element_component_path = os.path.join(working_directory, "element_component.csv")

if not os.path.isfile(edges_path):
    raise Exception("File not found: %s" % edges_path)
if not os.path.isfile(element_component_path):
    raise Exception("File not found: %s" % element_component_path)

# as of january 2020, it seems that all elements in element_component table are ambiguous
ambiguous_edges = set()
for chunk in pd.read_csv(element_component_path, sep=",", chunksize=1000000):
    ambiguous_edges |= set(chunk["element_id"].values.tolist())

non_ambiguous_path = os.path.join(working_directory, "non-ambiguous_edges.csv")
ambiguous_path = os.path.join(working_directory, "ambiguous_edges.csv")

with open(non_ambiguous_path, "w") as non_ambiguous:
    non_ambiguous.write("id,type,source_node_id,target_node_id\n")
with open(ambiguous_path, "w") as ambiguous:
    ambiguous.write("id,type,source_node_id,target_node_id\n")
            
for chunk_ind, chunk in enumerate(pd.read_csv(edges_path, chunksize=100000)):
    amb_ind = chunk["id"].apply(lambda x: x in ambiguous_edges).values
    chunk[~amb_ind].to_csv(non_ambiguous_path, header=False, index=False, mode='a')
    chunk[amb_ind].to_csv(ambiguous_path, header=False, index=False, mode='a')
