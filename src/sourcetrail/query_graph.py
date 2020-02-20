#%%
import pandas

nodes_path = "/Volumes/External/datasets/Code/source-graphs/java-s/normalized_sourcetrail_nodes.csv"
edges_path = "/Volumes/External/datasets/Code/source-graphs/java-s/edges.csv"

nodes = pandas.read_csv(nodes_path)
edges = pandas.read_csv(edges_path)

#%%

query_node = 1535303

#%%

dsts = [edges.query(f"source_node_id == {query_node}")]

# for dst in dsts[0][['target_node_id']].values:
#     dsts.append(edges.query(f"source_node_id == {dst}"))

n_esges = pandas.concat(dsts, axis=0)


# %%

dsts2 = [edges.query(f"target_node_id == {query_node}")]

# for dst in dsts2[0][['source_node_id']].values:
#     dsts2.append(edges.query(f"target_node_id == {dst}"))

n_esges2 = pandas.concat(dsts2, axis=0)

# %%
n_esges3 = pandas.concat([n_esges, n_esges2], axis=0)

# %%
ids = set(n_esges3['source_node_id'].values.tolist()) | set(n_esges3['target_node_id'].values.tolist())

query = " or ".join([f"id == {node_id}" for node_id in ids])

n_nodes = nodes.query(query)

# %%
n_nodes.to_csv("/Volumes/External/datasets/Code/source-graphs/java-s/query_nodes.csv", index=False)
n_esges3.to_csv("/Volumes/External/datasets/Code/source-graphs/java-s/query_edges.csv", index=False)

# %%
