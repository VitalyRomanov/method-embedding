import sys
import pandas

nodes_path = sys.argv[1]
edges_path = sys.argv[2]

nodes = pandas.read_csv(nodes_path)
edges = pandas.read_csv(edges_path)

edges = edges[edges['type'] == 8]

src = set(edges['source_node_id'].values.tolist())
dst = set(edges['target_node_id'].values.tolist())

for ind, row in nodes.iterrows():
    nid = row['id']
    name = row['serialized_name']

    if nid in src and nid in dst:
        print(name)