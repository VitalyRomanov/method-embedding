import pandas as pd
import sys, os

all_nodes_path = sys.argv[1]
map_path = sys.argv[2]

orig_nodes_path = os.path.join(map_path, "normalized_sourcetrail_nodes.csv")
edges_path = os.path.join(map_path, "edges.csv")
bodies_path = os.path.join(map_path, "source-graph-bodies.csv")

all_nodes = pd.read_csv(all_nodes_path)
orig_nodes = pd.read_csv(orig_nodes_path)
edges = pd.read_csv(edges_path)
bodies = pd.read_csv(bodies_path)

rev_id_map = dict(zip(all_nodes['serialized_name'].tolist(), all_nodes['id'].tolist()))

id_map = dict(zip(orig_nodes["id"].tolist(), map(lambda x: rev_id_map[x], orig_nodes["serialized_name"].tolist())))

edges['source_node_id'] = edges['source_node_id'].apply(lambda x: id_map[x])
edges['target_node_id'] = edges['target_node_id'].apply(lambda x: id_map[x])
edges[['type','source_node_id','target_node_id']].to_csv(os.path.join(map_path, "edges_global.csv"), index=False, header=False)

bodies['id'] = bodies['id'].apply(lambda x: id_map[x])
bodies.to_csv(os.path.join(map_path, "source-graph-bodies_global.csv"), index=False, header=False)