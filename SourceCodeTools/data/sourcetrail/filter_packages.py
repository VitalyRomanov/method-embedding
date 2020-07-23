import sys, os
import pandas as pd
from csv import QUOTE_NONNUMERIC

nodes = pd.read_csv(sys.argv[1])
edges = pd.read_csv(sys.argv[2])
bodies = pd.read_csv(sys.argv[3])
tofilter = set(sys.argv[4:])

filtered_nodes = nodes[
    nodes['serialized_name'].apply(lambda x: x.split(".")[0] in tofilter)
]

allowed_ids = set(filtered_nodes['id'].tolist())

filtered_edges = edges[
    edges['source_node_id'].apply(lambda x: x in allowed_ids)
]

filtered_edges = filtered_edges[
    filtered_edges['target_node_id'].apply(lambda x: x in allowed_ids)
]

filtered_bodies = bodies[
    bodies['id'].apply(lambda x: x in allowed_ids)
]

filtered_nodes.to_csv(os.path.join(os.path.dirname(sys.argv[1]), "nodes_filtered_packages.csv"), index=False, quoting=QUOTE_NONNUMERIC)
filtered_edges.to_csv(os.path.join(os.path.dirname(sys.argv[2]), "edges_filtered_packages.csv"), index=False, quoting=QUOTE_NONNUMERIC)
filtered_bodies.to_csv(os.path.join(os.path.dirname(sys.argv[3]), "bodies_filtered_packages.csv"), index=False, quoting=QUOTE_NONNUMERIC)