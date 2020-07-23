import pandas as p
import sys, os

nodes = p.read_csv(sys.argv[1])

edges = p.read_csv(sys.argv[2])

from pprint import pprint
df_dict = nodes[['id', 'serialized_name']].to_dict()
ids = df_dict['id'].values()
serialized_name = df_dict['serialized_name'].values()
id_map = dict(zip(ids, serialized_name))

def map_name(node_id):
    return id_map.get(node_id, node_id)

edges['source_node_id'] = edges['source_node_id'].apply(map_name)
edges['target_node_id'] = edges['target_node_id'].apply(map_name)

edges.to_csv(os.path.join(os.path.dirname(sys.argv[2]), 'normalized_sourcetrail_edges.csv'), index=False)