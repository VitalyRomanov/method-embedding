import sys
from SourceCodeTools.code.data.file_utils import *

nodes = unpersist(sys.argv[1])
edges = unpersist(sys.argv[2])

id_map = dict(zip(nodes['id'], nodes['serialized_name']))

def map_name(node_id):
    return id_map.get(node_id, node_id)

edges['source_node_id'] = edges['source_node_id'].apply(map_name)
edges['target_node_id'] = edges['target_node_id'].apply(map_name)

edges.to_csv(os.path.join(os.path.dirname(sys.argv[2]), 'normalized_sourcetrail_edges.csv'), index=False)