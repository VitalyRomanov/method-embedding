import pandas as pd
import sys, os
from csv import QUOTE_NONNUMERIC

from SourceCodeTools.data.sourcetrail.file_utils import *
from SourceCodeTools.data.sourcetrail.common import create_node_repr, \
    create_local_to_global_id_map

# TODO
# sometimes ast nodes are preferred

all_nodes = unpersist(sys.argv[1])
orig_nodes = unpersist(sys.argv[2])
# all_nodes = pd.read_csv(sys.argv[1], dtype={"id": int, "type": str, "serialized_name": str})
# orig_nodes = pd.read_csv(sys.argv[2], dtype={"id": int, "type": str, "serialized_name": str})
local_map_path = sys.argv[3]

id_map = create_local_to_global_id_map(local_nodes=orig_nodes, global_nodes=all_nodes)
# all_nodes['node_repr'] = create_node_repr(all_nodes)
# orig_nodes['node_repr'] = create_node_repr(orig_nodes)
#
# rev_id_map = dict(zip(all_nodes['node_repr'].tolist(), all_nodes['id'].tolist()))
# id_map = dict(zip(orig_nodes["id"].tolist(), map(lambda x: rev_id_map[x], orig_nodes["node_repr"].tolist())))

orig_nodes['global_id'] = orig_nodes['id'].apply(lambda x: id_map.get(x, -1))

# orig_nodes[['id', 'global_id']].to_csv(local_map_path, index=False)
persist(orig_nodes[['id', 'global_id']], local_map_path)