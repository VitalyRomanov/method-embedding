import pandas as pd
import sys, os
from csv import QUOTE_NONNUMERIC

from SourceCodeTools.data.sourcetrail.file_utils import *
from SourceCodeTools.data.sourcetrail.common import create_node_repr, \
    create_local_to_global_id_map


def add_global_ids(global_nodes, local_nodes):
    id_map = create_local_to_global_id_map(local_nodes=local_nodes, global_nodes=global_nodes)

    local_nodes['global_id'] = local_nodes['id'].apply(lambda x: id_map.get(x, -1))


if __name__ == "__main__":
    global_nodes = unpersist_or_exit(sys.argv[1], "Global nodes do not exist!")
    local_nodes = unpersist_or_exit(sys.argv[2], "No processed nodes, skipping")
    local_map_path = sys.argv[3]

    add_global_ids(global_nodes, local_nodes)

    persist(local_nodes[['id', 'global_id']], local_map_path)
