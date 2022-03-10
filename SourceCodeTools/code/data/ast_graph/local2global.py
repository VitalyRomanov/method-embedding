import sys

from SourceCodeTools.code.common import compute_long_id, create_node_repr
from SourceCodeTools.code.data.file_utils import *


def create_local_to_global_id_map(local_nodes, global_nodes):
    # local_nodes = local_nodes.copy()
    # global_nodes = global_nodes.copy()
    #
    # global_nodes['node_repr'] = create_node_repr(global_nodes)
    # local_nodes['node_repr'] = create_node_repr(local_nodes)
    #
    # rev_id_map = dict(zip(
    #     global_nodes['node_repr'].tolist(), global_nodes['id'].tolist()
    # ))
    # id_map = dict(zip(
    #     local_nodes["id"].tolist(), map(
    #         lambda x: rev_id_map[x], local_nodes["node_repr"].tolist()
    #     )
    # ))
    id_map = dict(zip(
        local_nodes["id"], map(compute_long_id, create_node_repr(local_nodes))
    ))

    return id_map


def get_local2global(global_nodes, local_nodes) -> pd.DataFrame:
    local_nodes = local_nodes.copy()
    id_map = create_local_to_global_id_map(local_nodes=local_nodes, global_nodes=global_nodes)

    local_nodes['global_id'] = local_nodes['id'].apply(lambda x: id_map.get(x, None))

    return local_nodes[['id', 'global_id']]


if __name__ == "__main__":
    global_nodes = unpersist_or_exit(sys.argv[1], "Global nodes do not exist!")
    local_nodes = unpersist_or_exit(sys.argv[2], "No processed nodes, skipping")
    local_map_path = sys.argv[3]

    local2global = get_local2global(global_nodes, local_nodes)

    persist(local2global, local_map_path)
