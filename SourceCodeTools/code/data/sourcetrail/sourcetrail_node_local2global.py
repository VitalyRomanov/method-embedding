import sys

from SourceCodeTools.code.data.file_utils import *
from SourceCodeTools.code.data.sourcetrail.common import create_local_to_global_id_map


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
