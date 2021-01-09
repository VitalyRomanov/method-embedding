import sys

from SourceCodeTools.data.sourcetrail.file_utils import *
from SourceCodeTools.data.sourcetrail.common import merge_with_file_if_exists

pd.options.mode.chained_assignment = None


def read_global_nodes(path):
    if os.path.isfile(path):
        common_nodes = unpersist(path)
        add_node_repr(common_nodes)
        existing_nodes = set(common_nodes['node_repr'].to_list())
        next_valid_id = common_nodes['id'].max() + 1
    else:
        existing_nodes = set()
        next_valid_id = 0
    return existing_nodes, next_valid_id


def read_local_nodes(path):
    batch_nodes = unpersist_or_exit(path)
    add_node_repr(batch_nodes)
    return batch_nodes


def add_node_repr(nodes):
    nodes['node_repr'] = list(zip(nodes['serialized_name'], nodes['type']))


def merge_global_with_local(existing_nodes, next_valid_id, local_nodes):
    new_nodes = local_nodes[
        local_nodes['node_repr'].apply(lambda x: x not in existing_nodes)
    ]

    assert len(new_nodes) == len(set(new_nodes['node_repr'].to_list()))

    ids_start = next_valid_id
    ids_end = ids_start + len(new_nodes)

    new_nodes['id'] = list(range(ids_start, ids_end))

    return new_nodes.drop('node_repr', axis=1)


def write_global_nodes(path, global_nodes):
    if len(global_nodes) > 0:
        persist(global_nodes, path)


def main():
    common_nodes_path = sys.argv[1]
    batch_nodes_path = sys.argv[2]

    existing_nodes, next_valid_id = read_global_nodes(common_nodes_path)
    local_nodes = read_local_nodes(batch_nodes_path)

    if local_nodes is None:
        sys.exit()

    local_id_mapped = merge_global_with_local(existing_nodes, next_valid_id, local_nodes)

    if len(local_id_mapped) == 0:
        sys.exit()

    global_nodes = merge_with_file_if_exists(local_id_mapped, common_nodes_path)

    write_global_nodes(common_nodes_path, global_nodes)


if __name__ == "__main__":
    main()





