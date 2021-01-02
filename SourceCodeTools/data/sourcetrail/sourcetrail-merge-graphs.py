import sys

from SourceCodeTools.data.sourcetrail.file_utils import *
from SourceCodeTools.data.sourcetrail.common import merge_with_file_if_exists

pd.options.mode.chained_assignment = None


def read_global_nodes(path):
    if os.path.isfile(path):
        common_nodes = unpersist(path)
        add_node_repr(common_nodes)
        existing_nodes = set(common_nodes['node_repr'].to_list())
        records = common_nodes
        next_valid_id = common_nodes['id'].max() + 1
    else:
        existing_nodes = set()
        records = None
        next_valid_id = 0
    return existing_nodes, next_valid_id


def read_local_nodes(path):
    batch_nodes = unpersist(path)
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

    # new_global = []
    # for ind, local_node in local_nodes.iterrows():
    #     node_repr = (local_node['serialized_name'], local_node['type'])
    #     if node_repr in existing_nodes: continue
    #
    #     # TODO
    #     # sometimes there are duplicate nodes from sourcetrail and ast analysis.
    #     # something does not work right when checking uniqueness of a name
    #
    #     local_node['id'] = len(global_nodes) + len(new_global)
    #
    #     new_global.append(local_node)
    #     existing_nodes.add(node_repr)  # should be redundant since all nodes within one file are unique



    # global_nodes = pd.concat([
    #     global_nodes.drop('node_repr', axis=1),
    #     new_nodes.drop('node_repr', axis=1)
    # ], axis=0)
    # return global_nodes


def write_global_nodes(path, global_nodes):
    if len(global_nodes) != 0:
        persist(global_nodes, path)
        # pd.DataFrame(global_nodes).to_csv(path, index=False, quoting=QUOTE_NONNUMERIC)
    # else:
    #     with open(path, "w") as sink:
    #         sink.write("id,type,serialized_name\n")


def main():
    common_nodes_path = sys.argv[1]
    batch_nodes_path = sys.argv[2]

    existing_nodes, next_valid_id = read_global_nodes(common_nodes_path)
    local_nodes = read_local_nodes(batch_nodes_path)

    local_id_mapped = merge_global_with_local(existing_nodes, next_valid_id, local_nodes)

    global_nodes = merge_with_file_if_exists(local_id_mapped, common_nodes_path)

    write_global_nodes(common_nodes_path, global_nodes)


if __name__ == "__main__":
    main()





