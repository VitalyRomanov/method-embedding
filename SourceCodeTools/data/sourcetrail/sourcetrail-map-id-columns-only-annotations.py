import sys

from SourceCodeTools.data.sourcetrail.common import create_node_repr, \
    map_id_columns, merge_with_file_if_exists, create_local_to_global_id_map
from SourceCodeTools.data.sourcetrail.file_utils import *

all_nodes = unpersist(sys.argv[1])
orig_nodes = unpersist(sys.argv[2])
# all_nodes = pd.read_csv(sys.argv[1], dtype={"id": int, "type": str, "serialized_name": str})
# orig_nodes = pd.read_csv(sys.argv[2], dtype={"id": int, "type": str, "serialized_name": str})

input_path = sys.argv[3]
output_path = sys.argv[4]
columns = sys.argv[5:]

input_table = unpersist(input_path)

if len(input_table.query("type == 'annotation_for' or type == 'returned_by'")) > 0:
    id_map = create_local_to_global_id_map(local_nodes=orig_nodes, global_nodes=all_nodes)

    # all_nodes['node_repr'] = create_node_repr(all_nodes)
    # orig_nodes['node_repr'] = create_node_repr(orig_nodes)
    #
    # rev_id_map = dict(zip(all_nodes['node_repr'].tolist(), all_nodes['id'].tolist()))
    # id_map = dict(zip(orig_nodes["id"].tolist(), map(lambda x: rev_id_map[x], orig_nodes["node_repr"].tolist())))

    input_table = map_id_columns(input_table, columns, id_map)

    if len(input_table) == 0:
        sys.exit()

    data = merge_with_file_if_exists(df=input_table, merge_with_file=output_path)

    persist(data, output_path)