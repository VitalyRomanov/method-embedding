import sys
from typing import Iterable

from SourceCodeTools.code.data.sourcetrail.common import \
    map_id_columns, merge_with_file_if_exists, create_local_to_global_id_map
from SourceCodeTools.code.data.sourcetrail.file_utils import *


def map_columns(input_table, id_map, columns, columns_special=None):

    input_table = map_id_columns(input_table, columns, id_map)

    if columns_special is not None:
        assert isinstance(columns_special, list), "`columns_special` should be iterable"
        for column, map_func in columns_special:
            input_table[column] = map_func(input_table[column], id_map)

    if len(input_table) == 0:
        return None
    else:
        return input_table


if __name__ == "__main__":
    global_nodes_path = sys.argv[1]
    local_nodes_path = sys.argv[2]
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    columns = sys.argv[5:]

    global_nodes = unpersist_or_exit(global_nodes_path, exit_message = "Error: global nodes do not exist!")
    local_nodes = unpersist_or_exit(local_nodes_path)
    input_table = unpersist_or_exit(input_path)

    id_map = create_local_to_global_id_map(local_nodes=local_nodes, global_nodes=global_nodes)

    data = map_columns(input_table, id_map, columns)

    if data is not None:
        data = merge_with_file_if_exists(df=input_table, merge_with_file=output_path)
        persist(data, output_path)
