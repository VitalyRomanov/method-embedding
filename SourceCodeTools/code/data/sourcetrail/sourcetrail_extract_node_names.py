import sys

from SourceCodeTools.code.data.ast_graph.extract_node_names import extract_node_names
from SourceCodeTools.code.data.file_utils import *


def get_node_name(full_name):
    """
    Used for processing java nodes
    """
    return full_name.split(".")[-1].split("___")[0]


def extract_node_names_(nodes, min_count):
    # some cells are empty, probably because of empty strings in AST
    # data = nodes.dropna(axis=0)
    # data = data[data['type'] != 262144]
    nodes = nodes.copy()
    nodes['serialized_name'] = nodes['serialized_name'].apply(get_node_name)
    return extract_node_names(nodes, min_count)


if __name__ == "__main__":
    nodes_path = sys.argv[1]
    out_path = sys.argv[2]
    try:
        min_count = int(sys.argv[3])
    except:
        min_count = 1

    nodes = unpersist_or_exit(nodes_path)

    names = extract_node_names(nodes, min_count)

    if names is not None:
        persist(names, out_path)
