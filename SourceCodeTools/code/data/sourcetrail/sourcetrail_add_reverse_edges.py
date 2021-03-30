from SourceCodeTools.code.data.sourcetrail.file_utils import *

import pandas as p
import sys

from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping


def get_reverse_type_name(type, special_mapping):
    return special_mapping.get(type, type+"_rev")


def add_reverse_edges(edges):
    rev_edges = edges.copy()

    rev_edges['source_node_id'] = edges['target_node_id']
    rev_edges['target_node_id'] = edges['source_node_id']
    rev_edges['type'] = rev_edges['type'].apply(lambda x: get_reverse_type_name(x, special_mapping))
    # rev_edges['type'] = rev_edges['type'].apply(lambda x: x + "_rev")
    rev_edges['id'] = -1

    return p.concat([edges, rev_edges], axis=0)


if __name__ == "__main__":
    edges_path = sys.argv[1]

    edges = unpersist_or_exit(edges_path)
    edges = add_reverse_edges(edges)

    persist(edges, edges_path)