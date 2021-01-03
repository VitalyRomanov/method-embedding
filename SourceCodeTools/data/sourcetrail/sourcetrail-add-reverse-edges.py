from SourceCodeTools.data.sourcetrail.sourcetrail_types import edge_types
from SourceCodeTools.data.sourcetrail.file_utils import *

import pandas as p
import sys
import os


def add_reverse_edges(edges):
    rev_edges = edges.copy()
    rev_edges['source_node_id'] = edges['target_node_id']
    rev_edges['target_node_id'] = edges['source_node_id']
    rev_edges['type'] = rev_edges['type'].apply(lambda x: x + "_rev")

    return p.concat([edges, rev_edges], axis=0)


edges_path = sys.argv[1]

edges = unpersist_or_exit(edges_path)
edges = add_reverse_edges(edges)

persist(edges, edges_path)