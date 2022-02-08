from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import edge_types
from SourceCodeTools.code.data.file_utils import *

import sys
import os


def decode_edge_types(edges_path, exit_if_empty=True):
    if exit_if_empty:
        edges = unpersist_or_exit(edges_path, exit_message="Sourcetrail edges are empty",
                                  dtype={"id": int, "type": int, "source_node_id": int, "target_node_id": int})
    else:
        edges = unpersist_if_present(edges_path, dtype={"id": int, "type": int, "source_node_id": int, "target_node_id": int})

    if edges is None:
        return None

    edges['type'] = edges['type'].apply(lambda x: edge_types[x])

    edges = edges.astype({"id": int, "type": str, "source_node_id": int, "target_node_id": int})

    if len(edges) > 0:
        return edges
    else:
        return None


if __name__ == "__main__":
    edges_path = sys.argv[1]
    edges = decode_edge_types(edges_path)

    if edges is not None:
        persist(edges, os.path.join(os.path.dirname(edges_path), filenames["edges"]))
