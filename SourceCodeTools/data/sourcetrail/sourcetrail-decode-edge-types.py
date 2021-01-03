from SourceCodeTools.data.sourcetrail.sourcetrail_types import edge_types
from SourceCodeTools.data.sourcetrail.file_utils import *

import pandas as p
import sys
import os


edges_path = sys.argv[1]

edges = unpersist_or_exit(edges_path, exit_message="Sourcetrail edges are empty", dtype={"id": int, "type": int, "source_node_id": int, "target_node_id": int})
# edges = read_csv(edges_path, dtype={"id": int, "type": int, "source_node_id": int, "target_node_id": int})
edges['type'] = edges['type'].apply(lambda x: edge_types[x])

edges = edges.astype({"id": int, "type": str, "source_node_id": int, "target_node_id": int})
persist(edges, os.path.join(os.path.dirname(edges_path), filenames["edges"]))
