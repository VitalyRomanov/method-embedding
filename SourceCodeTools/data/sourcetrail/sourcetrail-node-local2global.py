import pandas as pd
import sys, os
from csv import QUOTE_NONNUMERIC

# TODO
# sometimes ast nodes are preferred

all_nodes = pd.read_csv(sys.argv[1], dtype={"id": int, "type": str, "serialized_name": str})
orig_nodes = pd.read_csv(sys.argv[2], dtype={"id": int, "type": str, "serialized_name": str})
local_map_path = sys.argv[3]

rev_id_map = dict(zip(all_nodes['serialized_name'].tolist(), all_nodes['id'].tolist()))
id_map = dict(zip(orig_nodes["id"].tolist(), map(lambda x: rev_id_map[x], orig_nodes["serialized_name"].tolist())))

orig_nodes['global_id'] = orig_nodes['id'].apply(lambda x: id_map[x])

orig_nodes[['id', 'global_id']].to_csv(local_map_path, index=False)