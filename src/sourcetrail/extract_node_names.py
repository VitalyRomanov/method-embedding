import sys, os
import pandas as p
from csv import QUOTE_NONNUMERIC

def get_node_name(full_name):
    return full_name.split(".")[-1].split("___")[0]

nodes_path = sys.argv[1]

data = p.read_csv(nodes_path)
data = data[data['type'] != 262144]
data['src'] = data['id']
data['dst'] = data['serialized_name'].apply(get_node_name)

data[['src', 'dst']].to_csv(os.path.join(os.path.dirname(nodes_path), "node_names.csv"), index=False, quoting=QUOTE_NONNUMERIC)