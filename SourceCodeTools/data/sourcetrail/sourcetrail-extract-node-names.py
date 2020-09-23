    #%%
import math
import sys, os
import pandas as p
from csv import QUOTE_NONNUMERIC

def get_node_name(full_name):
    """
    Used for processing java nodes
    """
    return full_name.split(".")[-1].split("___")[0]


def extract_and_write_node_names(nodes_path, out_path):

    data = p.read_csv(nodes_path)
    # some cells are empty, probably because of empty strings in AST
    data = data.dropna(axis=0)
    data = data[data['type'] != 262144]
    data['src'] = data['id']
    data['dst'] = data['serialized_name'].apply(get_node_name)

    counts = data['dst'].value_counts()

    data['counts'] = data['dst'].apply(lambda x: counts[x])
    data = data.query("counts > 1")

    data[['src', 'dst']].to_csv(out_path, index=False, quoting=QUOTE_NONNUMERIC)

if __name__ == "__main__":
    nodes_path = sys.argv[1]
    out_path = sys.argv[2]

    extract_and_write_node_names(nodes_path, out_path)

