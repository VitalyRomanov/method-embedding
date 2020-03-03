import sys
import pandas

def load_data(node_path, edge_path):
    nodes = pandas.read_csv(node_path)
    edges = pandas.read_csv(edge_path)

    nodes['new_id'] = list(range(nodes.shape[0]))

    node_id_map = dict(zip(nodes['id'].values, nodes['new_id'].values))

    id2new = lambda id: node_id_map[id]

    edges['new_src'] = edges['source_node_id'].apply(id2new)
    edges['new_dst'] = edges['target_node_id'].apply(id2new)