import pandas
import numpy as np

def load_data(node_path, edge_path):
    nodes = pandas.read_csv(node_path)
    edges = pandas.read_csv(edge_path)

    # nodes['new_id'] = list(range(nodes.shape[0]))
    #
    # node_id_map = dict(zip(nodes['id'].values, nodes['new_id'].values))
    #
    # id2new = lambda id: node_id_map[id]
    #
    # edges['new_src'] = edges['source_node_id'].apply(id2new)
    # edges['new_dst'] = edges['target_node_id'].apply(id2new)
    #
    # nodes_ = nodes[['new_id', 'type', 'serialized_name']].rename(mapper={
    #     'new_id': "id",
    #     'serialized_name': 'name'
    # }, axis=1)
    #
    # edges_ = edges[['type', 'new_src', 'new_dst']].rename(mapper={
    #     'new_src': 'src',
    #     'new_dst': 'dst'
    # }, axis=1)

    nodes_ = nodes.rename(mapper={
        'serialized_name': 'name'
    }, axis=1)

    edges_ = edges.rename(mapper={
        'source_node_id': 'src',
        'target_node_id': 'dst'
    }, axis=1)

    return nodes_, edges_


