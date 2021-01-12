from SourceCodeTools.data.sourcetrail.file_utils import *
from SourceCodeTools.data.sourcetrail.common import custom_tqdm

import networkx as nx
from collections import Counter


def get_node2mention(nodes):
    return dict(zip(nodes['id'], nodes['mentioned_in']))


def compute_edge_mentions(edges, node2mention):
    edges['src_mentioned_in'] = edges['source_node_id'].apply(lambda id_: node2mention[id_])
    edges['dst_mentioned_in'] = edges['target_node_id'].apply(lambda id_: node2mention[id_])


def get_function_edges(edges, function_nodes):
    function_edges = edges[
        edges['source_node_id'].apply(lambda id_: id_ in function_nodes)
    ]
    function_edges = function_edges[
        function_edges['target_node_id'].apply(lambda id_: id_ in function_nodes)
    ]
    return function_edges


def get_function_edges(edges, func_id):
    function_edges = edges.query(f"src_mentioned_in == {func_id} and dst_mentioned_in == {func_id}")
    return function_edges


def compute_diameter(edges):
    try:
        g = nx.convert_matrix.from_pandas_edgelist(edges, source='source_node_id', target='target_node_id')
        return nx.algorithms.distance_measures.diameter(g)
    except nx.exception.NetworkXError:
        return "inf"


def find_diameter_distribution(nodes, edges):
    diameters = []
    function_nodes = nodes.groupby("mentioned_in")
    node2mention = get_node2mention(nodes)
    compute_edge_mentions(edges, node2mention)
    for func_id, func_nodes in custom_tqdm(function_nodes, total=len(function_nodes), message="Computing diameters"):
        # nodes_in_function = set(nodes['id'].tolist())
        function_edges = get_function_edges(edges, func_id)
        if len(function_edges) == 0:
            continue
        diameter = compute_diameter(function_edges)
        diameters.append(diameter)

    diameters_dist = Counter(diameters)
    diam_count = []
    for diam, count in diameters_dist.most_common(len(diameters_dist)):
        diam_count.append({
            'diameter': diam,
            'count': count
        })

    if len(diam_count) > 0:
        return pd.DataFrame(diam_count)
    else:
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nodes")
    parser.add_argument("edges")

    args = parser.parse_args()

    nodes = unpersist(args.nodes).astype({'mentioned_in': 'Int32'})
    edges = unpersist(args.edges)

    diameters = find_diameter_distribution(nodes, edges)

    if diameters is not None:
        persist(diameters, os.path.join(os.path.dirname(args.nodes), "diameter_counts.csv"))