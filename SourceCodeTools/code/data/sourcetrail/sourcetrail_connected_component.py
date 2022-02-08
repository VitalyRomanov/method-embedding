import networkx as nx
import sys

from SourceCodeTools.code.data.file_utils import *


def main():
    nodes_path = sys.argv[1]
    edges_path = sys.argv[2]
    out_path = sys.argv[3]

    in_count_path = os.path.join(out_path, "nodes_in_count.csv.bz2")
    out_count_path = os.path.join(out_path, "nodes_out_count.csv.bz2")
    node_type_count_path = os.path.join(out_path, "node_type_count.csv")
    edge_type_count_path = os.path.join(out_path, "edge_type_count.csv")

    nodes = unpersist(nodes_path)
    edges = unpersist(edges_path)

    g = nx.from_pandas_edgelist(edges, source="source_node_id",
                                target="target_node_id", create_using=nx.DiGraph,)

    print(f"Loading graph")
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")

    g.remove_nodes_from(list(nx.isolates(g)))

    connected_nodes = max(nx.weakly_connected_components(g), key=len)
    g = g.subgraph(connected_nodes)

    print(f"Largest component")
    print(f"Nodes: {g.number_of_nodes()}, Edges: {g.number_of_edges()}")

    in_count = dict(g.in_degree(connected_nodes))
    out_count = dict(g.out_degree(connected_nodes))

    nodes['in_count'] = nodes['id'].apply(lambda x: in_count.get(x, -1))
    nodes['out_count'] = nodes['id'].apply(lambda x: out_count.get(x, -1))

    nodes[['serialized_name', 'in_count']]\
        .sort_values(by=["in_count"], ascending=[False])\
        .to_csv(in_count_path, index=False)
    nodes[['serialized_name', 'out_count']]\
        .sort_values(by=["out_count"], ascending=[False])\
        .to_csv(out_count_path, index=False)

    edges['edge_repr'] = list(zip(edges['source_node_id'], edges['target_node_id']))
    connected_edges = set(g.edges)
    connected_edges_with_data = edges[
        edges['edge_repr'].apply(lambda edge: edge in connected_edges)
    ].drop('edge_repr', axis=1)

    connected_nodes_with_data = nodes[
        nodes['id'].apply(lambda id_: id_ in connected_nodes)
    ]

    write_nodes(connected_nodes_with_data, out_path)
    write_edges(connected_edges_with_data, out_path)

    connected_nodes_with_data[['type', 'id']]\
        .groupby("type").count()\
        .rename({'id': 'count'}, axis=1)\
        .reset_index()\
        .sort_values(by=["count"], ascending=[False])\
        .to_csv(node_type_count_path, index=False)
    connected_edges_with_data[['type', 'id']] \
        .groupby("type").count() \
        .rename({'id': 'count'}, axis=1) \
        .reset_index() \
        .sort_values(by=["count"], ascending=[False]) \
        .to_csv(edge_type_count_path, index=False)


if __name__ == "__main__":
    main()
