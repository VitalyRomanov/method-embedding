from os.path import join

import networkx as nx
from tqdm import tqdm

from SourceCodeTools.code.data.dataset.Dataset import load_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("working_directory")
    parser.add_argument("k_hops", type=int)
    parser.add_argument("output")

    args = parser.parse_args()

    nodes, edges = load_data(
        join(args.working_directory, "common_nodes.bz2"), join(args.working_directory, "common_edges.bz2")
    )

    edge_types = {}
    edge_lists = {}
    for s, d, t in edges[["src", "dst", "type"]].values:
        edge_types[(s,d)] = t
        if s not in edge_lists:
            edge_lists[s] = []
        edge_lists[s].append(d)

    g = nx.from_pandas_edgelist(
        edges, source="src", target="dst", create_using=nx.DiGraph, edge_attr="type"
    )

    # def expand_edges(node_id, view, edge_prefix, level=0):
    #     edges = []
    #     if level <= args.k_hops:
    #         if edge_prefix != "":
    #             edge_prefix += "|"
    #         for e in view:
    #             edges.append((node_id, e, edge_prefix + view[e]["type"]))
    #             edges.extend(expand_edges(node_id, g[e], edge_prefix + view[e]["type"], level=level+1))
    #     return edges
    #
    # edges = []
    # for node in tqdm(g.nodes):
    #     edges.extend(expand_edges(node, g[node], "", level=0))

    def expand_edges(node_id, s, dlist, edge_prefix, level=0):
        edges = []
        if level <= args.k_hops:
            if edge_prefix != "":
                edge_prefix += "|"
            for d in dlist:
                etype = edge_prefix + edge_types[(s,d)]
                edges.append((node_id, d, etype))
                edges.extend(expand_edges(node_id, d, edge_lists[d], etype, level=level+1))
        return edges

    edges = []
    for node in tqdm(edge_lists):
        edges.extend(expand_edges(node, node, edge_lists[node], "", level=0))

    print()

if __name__ == "__main__":
    main()