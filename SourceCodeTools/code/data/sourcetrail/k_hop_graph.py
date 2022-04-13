from collections import defaultdict
from os.path import join
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm

# from SourceCodeTools.code.data.dataset.Dataset import load_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("working_directory")
    parser.add_argument("k_hops", type=int)
    parser.add_argument("output")

    args = parser.parse_args()

    # nodes, edges = load_data(
    #     join(args.working_directory, "common_nodes.bz2"), join(args.working_directory, "common_edges.bz2")
    # )

    wd = Path(args.working_directory)

    edges = pd.read_csv(wd.joinpath("edges_train_dglke.tsv"), sep="\t", dtype={"src": "int32", "dst": "int32"})

    # edge_types = {}
    # edge_lists = {}
    # for s, d, t in edges[["src", "dst", "type"]].values:
    #     edge_types[(s,d)] = t
    #     if s not in edge_lists:
    #         edge_lists[s] = []
    #     edge_lists[s].append(d)

    g = nx.from_pandas_edgelist(
        edges, source="src", target="dst", create_using=nx.DiGraph, edge_attr="type"
    )

    from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping
    skip_edges = set(special_mapping.keys()) | set(special_mapping.values())
    # skip_edges = skip_edges | {"subword", "attr", "arg", "value"}

    def expand_edges(edges, node_id, view, edge_prefix, level=0):
        # edges = []
        if level < args.k_hops:
            if edge_prefix != "":
                edge_prefix += "|"
            for e in view:
                next_edge_type = view[e]["type"]
                if level > 0:
                    new_prefix = f"{level}_hop_connection"
                else:
                    new_prefix = edge_prefix + next_edge_type
                edges.append((node_id, e, new_prefix.rstrip("|")))
                if next_edge_type in skip_edges:
                    continue
                expand_edges(edges, node_id, g[e], new_prefix, level=level + 1)
                # edges.extend(expand_edges(node_id, g[e], new_prefix, level=level+1))
        return edges

    with open(wd.joinpath(f"{args.k_hops}_hop_edges_temp.tsv"), "w") as sink:
        sink.write("src\tdst\ttype\n")
        edges = []
        for node in tqdm(g.nodes):
            expand_edges(edges, node, g[node], "", level=0)
            for s,d,t in edges:
                sink.write(f"{s}\t{d}\t{t}\n")
            edges.clear()
            # edges.extend(expand_edges(node, g[node], "", level=0))

    del g

    parallel = defaultdict(list)
    with open(wd.joinpath(f"{args.k_hops}_hop_edges_temp.tsv"), "r") as source:
        for ind, line in tqdm(enumerate(source)):
            if ind == 0:
                continue
            s,d,t = line.strip().split('\t')
            parallel[(int(s),int(d))].append(t)

    with open(wd.joinpath(f"{args.k_hops}_hop_edges.tsv"), "w") as sink:
        sink.write("src\tdst\ttype\n")

        for (s,d), types in tqdm(parallel.items()):
            if len(types) > 1:
                t = sorted(types)[0]
            else:
                t = types[0]
            sink.write(f"{s}\t{d}\t{t}\n")

    # def expand_edges(node_id, s, dlist, edge_prefix, level=0):
    #     edges = []
    #     if level <= args.k_hops:
    #         if edge_prefix != "":
    #             edge_prefix += "|"
    #         for d in dlist:
    #             etype = edge_prefix + edge_types[(s,d)]
    #             edges.append((node_id, d, etype))
    #             edges.extend(expand_edges(node_id, d, edge_lists[d], etype, level=level+1))
    #     return edges
    #
    # edges = []
    # for node in tqdm(edge_lists):
    #     edges.extend(expand_edges(node, node, edge_lists[node], "", level=0))

    print()

if __name__ == "__main__":
    main()