import logging
import os
from os.path import isdir, join, isfile

from SourceCodeTools.code.data.sourcetrail.Dataset import load_data, compact_property, SourceGraphDataset

import argparse

from SourceCodeTools.code.data.sourcetrail.file_utils import unpersist, persist


def get_paths(dataset_path, use_extra_objectives):
    extra_objectives = ["node_names.bz2", "common_function_variable_pairs.bz2", "common_call_seq.bz2", "type_annotations.bz2"]

    largest_component = join(dataset_path, "largest_component")
    if isdir(largest_component):
        logging.info("Using graph from largest_component directory")
        nodes_path = join(largest_component, "nodes.bz2")
        edges_path = join(largest_component, "edges.bz2")
    else:
        nodes_path = join(dataset_path, "common_nodes.bz2")
        edges_path = join(dataset_path, "common_edges.bz2")


    extra_paths = list(map(
        lambda file: file if use_extra_objectives and isfile(file) else None,
        (join(dataset_path, objective) for objective in extra_objectives)
    ))

    return nodes_path, edges_path, extra_paths


def filter_relevant(data, node_ids):
    return data.query("src in @allowed", local_dict={"allowed": node_ids})


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_path', default=None, help='Path to the dataset')
    parser.add_argument('output_path', default=None, help='')
    parser.add_argument("--extra_objectives", action="store_true", default=False)
    parser.add_argument("--eval_frac", dest="eval_frac", default=0.05, type=float)

    args = parser.parse_args()

    nodes_path, edges_path, extra_paths = get_paths(
        args.dataset_path, use_extra_objectives=args.extra_objectives
    )

    dataset = SourceGraphDataset(
        # args.node_path, args.edge_path,
        args.dataset_path,
        label_from="type",
        use_node_types=False,
        use_edge_types=True,
        filter=None,
        self_loops=False,
        train_frac=0.99,
        tokenizer_path=None,
        random_seed=42,
        min_count_for_objectives=2,
        no_global_edges=True,
        remove_reverse=False,
        custom_reverse=None,
        package_names=None,
        restricted_id_pool=None,
        use_ns_groups=False
    )

    # nodes, edges = load_data(nodes_path, edges_path)
    # nodes, edges, holdout = SourceGraphDataset.holdout(nodes, edges)
    nodes = dataset.nodes
    edges = dataset.edges.astype({"src": 'str', "dst": "str", "type": 'str'})[['src', 'dst', 'type']]
    holdout = dataset.heldout.astype({"src": 'str', "dst": "str", "type": 'str'})[['src', 'dst', 'type']]

    node2graph_id = compact_property(nodes['id'])
    nodes['global_graph_id'] = nodes['id'].apply(lambda x: node2graph_id[x])

    node_ids = set(nodes['id'].unique())

    if args.extra_objectives:
        for objective_path in extra_paths:
            data = unpersist(objective_path)
            data = filter_relevant(data, node_ids)
            data["type"] = objective_path.split(".")[0]
            edges = edges.append(data)

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    edges = edges[['src','dst','type']]
    eval_sample = edges.sample(frac=args.eval_frac)

    persist(nodes, join(args.output_path, "nodes_dglke.csv"))
    persist(edges, join(args.output_path, "edges_train_dglke.tsv"), header=False, sep="\t")
    persist(edges, join(args.output_path, "edges_train_node2vec.tsv"), header=False, sep=" ")
    persist(eval_sample, join(args.output_path, "edges_eval_dglke.tsv"), header=False, sep="\t")
    persist(eval_sample, join(args.output_path, "edges_eval_node2vec.tsv"), header=False, sep=" ")
    persist(holdout, join(args.output_path, "edges_eval_dglke_10000.tsv"), header=False, sep="\t")
    persist(holdout, join(args.output_path, "edges_eval_node2vec_10000.tsv"), header=False, sep=" ")




if __name__ == "__main__":
    main()