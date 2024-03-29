import logging
import os
from collections import defaultdict
from os.path import isdir, join, isfile
from random import random

import numpy as np
import pandas as pd

from SourceCodeTools.code.common import read_edges, read_nodes
from SourceCodeTools.code.data.dataset.Dataset import load_data, compact_property, SourceGraphDataset

import argparse

from SourceCodeTools.code.data.file_utils import unpersist, persist
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types


def get_paths(dataset_path, use_extra_objectives):
    extra_objectives = ["node_names.json", "common_function_variable_pairs.json", "common_call_seq.json", "type_annotations.json"]

    largest_component = join(dataset_path, "largest_component")
    if isdir(largest_component):
        logging.info("Using graph from largest_component directory")
        nodes_path = join(largest_component, "common_nodes.json")
        edges_path = join(largest_component, "common_edges.json")
    else:
        nodes_path = join(dataset_path, "common_nodes.json")
        edges_path = join(dataset_path, "common_edges.json")

    extra_paths = list(filter(lambda x: x is not None, map(
        lambda file: file if use_extra_objectives and isfile(file) else None,
        (join(dataset_path, objective) for objective in extra_objectives)
    )))

    return nodes_path, edges_path, extra_paths


def filter_relevant(data, node_ids):
    return data.query("src in @allowed", local_dict={"allowed": node_ids})


def add_counts(counter, node_ids):
    for node_id in node_ids:
        counter[node_id] += 1


def count_degrees(edges_path):
    counter = defaultdict(lambda: 0)
    for edges in read_edges(edges_path, as_chunks=True):
        add_counts(counter, edges["source_node_id"])
        add_counts(counter, edges["target_node_id"])

    return counter


def count_with_occurrence(counter, min_occurrence):
    c = 0
    for id_, count in counter.items():
        if count > min_occurrence:
            c += 1
    return c


def get_writing_mode(is_csv, first_written):
    kwargs = {}
    if first_written is True:
        kwargs["mode"] = "a"
        if is_csv:
            kwargs["header"] = False
    return kwargs


def do_holdout(edges_path, output_path, node_descriptions, holdout_size=10000, min_count=2):

    counter = count_degrees(edges_path)
    num_valid_candidates = count_with_occurrence(counter, min_count)

    frac = holdout_size / num_valid_candidates

    # temp_edges = join(os.path.dirname(edges_path), "temp_" + os.path.basename(edges_path))
    out_edges_path = join(output_path, "edges_train_dglke.tsv")
    out_holdout_path = join(output_path, "edges_eval_dglke_10000.tsv")
    is_csv = True

    first_edges = False
    first_holdout = False

    total_edges = 0
    total_holdout = 0

    seen = set()

    for edges in read_edges(edges_path, as_chunks=True):
        edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1, inplace=True)
        edges = edges[['src', 'dst', 'type']]

        sufficient_count = edges["src"].apply(lambda x: counter[x] > min_count) & \
                           edges["dst"].apply(lambda x: counter[x] > min_count)

        definitely_keep = edges[~sufficient_count]
        probably_keep = edges[sufficient_count]

        probably_holdout_mask = np.array([random() < frac for _ in range(len(probably_keep))])

        probably_holdout = probably_keep[probably_holdout_mask]

        definitely_holdout_mask = []
        for src, dst, type_ in probably_holdout.values:
            if counter[src] > min_count and counter[dst] > min_count:
                definitely_holdout_mask.append(True)
                counter[src] -= 1
                counter[dst] -= 1
            else:
                definitely_holdout_mask.append(False)

        definitely_holdout_mask = np.array(definitely_holdout_mask)

        definitely_holdout = probably_holdout[definitely_holdout_mask]
        definitely_keep = pd.concat([definitely_keep, probably_keep[~probably_holdout_mask], probably_holdout[~definitely_holdout_mask]])

        total_edges += len(definitely_keep)
        total_holdout += len(definitely_holdout)

        def apply_description(edges):
            edges["src"] = edges["src"].apply(node_descriptions.get)
            edges["dst"] = edges["dst"].apply(node_descriptions.get)
            return edges

        def write_filtered(table, path, first_written):
            with_description = apply_description(table)

            with_description.drop_duplicates(inplace=True)

            reprs = [(src, dst, type_) for src, dst, type_ in with_description.values]

            seen_mask = np.array(list(map(lambda x: x in seen, reprs)))

            with_description = with_description.loc[~seen_mask]
            seen.update(reprs)

            kwargs = get_writing_mode(is_csv, first_written)
            persist(with_description, path, sep="\t", **kwargs)

        write_filtered(definitely_keep, out_edges_path, first_edges)
        first_edges = True

        if len(definitely_holdout) > 0:
            write_filtered(definitely_holdout, out_holdout_path, first_holdout)
            first_holdout = True

    return counter, total_edges, total_holdout


def add_extra_objectives(extra_paths, output_path, node_ids):
    out_edges_path = join(output_path, "edges_train_dglke.tsv")

    total_extra = 0

    for objective_path in extra_paths:
        data = unpersist(objective_path)
        data = filter_relevant(data, node_ids)
        data["type"] = data["type"].split(".")[0]
        data = data[["src", "dst", "type"]]
        raise NotImplementedError()
        kwargs = get_writing_mode(is_csv=True, first_written=True)
        persist(data, out_edges_path, sep="\t", **kwargs)  # write_filtered

        total_extra += len(data)

    return total_extra


def save_eval(output_dir, eval_frac):
    eval_path = join(output_dir, "edges_eval_dglke.tsv")

    total_eval = 0

    for ind, edges in enumerate(read_edges(join(output_dir, "edges_train_dglke.tsv"), as_chunks=True)):
        eval = edges.sample(frac=eval_frac)
        if len(eval) > 0:
            kwargs = get_writing_mode(is_csv=True, first_written=ind != 0)
            persist(eval, eval_path, sep="\t", **kwargs)

            total_eval += len(eval)

    return total_eval


def get_node_descriptions(nodes_path, distinct_node_types):

    description = {}

    for nodes in read_nodes(nodes_path, as_chunks=True):
        transform_mask = nodes.eval("type in @distinct_node_types", local_dict={"distinct_node_types": distinct_node_types})

        nodes.loc[transform_mask, "transformed"] = nodes.loc[transform_mask, "id"].astype("string")
        nodes.loc[~transform_mask, "transformed"] = nodes.loc[~transform_mask, "type"]

        for id, desc in nodes[["id", "transformed"]].values:
            description[id] = desc

    return description


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('dataset_path', default=None, help='Path to the dataset')
    parser.add_argument('output_path', default=None, help='')
    parser.add_argument("--extra_objectives", action="store_true", default=False)
    parser.add_argument("--eval_frac", dest="eval_frac", default=0.05, type=float)

    distinct_node_types = set(node_types.values()) | {
        "FunctionDef", "mention", "Op", "#attr#", "#keyword#", "subword"
    }

    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    nodes_path, edges_path, extra_paths = get_paths(
        args.dataset_path, use_extra_objectives=args.extra_objectives
    )

    node_descriptions = get_node_descriptions(nodes_path, distinct_node_types)

    counter, total_edges, total_holdout = do_holdout(edges_path, args.output_path, node_descriptions)

    total_extra = add_extra_objectives(extra_paths, args.output_path, set(counter.keys()))

    temp_edges = join(args.output_path, "temp_common_edges.tsv")

    total_eval = save_eval(args.output_path, args.eval_frac)

    # nodes, edges = load_data(nodes_path, edges_path)
    # nodes, edges, holdout = SourceGraphDataset.holdout(nodes, edges)
    # edges = edges.astype({"src": 'str', "dst": "str", "type": 'str'})[['src', 'dst', 'type']]
    # holdout = holdout.astype({"src": 'str', "dst": "str", "type": 'str'})[['src', 'dst', 'type']]
    #
    # node2graph_id = compact_property(nodes['id'])
    # nodes['global_graph_id'] = nodes['id'].apply(lambda x: node2graph_id[x])
    #
    # node_ids = set(nodes['id'].unique())
    #
    # if args.extra_objectives:
    #     for objective_path in extra_paths:
    #         data = unpersist(objective_path)
    #         data = filter_relevant(data, node_ids)
    #         data["type"] = objective_path.split(".")[0]
    #         edges = edges.append(data)



    # edges = edges[['src','dst','type']]
    # eval_sample = edges.sample(frac=args.eval_frac)
    #
    # persist(nodes, join(args.output_path, "nodes_dglke.csv"))
    # persist(edges, join(args.output_path, "edges_train_dglke.tsv"), header=False, sep="\t")
    # persist(edges, join(args.output_path, "edges_train_node2vec.tsv"), header=False, sep=" ")
    # persist(eval_sample, join(args.output_path, "edges_eval_dglke.tsv"), header=False, sep="\t")
    # persist(eval_sample, join(args.output_path, "edges_eval_node2vec.tsv"), header=False, sep=" ")
    # persist(holdout, join(args.output_path, "edges_eval_dglke_10000.tsv"), header=False, sep="\t")
    # persist(holdout, join(args.output_path, "edges_eval_node2vec_10000.tsv"), header=False, sep=" ")




if __name__ == "__main__":
    main()