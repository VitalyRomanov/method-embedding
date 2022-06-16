import json
from functools import partial
from os.path import join
from random import random

import pandas

from SourceCodeTools.code.common import read_edges, read_nodes
from SourceCodeTools.code.data.file_utils import persist

pandas.options.mode.chained_assignment = None

def add_splits(items, train_frac, restricted_id_pool=None):
    items = items.copy()

    def random_partition():
        r = random()
        if r < train_frac:
            return "train"
        elif r < train_frac + (1 - train_frac) / 2:
            return "val"
        else:
            return "test"

    import numpy as np
    # define partitioning
    masks = np.array([random_partition() for _ in range(len(items))])

    # create masks
    items["train_mask"] = masks == "train"
    items["val_mask"] = masks == "val"
    items["test_mask"] = masks == "test"

    if restricted_id_pool is not None:
        # if `restricted_id_pool` is provided, mask all nodes not in `restricted_id_pool` negatively
        to_keep = items.eval("id in @restricted_ids", local_dict={"restricted_ids": restricted_id_pool})
        items["train_mask"] = items["train_mask"] & to_keep
        items["test_mask"] = items["test_mask"] & to_keep
        items["val_mask"] = items["val_mask"] & to_keep

    return items


def subgraph_partitioning(path_to_dataset, partition_column, train_frac=0.8):

    get_path = partial(join, path_to_dataset)

    # nodes = read_nodes(get_path("common_nodes.json.bz2"))
    edges = read_edges(get_path("common_edges.json.bz2"))

    subgraph_ids = add_splits(
        edges[[partition_column]].dropna(axis=0).drop_duplicates(), train_frac=train_frac
    ).rename({partition_column: "id"}, axis=1)

    valid_subgraph_ids = set(subgraph_ids["id"])

    persist(subgraph_ids, get_path("subgraph_partition.json"))

    edges = edges[["source_node_id", "target_node_id", partition_column]].sort_values(partition_column)

    last_subgraph_id = -1
    pool = set()

    with open(get_path("subgraph_mapping.json"), "w") as subgraph_mapping:
        for src, dst, subgraph_id in edges[["source_node_id", "target_node_id", partition_column]].values:
            if subgraph_id in valid_subgraph_ids:
                if subgraph_id != last_subgraph_id and last_subgraph_id != -1:
                    record = json.dumps({
                        "subgraph_id": last_subgraph_id,
                        "node_ids": list(pool)
                    })
                    subgraph_mapping.write(f"{record}\n")

                    pool.clear()

                pool.add(src)
                pool.add(dst)
                last_subgraph_id = subgraph_id



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("subgraph_column")

    args = parser.parse_args()

    subgraph_partitioning(args.dataset_path, args.subgraph_column)