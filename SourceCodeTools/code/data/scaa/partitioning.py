from functools import partial
from os.path import join
from random import random

import numpy as np

from SourceCodeTools.code.common import read_edges, read_nodes
from SourceCodeTools.code.data.file_utils import persist, unpersist


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
        to_keep = items.eval("id in @restricted_ids",
                             local_dict={"restricted_ids": restricted_id_pool})
        items["train_mask"] = items["train_mask"] & to_keep
        items["test_mask"] = items["test_mask"] & to_keep
        items["val_mask"] = items["val_mask"] & to_keep

    return items


def subgraph_partitioning(path_to_dataset, partition_column, train_frac=0.7):

    get_path = partial(join, path_to_dataset)

    # nodes = read_nodes(get_path("common_nodes.json.bz2"))
    edges = read_edges(get_path("common_edges.json.bz2"))
    filecontent = unpersist(get_path("common_filecontent.json.bz2"))

    def task_partition(task):
        # 9 = 1 + 1 + 5
        if task < 6:
            return "train"
        elif task < 7:
            return "val"
        else:
            return "test"

    task2split = dict(zip(filecontent[partition_column], [
                      task_partition(task) for task in filecontent[partition_column]]))
    file_id2split = np.array([task2split[task]
                             for task in filecontent[partition_column]])

    subgraph_ids = filecontent[["id"]]

    subgraph_ids["train_mask"] = file_id2split == "train"
    subgraph_ids["val_mask"] = file_id2split == "val"
    subgraph_ids["test_mask"] = file_id2split == "test"

    persist(subgraph_ids, get_path("subgraph_partition.json"))


if __name__ == "__main__":
    # not scalable, but works
    path = './examples/one_vs_10/with_ast'
    column = 'task'
    subgraph_partitioning(path, column)
    print('Done')
