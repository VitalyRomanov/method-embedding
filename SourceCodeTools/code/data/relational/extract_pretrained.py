from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from SourceCodeTools.code.common import read_nodes
from SourceCodeTools.code.data.GraphStorage import OnDiskGraphStorage
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types


def is_int(val):
    try:
        int(val)
        return True
    except:
        return False


def get_relationship_type_mapping(path):
    types = pd.read_csv(path, sep="\t", header=None, names=["id", "name"])
    return dict(zip(types["name"], types["id"]))


def get_node_id_mapping(path, shared_node_ids, shared_global_node_ids):
    node_ids = {}
    global_node_ids = {}
    node_type_ids = {}

    with open(path, "r") as source:
        for line in source:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pid, nodeid = parts
                pid = int(pid)
                nodeid = nodeid.strip("\"")
                if is_int(nodeid):
                    nodeid = int(nodeid)
                    if nodeid in shared_node_ids:
                        node_ids[nodeid] = pid
                    elif nodeid in shared_global_node_ids:
                        global_node_ids[nodeid] = pid
                else:
                    node_type_ids[nodeid] = pid

    return node_ids, global_node_ids, node_type_ids


def get_edge_type_params(pretrained_path, param_path):
    relation_type_ids = get_relationship_type_mapping(pretrained_path.joinpath("relations.tsv"))
    params = np.load(param_path)
    return {
        key: params[val] for key, val in relation_type_ids.items()
    }


def get_node_params(pretrained_path, param_path, nodes_path):
    shared_node_types = {"Op", "#attr#", "#keyword#", "subword"}
    global_nodes_types = set(node_types.values())
    shared_node_ids = dict()
    shared_global_node_ids = dict()

    for nodes in tqdm(read_nodes(nodes_path, as_chunks=True)):
        for node_id, node_type, name in zip(nodes["id"], nodes["type"], nodes["serialized_name"]):
            if node_type in shared_node_types:
                shared_node_ids[node_id] = (node_type, name)
            elif node_type in global_nodes_types:
                shared_global_node_ids[node_id] = (node_type, name)

    node_ids, global_node_ids, node_type_ids = get_node_id_mapping(pretrained_path.joinpath("entities.tsv"), shared_node_ids, shared_global_node_ids)

    params = np.load(param_path, mmap_mode="r")
    node_type_params = {
        key: np.array(params[val]) for key, val in node_type_ids.items()
    }

    shared_node_params = {
        shared_node_ids[key]: np.array(params[val]) for key, val in node_ids.items()
    }

    global_node_params = {
        shared_global_node_ids[key]: np.array(params[val]) for key, val in global_node_ids.items()
    }

    return node_type_params, global_node_params, shared_node_params


def store_params(output_path, params, name, save_tsv=False):
    order = []
    vectors = []
    for key, val in params.items():
        order.append(key)
        vectors.append(val)

    vectors = np.vstack(vectors)

    output_path.mkdir(exist_ok=True)

    with open(output_path.joinpath(f"{name}_order"), "w") as order_sink:
        for o in order:
            if isinstance(o, tuple):
                order_sink.write(f"{o[0]}\t{o[1]}\n")
            else:
                order_sink.write(f"{o}\n")

    np.save(output_path.joinpath(f"{name}_params.npy"), vectors)

    if save_tsv:
        with open(output_path.joinpath(f"{name}_params.tsv"), "w") as params_sink:
            for v in vectors:
                p_string = '\t'.join(map(str, list(v)))
                params_sink.write(f"{p_string}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes_path")
    parser.add_argument("pretrained_path")
    parser.add_argument("edge_type_params")
    parser.add_argument("node_params")
    parser.add_argument("output_path")

    args = parser.parse_args()

    pretrained_path = Path(args.pretrained_path)

    edge_type_params = get_edge_type_params(pretrained_path, args.edge_type_params)
    node_type_params, global_node_params, shared_node_params = get_node_params(pretrained_path, args.node_params, args.nodes_path)

    store_params(Path(args.output_path), edge_type_params, "edge_types", save_tsv=True)
    store_params(Path(args.output_path), node_type_params, "node_types", save_tsv=True)
    store_params(Path(args.output_path), global_node_params, "global_nodes", save_tsv=False)
    store_params(Path(args.output_path), shared_node_params, "shared_nodes", save_tsv=False)




if __name__ == "__main__":
    main()