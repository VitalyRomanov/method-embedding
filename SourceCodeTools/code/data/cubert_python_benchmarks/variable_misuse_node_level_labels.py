from os.path import join

import pandas as pd
from tqdm import tqdm

from SourceCodeTools.code.data.file_utils import unpersist, persist


def get_node_labels(dataset_path):
    filecontent = unpersist(join(dataset_path, "common_filecontent.json"))
    nodes = unpersist(join(dataset_path, "common_nodes.json"))
    edges = unpersist(join(dataset_path, "common_edges.json"))

    id2comment = dict(zip(filecontent["id"], filecontent["comment"]))
    nodeid2name = dict(zip(nodes["id"], nodes["serialized_name"]))
    nodeid2type = dict(zip(nodes["id"], nodes["type"]))

    edges.sort_values("file_id", inplace=True)

    del nodes, filecontent

    misuse_labels = []

    for file_id, file_edges in tqdm(edges.groupby("file_id")):
        comment = id2comment[file_id]

        if comment is None:
            continue

        if comment.startswith("original"):
            continue

        misused_vars = set(comment.split(" ")[-1].strip("`").split("`->`"))

        if len(misused_vars) != 2:
            print(f"Error in {file_id}")
            continue

        all_nodes = set(file_edges["source_node_id"].append(file_edges["target_node_id"]))

        for node_id in all_nodes:
            if nodeid2type[node_id] == "mention":
                node_name = nodeid2name[node_id].split("@")[0]
                if node_name in misused_vars:
                    misuse_labels.append({
                        "src": node_id,
                        "dst": "misused"
                    })
                else:
                    misuse_labels.append({
                        "src": node_id,
                        "dst": "correct"
                    })

    labels = pd.DataFrame.from_records(misuse_labels)

    persist(labels, join(dataset_path, "misuse_labels.json"))





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")

    args = parser.parse_args()

    get_node_labels(args.dataset_path)