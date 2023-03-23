from os.path import join

import pandas as pd
from tqdm import tqdm

from SourceCodeTools.code.common import read_edges, read_nodes
from SourceCodeTools.code.data.file_utils import unpersist, persist
from SourceCodeTools.code.data.cubert_python_benchmarks.partitioning import add_splits


def get_node_labels(dataset_path):
    # nodes = unpersist(join(dataset_path, "common_nodes.json"))
    # edges = unpersist(join(dataset_path, "common_edges.json"))
    edges_path = join(dataset_path, "common_edges.json")
    nodes_path = join(dataset_path, "common_nodes.json")

    print("Reading comments")
    filecontent = unpersist(join(dataset_path, "common_filecontent.json"))
    id2comment = dict(zip(filecontent["id"], filecontent["comment"]))
    print("done")

    nodeid2name = {}
    nodeid2type = {}
    for nodes in tqdm(read_nodes(nodes_path, as_chunks=True), desc="Reading node info"):
        nodeid2name.update(dict(zip(nodes["id"], nodes["serialized_name"])))
        nodeid2type.update(dict(zip(nodes["id"], nodes["type"])))

    # edges.sort_values("file_id", inplace=True)

    del nodes, filecontent

    misuse_labels = []
    nodes_in_file = set()

    last_file_id = None
    total = 0
    for chunk_ind, edges in tqdm(enumerate(read_edges(edges_path, as_chunks=True)), desc="Extracting misuse edges"):
        for file_id, source_node_id, target_node_id in \
                edges[["file_id", "source_node_id", "target_node_id"]].values:
            if last_file_id != file_id:
                total += 1

                nodes_to_process = nodes_in_file
                nodes_in_file = set()
                file_id_to_process = last_file_id
                last_file_id = file_id

                nodes_in_file.add(source_node_id)
                nodes_in_file.add(target_node_id)

                if file_id_to_process is not None:
                    comment = id2comment[file_id_to_process]

                    if comment is not None:
                        if comment == "original":
                            misused_vars = set()
                        else:
                            misused_vars = set(comment.split(" ")[-1].strip("`").split("`->`"))
                            if len(misused_vars) != 2:
                                print(f"Error in {file_id}")
                                continue

                        for node_id in nodes_to_process:
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

            else:
                nodes_in_file.add(source_node_id)
                nodes_in_file.add(target_node_id)

    labels = pd.DataFrame.from_records(misuse_labels)
    partition = add_splits(labels[["src"]].rename({"src": "id"}, axis=1), 0.8)
    labels_ = labels.merge(partition, how="left", left_on="src", right_on="id")
    labels_.drop("id", axis=1, inplace=True)
    assert (labels_.isna().sum() == 0).all()
    labels_["id"] = labels_["src"]
    persist(labels_, join(dataset_path, "misuse_labels.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")

    args = parser.parse_args()

    get_node_labels(args.dataset_path)
