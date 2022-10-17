from ast import literal_eval
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.cubert_python_benchmarks.partitioning import add_splits
from SourceCodeTools.code.data.file_utils import unpersist, persist


def create_edge_labels(dataset_directory):

    dataset = Path(dataset_directory)
    filecontent_path = dataset.joinpath("common_filecontent.json.bz2")
    nodes_path = dataset.joinpath("common_nodes.json.bz2")
    edges_path = dataset.joinpath("common_edges.json.bz2")

    def get_node_names():
        id2node_name = dict()
        id2node_type = dict()
        for nodes in read_nodes(nodes_path, as_chunks=True):
            id2node_name.update(dict(zip(nodes["id"], nodes["serialized_name"])))
            id2node_type.update(dict(zip(nodes["id"], nodes["type"])))
        return id2node_name, id2node_type

    id2node_name, id2node_type = get_node_names()

    def get_misuse_spans():
        filecontent = unpersist(filecontent_path)

        def try_get_misused_name(comment):
            parts = comment.split("`")
            try:
                return parts[-2]
            except:
                return None  # this happens when function does not have a misuse

        return dict(zip(filecontent["id"], filecontent["misuse_span"].map(literal_eval))), dict(zip(filecontent["id"], filecontent["comment"].map(try_get_misused_name))), dict(zip(filecontent["id"], filecontent["partition"]))

    file_id2misuse_span, file_id2replacement_var, file_id2partition = get_misuse_spans()

    file_id2incorrect_edge = set()
    skipped = []
    file_id2labeled_edges = []

    last_file_id = None
    total = 0
    for chunk_ind, edges in tqdm(enumerate(read_edges(edges_path, as_chunks=True)), desc="Extracting misuse edges"):
        edges = edges.astype({"offset_start": "Int32", "offset_end": "Int32"})
        for file_id, source_node_id, target_node_id, type, offset_start, offset_end in \
                edges[["file_id", "source_node_id", "target_node_id", "type", "offset_start", "offset_end"]].values:
            if last_file_id != file_id:
                total += 1
                if last_file_id is not None and last_file_id not in file_id2incorrect_edge:
                    # print(f"Did not find edge for {last_file_id}")
                    skipped.append(last_file_id)
                    # print(f"Skipped {len(skipped)} files out of {total}")
                last_file_id = file_id

            if not pd.isna(offset_start):
                # file_id = edge["file_id"]
                # src = edge["source_node_id"]
                # dst = edge["target_node_id"]
                # type = edge["type"]
                needed_span = tuple(file_id2misuse_span[file_id])
                given_span = (offset_start, offset_end)
                if needed_span == given_span:
                    replacement = file_id2replacement_var[file_id]
                    if replacement is not None:
                        node_name = id2node_name[source_node_id]
                        assert node_name.startswith(replacement)
                        file_id2incorrect_edge.add(file_id)
                        file_id2labeled_edges.append({
                            "src": source_node_id,
                            "dst": target_node_id,
                            "type": type,
                            "file_id": file_id,
                            "label": "misuse"
                        })
                else:
                    node_type = id2node_type[source_node_id]
                    if node_type == "mention":
                        file_id2labeled_edges.append({
                            "src": source_node_id,
                            "dst": target_node_id,
                            "type": type,
                            "file_id": file_id,
                            "label": "correct"
                        })

    misuse_edges = pd.DataFrame.from_records(file_id2labeled_edges)

    misuse_nodes = misuse_edges.query("label == 'misuse'")["src"].unique()
    assert len(misuse_nodes) == len(misuse_edges.query("label == 'misuse'"))
    misuse_edges["train_mask"] = misuse_edges["file_id"].apply(lambda x: file_id2partition[x] == "train")
    misuse_edges["test_mask"] = misuse_edges["file_id"].apply(lambda x: file_id2partition[x] == "eval")
    misuse_edges["val_mask"] = misuse_edges["file_id"].apply(lambda x: file_id2partition[x] == "dev")
    misuse_edges["id"] = misuse_edges["src"]
    # partition = add_splits(misuse_edges[["file_id"]].rename({"file_id": "id"}, axis=1), 0.8)
    # misuse_edges_ = misuse_edges.merge(partition, how="left", left_on="file_id", right_on="id")
    # misuse_edges_.drop("id", axis=1, inplace=True)
    # assert (misuse_edges_.isna().sum() == 0).all()

    print(f"Found {len(misuse_edges)} misuse edges, skipped {len(skipped)} files")
    pd.Series(skipped).to_csv(dataset.joinpath("skipped.csv"), index=False)
    persist(misuse_edges, dataset.joinpath("misuse_edge_labels.json.bz2"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory")

    args = parser.parse_args()

    create_edge_labels(args.dataset_directory)