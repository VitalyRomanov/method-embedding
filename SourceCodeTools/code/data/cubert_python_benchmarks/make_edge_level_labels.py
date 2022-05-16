from ast import literal_eval
from pathlib import Path

import pandas as pd

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.cubert_python_benchmarks.partitioning import add_splits
from SourceCodeTools.code.data.file_utils import unpersist, persist


def create_edge_labels(dataset_directory):

    dataset = Path(dataset_directory)
    filecontent_path = dataset.joinpath("common_filecontent.json")
    nodes_path = dataset.joinpath("common_nodes.json")
    edges_path = dataset.joinpath("common_edges.json")

    def get_node_names():
        id2node_name = dict()
        for nodes in read_nodes(nodes_path, as_chunks=True):
            id2node_name.update(dict(zip(nodes["id"], nodes["serialized_name"])))
        return id2node_name

    id2node_name = get_node_names()

    def get_misuse_spans():
        filecontent = unpersist(filecontent_path)
        return dict(zip(filecontent["id"], filecontent["misuse_span"].map(literal_eval))), dict(zip(filecontent["id"], filecontent["comment"].map(lambda x: x.split("`")[-2])))

    file_id2misuse_span, file_id2replacement_var = get_misuse_spans()

    file_id2incorrect_edge = set()
    skipped = []
    file_id2incorrect_edges = []

    last_file_id = None
    total = 0
    for chunk_ind, edges in enumerate(read_edges(edges_path, as_chunks=True)):
        edges = edges.astype({"offset_start": "Int32", "offset_end": "Int32"})
        for ind, edge in edges.iterrows():
            if last_file_id != edge.file_id:
                total += 1
                if last_file_id is not None and last_file_id not in file_id2incorrect_edge:
                    # print(f"Did not find edge for {last_file_id}")
                    skipped.append(last_file_id)
                    # print(f"Skipped {len(skipped)} files out of {total}")
                last_file_id = edge.file_id

            if not pd.isna(edge["offset_start"]):
                file_id = edge["file_id"]
                src = edge["source_node_id"]
                dst = edge["target_node_id"]
                type = edge["type"]
                needed_span = tuple(file_id2misuse_span[file_id])
                given_span = (edge["offset_start"], edge["offset_end"])
                if needed_span == given_span:
                    replacement = file_id2replacement_var[file_id]
                    node_name = id2node_name[src]
                    assert node_name.startswith(replacement)
                    file_id2incorrect_edge.add(file_id)
                    file_id2incorrect_edges.append({
                        "src": src,
                        "dst": dst,
                        "type": type,
                        "file_id": file_id
                    })

    misuse_edges = pd.DataFrame.from_records(file_id2incorrect_edges)

    misuse_nodes = misuse_edges["src"].unique()
    assert len(misuse_nodes) == len(misuse_edges)
    partition = add_splits(misuse_edges[["src"]].rename({"src": "id"}, axis=1), 0.8)
    misuse_edges_ = misuse_edges.merge(partition, how="left", left_on="src", right_on="id")
    misuse_edges_.drop("id", axis=1, inplace=True)
    assert (misuse_edges_.isna().sum() == 0).all()

    print(f"Found {len(misuse_edges)} misuse edges, skipped {len(skipped)} files")
    pd.Series(skipped).to_csv(dataset.joinpath("skipped.csv"), index=False)
    persist(misuse_edges_, dataset.joinpath("misuse_edges.json.bz2"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory")

    args = parser.parse_args()

    create_edge_labels(args.dataset_directory)