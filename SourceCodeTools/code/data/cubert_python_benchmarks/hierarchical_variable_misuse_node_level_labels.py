from collections import defaultdict, Counter

from SourceCodeTools.code.data.FileSystemStorage import FileSystemStorage, LocationIterator
from SourceCodeTools.code.data.file_utils import read_mapping_from_json, write_mapping_to_json


def get_node_labels(dataset_path):
    storage = FileSystemStorage(dataset_path)

    summary = {
        "train": Counter(),
        "test": Counter(),
        "val": Counter(),
    }

    objective_name = "variable_misuse_node"

    summary_path = storage.get_objective_summary_path(objective_name)
    label_locations_path = storage.get_objective_labels_location_path(objective_name)

    labels_name = f"{objective_name}.json"
    locations = []

    for file_path in storage._stream_files():
        metadata = read_mapping_from_json(file_path.joinpath("metadata.json"))
        original_span = tuple(metadata["original_span"])
        misuse_span = tuple(metadata["misuse_span"])

        has_misuse = len(misuse_span) > 0
        seek_span = misuse_span if has_misuse else original_span
        labels = defaultdict(list)
        found = False

        for entry_path in storage._iterate_all(file_path, "entry.json"):
            entry = read_mapping_from_json(entry_path.joinpath("entry.json"))
            span_to_edge = defaultdict(list)
            for start, end, edge in entry["normalized_edge_offsets"]:
                span_to_edge[(start, end)].append(edge)
            needed_edges = span_to_edge[seek_span]
            # node_names = dict(zip(map(int, entry["node_names"].keys()), entry["node_names"].values()))

            if len(needed_edges) > 0:
                assert len(needed_edges) == 1
                seek_edge = needed_edges[0]
                for eid, src, dst, etype in zip(entry["edge_id"], entry["src_id"], entry["dst_id"], entry["edge_type"]):
                    if eid != seek_edge:
                        continue
                    # assert node_names[src] == text[seek_span[0]: seek_span[1]]
                    labels["node_id"].append(src)
                    labels["label"].append("misused" if has_misuse else "correct")
                    found = True
                    write_mapping_to_json(labels, entry_path.joinpath(labels_name))
                    locations.append(entry_path)
                    break

            if found is True:
                break

        summary[metadata["partition"]] += Counter(labels["label"])

    summary = {
        "train": dict(summary["train"].most_common()),
        "test": dict(summary["test"].most_common()),
        "val": dict(summary["val"].most_common()),
        "labels_for": "nodes",
        "filter_edges": [],  # "returned_by" is not used in this objective
        "mask": "node_name",
        "objective_type": "clf",  # clf | link | gen
        "labels_filename": labels_name
    }
    summary_path.parent.mkdir(exist_ok=True)
    write_mapping_to_json(summary, summary_path)
    LocationIterator.write_locations(locations, label_locations_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")

    args = parser.parse_args()

    get_node_labels(args.dataset_path)
