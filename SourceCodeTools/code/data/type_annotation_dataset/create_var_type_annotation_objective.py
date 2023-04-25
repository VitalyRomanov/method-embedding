import os
from collections import Counter, defaultdict

from tqdm import tqdm

from SourceCodeTools.code.data.FileSystemStorage import FileSystemStorage, LocationIterator
from SourceCodeTools.code.data.file_utils import write_mapping_to_json, read_mapping_from_json
from SourceCodeTools.code.data.type_annotation_dataset.type_parser import TypeHierarchyParser


def collect_objective_summary(storage, objective_name):
    summary_path = storage.get_objective_summary_path(objective_name)
    labels_locations = storage.get_objective_labels_location_path(objective_name)
    locations = []

    summary = {
        "train": Counter(),
        "test": Counter(),
        "val": Counter()
    }

    for loc_path in tqdm(storage._iterate_all(storage._path, "entry.json")):
        entry = read_mapping_from_json(loc_path.joinpath("entry.json"))
        if "type_annotations" not in entry or len(entry["type_annotations"]) == 0:
            continue

        objective_data = defaultdict(list)

        for edge_id, src, dst, etype in zip(entry["edge_id"], entry["src_id"], entry["dst_id"], entry["edge_type"]):
            if etype != "annotation_for":
                continue

            objective_data["node_id"].append(dst)
            objective_data["label"].append(
                TypeHierarchyParser(
                    entry["node_names"][str(src)], normalize=True
                ).assemble(max_level=3, simplify_nodes=True)
            )

        if len(objective_data) > 0:
            locations.append(loc_path)
            partition = storage._determine_partition(loc_path)
            targets = Counter(objective_data["label"])
            summary[partition] += targets

            write_mapping_to_json(objective_data, loc_path.joinpath(objective_name + ".json"))

    allowed_types = set(summary["train"].keys())
    for loc_path in locations:
        objective_data = read_mapping_from_json(loc_path.joinpath(objective_name + ".json"))
        to_write = defaultdict(list)
        for nid, lbl in zip(objective_data["node_id"], objective_data["label"]):
            if lbl in allowed_types:
                to_write["node_id"].append(nid)
                to_write["label"].append(lbl)

        if len(to_write) == 0:
            os.remove(loc_path.joinpath(objective_name + ".json"))

        write_mapping_to_json(to_write, loc_path.joinpath(objective_name + ".json"))

    summary = {
        "train": dict(summary["train"].most_common()),
        "test": dict((k, v) for k, v in summary["test"].most_common() if k in allowed_types),
        "val": dict((k, v) for k, v in summary["val"].most_common() if k in allowed_types),
        "labels_for": "nodes",
        "filter_edges": ["annotation_for"],  # "returned_by" is not used in this objective
        "mask": "node_name",
        "objective_type": "clf",  # clf | link | gen
        "labels_filename": objective_name + ".json"
    }

    summary_path.parent.mkdir(exist_ok=True)
    write_mapping_to_json(summary, summary_path)
    if locations is not None:
        LocationIterator.write_locations(locations, labels_locations)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    s = FileSystemStorage(args.path)
    collect_objective_summary(s, "type_annotation")
