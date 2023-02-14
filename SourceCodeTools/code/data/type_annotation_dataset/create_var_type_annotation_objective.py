from collections import Counter

from tqdm import tqdm

from SourceCodeTools.code.data.FileSystemStorage import FileSystemStorage, LocationIterator
from SourceCodeTools.code.data.file_utils import unpersist, write_mapping_to_json, read_mapping_from_json


def collect_objective_summary(s, labels_filename):
    summary_path = s._path.joinpath(".summary").joinpath("summary_" + labels_filename.split(".")[0] + ".json")
    labels_locations = s._path.joinpath(".summary").joinpath("location_" + labels_filename.split(".")[0] + ".txt")
    locations = []

    # if labels_locations.is_file():
    #     locations = None

    summary = {
        "train": Counter(),
        "test": Counter(),
        "val": Counter()
    }

    for loc_path in tqdm(s._iterate_all(s._path, "metadata.json")):
        metadata = read_mapping_from_json(loc_path.joinpath("metadata.json"))
        if "type_annotations" not in metadata or len(metadata["type_annotations"]) == 0:
            continue

        edges = unpersist(loc_path.joinpath("edges.parquet"))
        type_ann_edges = edges.query("type == 'annotation_for'")
        if len(type_ann_edges) > 0:
            locations.append(str(loc_path.relative_to(s._path)))
            nodes = unpersist(loc_path.joinpath("nodes.parquet"))
            node2name = dict(zip(nodes["id"], nodes["serialized_name"]))
            type_ann_edges["type_name"] = type_ann_edges["source_node_id"].apply(node2name.get)
            type_ann_edges["type_name_normalized"] = type_ann_edges["type_name"].apply(lambda x: x.split("[")[0].split(".")[-1].strip("\"").strip("'"))
            type_ann = type_ann_edges.rename({"target_node_id": "src", "type_name_normalized": "dst"}, axis=1)

            partition = s._determine_partition(loc_path)
            targets = Counter(type_ann["dst"])
            summary[partition] |= targets

    summary = {
        "train": dict(summary["train"].most_common()),
        "test": dict(summary["test"].most_common()),
        "val": dict(summary["val"].most_common()),
    }

    summary_path.parent.mkdir(exist_ok=True)
    write_mapping_to_json(summary, summary_path)
    if locations is not None:
        LocationIterator.write_locations(locations, labels_locations)

if __name__ == "__main__":
    s = FileSystemStorage("/Users/LTV/Documents/popular_packages/graph")
    collect_objective_summary(s, "type_ann_edges.parquet")