import sys
from os.path import join

from SourceCodeTools.code.data.sourcetrail.file_utils import *


def filter_type_edges(nodes, edges, keep_proportion=0.0):
    annotations = edges.query(f"type == 'annotation_for' or type == 'returned_by'")
    no_annotations = edges.query(f"type != 'annotation_for' and type != 'returned_by'")

    to_keep = int(len(annotations) * keep_proportion)
    if to_keep == 0:
        annotations_removed = annotations
        annotations_kept = None
    elif to_keep == len(annotations):
        annotations_removed = None
        annotations_kept = annotations
    else:
        annotations = annotations.sample(frac=1.)
        annotations_kept, annotations_removed = annotations.iloc[:to_keep], annotations.iloc[to_keep:]

    if annotations_kept is not None:
        no_annotations = no_annotations.append(annotations_kept)

    annotations = annotations_removed
    if annotations is not None:
        annotations = annotations_removed
        node2name = dict(zip(nodes["id"], nodes["serialized_name"]))
        get_name = lambda id_: node2name[id_]
        annotations["source_node_id"] = annotations["source_node_id"].apply(get_name)
        # rename columns to use as a dataset
        annotations.rename({"source_node_id": "dst", "target_node_id": "src"}, axis=1, inplace=True)
        annotations = annotations[["src","dst"]]

    return no_annotations, annotations


def main():
    working_directory = sys.argv[1]
    nodes = unpersist(join(working_directory, "nodes.bz2"))
    edges = unpersist(join(working_directory, "edges.bz2"))
    out_annotations = sys.argv[2]
    out_no_annotations = sys.argv[3]

    no_annotations, annotations = filter_type_edges(nodes, edges)

    persist(annotations, out_annotations)
    persist(no_annotations, out_no_annotations)

if __name__ == "__main__":
    main()