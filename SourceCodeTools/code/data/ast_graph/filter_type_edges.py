import os
from os.path import join

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.file_utils import persist


def filter_type_edges(nodes, edges, keep_proportion=0.0):
    annotations = edges.query(f"type == 'annotation_for' or type == 'returned_by' or type == 'annotation_for_rev' or type == 'returned_by_rev'")
    no_annotations = edges.query(f"type != 'annotation_for' and type != 'returned_by' and type != 'annotation_for_rev' and type != 'returned_by_rev'")

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


def filter_type_edges_with_chunks(nodes_path, edges_path, kwarg_fn):

    node2name = {}
    for nodes in read_nodes(nodes_path, as_chunks=True):
        node2name.update(dict(zip(nodes["id"], nodes["serialized_name"])))

    temp_edges = join(os.path.dirname(edges_path), "temp_" + os.path.basename(edges_path))
    annotations_path = join(os.path.dirname(edges_path), "type_annotations.json")

    annotations_written = False

    for ind, edges in enumerate(read_edges(edges_path, as_chunks=True)):
        annotations = edges.query(
            f"type == 'annotation_for' or type == 'returned_by' or type == 'annotation_for_rev' or type == 'returned_by_rev'")
        no_annotations = edges.query(
            f"type != 'annotation_for' and type != 'returned_by' and type != 'annotation_for_rev' and type != 'returned_by_rev'")

        if annotations is not None and len(annotations) > 0:
            annotations["type_string"] = annotations["source_node_id"].apply(node2name.get)
            # rename columns to use as a dataset
            annotations.rename({"source_node_id": "dst", "type_string": "src"}, axis=1, inplace=True)
            annotations = annotations[["src", "dst"]]

            kwargs = kwarg_fn(annotations_path.endswith("csv"), first_written=annotations_written)
            persist(annotations, annotations_path, **kwargs)

            annotations_written = True

        kwargs = kwarg_fn(temp_edges.endswith("csv"), first_written=ind != 0)
        persist(no_annotations, temp_edges, **kwargs)

    os.remove(edges_path)
    os.rename(temp_edges, edges_path)