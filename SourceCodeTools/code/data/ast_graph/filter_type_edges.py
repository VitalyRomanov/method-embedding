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