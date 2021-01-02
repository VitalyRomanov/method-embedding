from csv import QUOTE_NONNUMERIC
import pandas as pd
import os


filenames = {
    "nodes": "nodes.bz2",
    "edges": "edges.bz2",
    "source_location": "source_location.csv",
    "occurrence": "occurrence.csv",
    "filecontent": "filecontent.csv",
    "source-graph-bodies": "source-graph-bodies.bz2",
    "call-seq": "call-seq.bz2",
    "function-variable-pairs": "function-variable-pairs.bz2"
}


def write_pickle(df, out_path, **kwargs):
    df.to_pickle(out_path, **kwargs)


def read_pickle(path, **kwargs):
    return pd.read_pickle(path, **kwargs)


def write_csv(df, out_path, **kwargs):
    df.to_csv(out_path, index=False, quoting=QUOTE_NONNUMERIC, **kwargs)


def read_csv(path, **kwargs):
    # quotechar = '"', escapechar = '\\'
    return pd.read_csv(path, na_filter=False, **kwargs)


def write_parquet(df, path, **kwargs):
    df.to_parquet(path, index=False, **kwargs)


def read_parquet(path, **kwargs):
    return pd.read_parquet(path, **kwargs)


def read_source_location(base_path):
    source_location_path = os.path.join(base_path, "source_location.csv")
    source_location = unpersist(source_location_path, sep=",",
                                  dtype={'id': int, 'file_node_id': int, 'start_line': int, 'start_column': int,
                                         'end_line': int, 'end_column': int, 'type': int}
                               )
    return source_location


def read_occurrence(base_path):
    occurrence_path = os.path.join(base_path, "occurrence.csv")
    occurrence = unpersist(occurrence_path, sep=",", dtype={'element_id': int, 'source_location_id': int})
    return occurrence


def read_filecontent(base_path):
    filecontent_path = os.path.join(base_path, "filecontent.csv")
    filecontent = unpersist(filecontent_path, sep=",", dtype={'id': int, 'content': str})
    return filecontent


def read_nodes(base_path):
    node_path = os.path.join(base_path, filenames["nodes"])
    nodes = unpersist(node_path)
    # pd.read_csv(node_path, sep=",", dtype={"id": int, "type": int, "serialized_name": str})
    return nodes


def write_nodes(nodes, base_path):
    node_path = os.path.join(base_path, filenames["nodes"])
    persist(nodes, node_path)


def read_edges(base_path):
    edge_path = os.path.join(base_path, filenames["edges"])
    edge = unpersist(edge_path)
    # pd.read_csv(edge_path, sep=",", dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})
    return edge


def write_edges(edges, base_path):
    edge_path = os.path.join(base_path, filenames["edges"])
    persist(edges, edge_path)


def read_processed_bodies(base_path):
    bodies_path = os.path.join(base_path, filenames["source-graph-bodies"])
    bodies = unpersist(bodies_path)
    return bodies


def write_processed_bodies(df, base_path):
    bodies_path = os.path.join(base_path, filenames["source-graph-bodies"])
    persist(df, bodies_path)


def persist(df:pd.DataFrame, path: str, **kwargs):
    if path.endswith(".csv"):
        write_csv(df, path, **kwargs)
    elif path.endswith(".pkl") or path.endswith(".bz2"):
        write_pickle(df, path, **kwargs)
    elif path.endswith(".parquet"):
        write_parquet(df, path, **kwargs)
    else:
        raise NotImplementedError("supported exrensions: csv, bz2, pkl, parquet")


def unpersist(path: str, **kwargs) -> pd.DataFrame:
    if path.endswith(".csv"):
        data = read_csv(path, **kwargs)
    elif path.endswith(".pkl") or path.endswith(".bz2"):
        data = read_pickle(path, **kwargs)
    elif path.endswith(".parquet"):
        data = read_parquet(path, **kwargs)
    else:
        raise NotImplementedError("supported exrensions: csv, bz2, pkl, parquet")
    return data