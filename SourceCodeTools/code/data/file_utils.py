import bz2
import logging
import tempfile
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import os


filenames = {
    "nodes_csv": "nodes.csv",
    "edges_csv": "edges.csv",
    "nodes": "nodes.bz2",
    "edges": "edges.bz2",
    "source_location": "source_location.csv",
    "occurrence": "occurrence.csv",
    "filecontent": "filecontent.csv",
    "source_graph_bodies": "source_graph_bodies.bz2",
    "call_seq": "call_seq.bz2",
    "function_variable_pairs": "function_variable_pairs.bz2",
    "element_component": "element_component.csv"
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


def write_json(df, path, **kwargs):
    if "mode" in kwargs:
        mode = kwargs.pop("mode")
    else:
        mode = None

    if mode == "a":
        with open(path, "a") as sink:
            sink.write("\n")
            sink.write(df.to_json(path_or_buf=None, orient="records", lines=True, **kwargs))
    else:
        df.to_json(path_or_buf=path, orient="records", lines=True, **kwargs)


def read_json_with_generator(path, chunksize, **kwargs):
    if str(path).endswith(".bz2"):
        source = bz2.open(path, mode="rt")
    else:
        source = open(path, "r")
    # need to read manually, probably a bug in pandas
    buffer = []
    last_index = 0

    def prepare_chunk(buffer, last_index):
        chunk = pd.read_json("".join(buffer), orient="records", lines=True, **kwargs)
        chunk.index = np.array(list(range(last_index, last_index + len(chunk))))
        last_index = last_index + len(chunk)
        return chunk, last_index

    for ind, line in enumerate(source):
        buffer.append(line)
        if len(buffer) >= chunksize:
            chunk, last_index = prepare_chunk(buffer, last_index)
            # chunk = pd.read_json("".join(buffer), orient="records", lines=True, **kwargs)
            # chunk.index = np.array(list(range(last_index, last_index + len(chunk))))
            # last_index = last_index + len(chunk)
            yield chunk
            buffer.clear()
    if len(buffer) != 0:
        chunk, last_index = prepare_chunk(buffer, last_index)
        # chunk = pd.read_json("".join(buffer), orient="records", lines=True, **kwargs)
        # chunk.index = np.array(list(range(last_index, last_index + len(chunk))))
        # last_index = last_index + len(chunk)
        yield chunk


def _grow_with_chunks(chunks):
    table = None
    for chunk in chunks:
        if table is None:
            table = chunk
        else:
            table = pd.concat([table, chunk], copy=False)
    return table


def read_json(path, **kwargs):
    if "chunksize" in kwargs:
        return read_json_with_generator(path, kwargs.pop("chunksize"), **kwargs)
    else:
        return _grow_with_chunks(read_json_with_generator(path, 100000, **kwargs))
        # return pd.read_json(path, orient="records", lines=True, **kwargs)


def read_source_location(base_path):
    source_location_path = os.path.join(base_path, filenames["source_location"])

    source_location = unpersist_or_exit(
        source_location_path,
        exit_message=f"Does not exist or empty: {filenames['source_location']}",
        sep=",",
        dtype={
            'id': int, 'file_node_id': int, 'start_line': int, 'start_column': int,
            'end_line': int, 'end_column': int, 'type': int
        }
    )
    return source_location


def read_occurrence(base_path):
    occurrence_path = os.path.join(base_path, filenames["occurrence"])

    occurrence = unpersist_or_exit(
        occurrence_path,
        exit_message=f"Does not exist or empty: {filenames['occurrence']}",
        sep=",", dtype={'element_id': int, 'source_location_id': int}
    )
    return occurrence


def read_element_component(base_path):
    occurrence_path = os.path.join(base_path, filenames["element_component"])

    # this file is not essential, hence it is okay if it is not there
    occurrence = unpersist_if_present(
        occurrence_path, dtype={'element_id': int, 'id': int, 'type': int}
    )
    return occurrence


def read_filecontent(base_path):
    filecontent_path = os.path.join(base_path, filenames["filecontent"])

    filecontent = unpersist_or_exit(
        filecontent_path,
        exit_message=f"Does not exist or empty: {filenames['filecontent']}",
        sep=",", dtype={'id': int, 'content': str}
    )
    return filecontent


def read_nodes(base_path):
    node_path = os.path.join(base_path, filenames["nodes"])

    nodes = unpersist_or_exit(
        node_path,
        exit_message=f"Does not exist or empty: {filenames['nodes']}"
    )
    # pd.read_csv(node_path, sep=",", dtype={"id": int, "type": int, "serialized_name": str})
    return nodes


def write_nodes(nodes, base_path):
    node_path = os.path.join(base_path, filenames["nodes"])
    persist(nodes, node_path)


def read_edges(base_path):
    edge_path = os.path.join(base_path, filenames["edges"])

    edge = unpersist_or_exit(
        edge_path,
        exit_message=f"Does not exist or empty: {filenames['edges']}"
    )
    # pd.read_csv(edge_path, sep=",", dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})
    return edge


def write_edges(edges, base_path):
    edge_path = os.path.join(base_path, filenames["edges"])
    persist(edges, edge_path)


def read_processed_bodies(base_path):
    bodies_path = os.path.join(base_path, filenames["source_graph_bodies"])

    bodies = unpersist_or_exit(
        bodies_path,
        exit_message=f"Does not exist or empty: {filenames['source_graph_bodies']}"
    )
    return bodies


def write_processed_bodies(df, base_path):
    bodies_path = os.path.join(base_path, filenames["source_graph_bodies"])
    persist(df, bodies_path)


def likely_format(path, kwargs):
    path = Path(path)
    name_parts = path.name.split(".")
    if len(name_parts) == 1:
        raise ValueError("Extension is not found for the file:", str(path))

    extensions = "." + ".".join(name_parts[1:])

    if ".csv" in extensions:
        ext = "csv"
    elif ".tsv" in extensions:
        ext = "csv"
        kwargs["sep"] = "\t"
    elif ".json" in extensions:
        ext = "json"
    elif ".pkl" in extensions or extensions.endswith(".bz2"):
        ext = "pkl"
    elif ".parquet" in extensions:
        ext = "parquet"
    else:
        raise NotImplementedError("supported extensions: csv, bz2, pkl, parquet, json", extensions)

    return ext, kwargs


def persist(df: pd.DataFrame, path: Union[str, Path, bytes], **kwargs):
    if isinstance(path, Path):
        path = str(path.absolute())

    format, kwargs = likely_format(path, kwargs)
    if format == "csv":
        write_csv(df, path, **kwargs)
    elif format == "pkl":
        write_pickle(df, path, **kwargs)
    elif format == "parquet":
        write_parquet(df, path, **kwargs)
    elif format == "json":
        write_json(df, path, **kwargs)


def get_cached_path(path):
    return path + "__cached.pkl"


def unpersist(path: Union[str, Path, bytes], **kwargs) -> pd.DataFrame:
    if isinstance(path, Path):
        path = str(path.absolute())

    # cached_path = get_cached_path(path)
    # if os.path.isfile(cached_path):
    #     return read_pickle(cached_path)

    format, kwargs = likely_format(path, kwargs)
    if format == "csv":
        data = read_csv(path, **kwargs)
    elif format == "pkl":
        data = read_pickle(path, **kwargs)
    elif format == "parquet":
        data = read_parquet(path, **kwargs)
    elif format == "json":
        data = read_json(path, **kwargs)
    else:
        data = None

    # if isinstance(data, pd.DataFrame):
    #     if data is not None:
    #         write_pickle(data, cached_path)

    return data


def unpersist_if_present(path, **kwargs):
    if os.path.isfile(path):
        return unpersist(path, **kwargs)
    else:
        return None


def unpersist_or_exit(path, exit_message=None, **kwargs):

    data = unpersist_if_present(path, **kwargs)
    if data is None or len(data) == 0:
        if exit_message:
            logging.warning(exit_message)
        import sys
        sys.exit()
    else:
        return data


def get_random_name(length=10):
    char_ranges = [chr(i) for i in range(ord("a"), ord("a")+26)] + \
                  [chr(i) for i in range(ord("A"), ord("A")+26)] + \
                  [chr(i) for i in range(ord("0"), ord("0")+10)]
    from random import sample
    return "".join(sample(char_ranges, k=length))


def get_temporary_filename():
    tmp_dir = tempfile.gettempdir()
    name_generator = tempfile._get_candidate_names()
    path = os.path.join(tmp_dir, next(name_generator))
    while os.path.isdir(path):
        path = os.path.join(tmp_dir, next(name_generator))
    return path
