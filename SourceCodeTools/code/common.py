import hashlib
import os
import sqlite3

import pandas as pd
from tqdm import tqdm

from SourceCodeTools.code.data.file_utils import unpersist


class SQLTable:
    def __init__(self, df, filename, table_name):
        self.conn = sqlite3.connect(filename)
        self.path = filename
        self.table_name = table_name

        df.to_sql(self.table_name, con=self.conn, if_exists='replace', index=False, index_label=df.columns)

    def query(self, query_string):
        return pd.read_sql(query_string, self.conn)

    def __del__(self):
        self.conn.close()
        if os.path.isfile(self.path):
            os.remove(self.path)


def create_node_repr(nodes):
    return list(zip(nodes['serialized_name'], nodes['type']))


def compute_long_id(obj):
    return hashlib.md5(repr(obj).encode('utf-8')).hexdigest()


def map_id_columns(df, column_names, mapper):
    df = df.copy()
    for col in column_names:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: mapper.get(x, pd.NA))
    return df


def merge_with_file_if_exists(df, merge_with_file):
    if os.path.isfile(merge_with_file):
        original_data = unpersist(merge_with_file)
        data = pd.concat([original_data, df], axis=0)
    else:
        data = df
    return data


def custom_tqdm(iterable, total, message):
    return tqdm(iterable, total=total, desc=message, leave=False, dynamic_ncols=True)


def map_columns(input_table, id_map, columns, columns_special=None):

    input_table = map_id_columns(input_table, columns, id_map)

    if columns_special is not None:
        assert isinstance(columns_special, list), "`columns_special` should be iterable"
        for column, map_func in columns_special:
            input_table[column] = map_func(input_table[column], id_map)

    if len(input_table) == 0:
        return None
    else:
        return input_table


def grow_with_chunks(chunks, additional_dtypes):
    dtypes = {}

    table = None

    for chunk in chunks:
        for col, type_ in additional_dtypes.items():
            if col in chunk.columns:
                dtypes[col] = type_

        chunk = chunk.astype(dtypes)

        if table is None:
            table = chunk
        else:
            table = pd.concat([table, chunk], copy=False)
    return table


def read_nodes(node_path):
    dtypes = {
        'type': 'category',
        "serialized_name": "string",
    }

    nodes_chunks = unpersist(node_path, dtype=dtypes, chunksize=100000)

    additional_dtypes = {
        "mentioned_in": "Int32",
        "string": "string"
    }

    return grow_with_chunks(nodes_chunks, additional_dtypes)


def read_edges(edge_path):
    dtypes = {
        'type': 'category'
    }

    edge_chunks = unpersist(edge_path, dtype=dtypes, chunksize=100000)

    additional_types = {
        "mentioned_in": "Int32",
        "file_id": "Int32"
    }

    return grow_with_chunks(edge_chunks, additional_types)