from functools import partial
from os.path import join

import pandas as pd

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.SQLiteStorage import SQLiteStorage


class OnDiskGraphStorage:
    def __init__(self, path):
        self.path = path
        self.database = SQLiteStorage(path)

    def _write_non_empty_table(self, table, table_name, create_index, **kwargs):
        if len(table) > 0:
            self.database.add_records(
                table=table,
                table_name=table_name,
                create_index=create_index,
                **kwargs
            )

    def _write_type_map(self, type_map, table_name):
        type_map_list = []
        for key, val in type_map.items():
            type_map_list.append({
                "type_id": val,
                "type_desc": key
            })

        self._write_non_empty_table(
            table=pd.DataFrame.from_records(type_map_list),
            table_name=table_name,
            create_index=["type_id", "type_desc"]
        )

    def _import_nodes(self, path):

        type_map = {}

        def update_types(nodes):
            for type_ in nodes["type"]:
                if type_ not in type_map:
                    type_map[type_] = len(type_map)

        for nodes in read_nodes(path, as_chunks=True):
            update_types(nodes)
            nodes["type"] = nodes["type"].apply(type_map.get)
            nodes.rename({"serialized_name": "name"}, axis=1, inplace=True)

            self._write_non_empty_table(
                table=nodes[["id", "type", "name"]],
                table_name="nodes",
                create_index=["id", "type"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "type": "INT NOT NULL",
                    "name": "TEXT NOT NULL"
                }
            )

            self._write_non_empty_table(
                table=nodes[["id", "mentioned_in"]].dropna(),
                table_name="node_hierarchy",
                create_index=["id", "mentioned_in"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "mentioned_in": "INT NOT NULL",
                }
            )

            self._write_non_empty_table(
                table=nodes[["id", "string"]].dropna(),
                table_name="node_strings",
                create_index=["id", "string"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "string": "TEXT NOT NULL",
                }
            )

        self._write_type_map(type_map, "node_types")

    def _import_edges(self, path):
        type_map = {}

        def update_types(nodes):
            for type_ in nodes["type"]:
                if type_ not in type_map:
                    type_map[type_] = len(type_map)

        for edges in read_edges(path, as_chunks=True):
            update_types(edges)
            edges["type"] = edges["type"].apply(type_map.get)
            edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1, inplace=True)

            self._write_non_empty_table(
                table=edges[["id", "type", "src", "dst"]],
                table_name="edges",
                create_index=["id", "type", "src", "dst"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "type": "INT NOT NULL",
                    "src": "INT NOT NULL",
                    "dst": "INT NOT NULL"
                }
            )

            self._write_non_empty_table(
                table=edges[["id", "file_id"]].dropna(),
                table_name="edge_file_id",
                create_index=["id", "file_id"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "file_id": "INT NOT NULL",
                }
            )

            self._write_non_empty_table(
                table=edges[["id", "mentioned_in"]].dropna(),
                table_name="edge_hierarchy",
                create_index=["id", "mentioned_in"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "mentioned_in": "INT NOT NULL",
                }
            )

        self._write_type_map(type_map, "edge_types")

    def import_from_files(self, path_to_dataset):
        get_path = partial(join, path_to_dataset)
        self._import_nodes(get_path("common_nodes.json.bz2"))
        self._import_edges(get_path("common_edges.json.bz2"))

    def get_node_types(self):
        return self.database.query("SELECT type_desc from node_types")["type_desc"]

    def get_edge_types(self):
        return self.database.query("SELECT type_desc from edge_types")["type_desc"]

    def iterate_nodes_with_chunks(self):
        return self.database.query("SELECT * FROM nodes", chunksize=10000)


if __name__ == "__main__":
    graph_storage = OnDiskGraphStorage("/Users/LTV/Downloads/v2_subsample_v4_new_ast2/with_ast/dataset.db")
    graph_storage.import_from_files("/Users/LTV/Downloads/v2_subsample_v4_new_ast2/with_ast")