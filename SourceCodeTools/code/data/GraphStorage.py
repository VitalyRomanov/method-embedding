from functools import partial
from os.path import join

import pandas as pd

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.SQLiteStorage import SQLiteStorage
from SourceCodeTools.code.data.file_utils import unpersist


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
                table=edges[["id", "file_id", "package"]].dropna(),
                table_name="edge_file_id",
                create_index=["id", "file_id", "package"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "file_id": "INT NOT NULL",
                    "package": "TEXT NOT NULL",
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

    def _import_filecontent(self, path):
        for filecontent in unpersist(path, chunksize=100000):
            self._write_non_empty_table(
                table=filecontent[["id", "content", "package"]].dropna(),
                table_name="filecontent",
                create_index=["id", "package"],
                dtype={
                    "id": "INT NOT NULL",
                    "content": "TEXT NOT NULL",
                    "package": "INT NOT NULL",
                }
            )


    def import_from_files(self, path_to_dataset):
        get_path = partial(join, path_to_dataset)
        self._import_nodes(get_path("common_nodes.json.bz2"))
        self._import_edges(get_path("common_edges.json.bz2"))
        # self._import_filecontent(get_path("common_filecontent.json.bz2"))

    def get_node_type_descriptions(self):
        return self.database.query("SELECT type_desc from node_types")["type_desc"]

    def get_edge_type_descriptions(self):
        return self.database.query("SELECT type_desc from edge_types")["type_desc"]

    def get_edge_types(self):
        table = self.database.query("SELECT * from edge_types")
        return dict(zip(table["type_id"], table["type_desc"]))

    def get_node_types(self):
        table = self.database.query("SELECT * from node_types")
        return dict(zip(table["type_id"], table["type_desc"]))

    def iterate_nodes_with_chunks(self):
        return self.database.query("SELECT * FROM nodes", chunksize=10000)

    def get_inbound_neighbors(self, ids):
        ids_query = ",".join(str(id_) for id_ in ids)
        return self.database.query(f"SELECT src FROM edges WHERE dst IN ({ids_query})")["src"]

    def get_subgraph_from_node_ids(self, ids):
        ids_query = ",".join(str(id_) for id_ in ids)
        nodes = self.database.query(
            f"""SELECT 
            id, node_types.type_desc as type 
            FROM 
            nodes
            JOIN node_types ON node_types.type_id = nodes.type 
            WHERE id IN ({ids_query})
            """
        )
        edges = self.database.query(
            f"""SELECT 
            edge_types.type_desc as type, src, dst 
            FROM edges
            JOIN edge_types ON edge_types.type_id = edges.type 
            WHERE dst IN ({ids_query}) and src IN ({ids_query})
            """
        )
        return nodes, edges

    def iterate_packages(self):
        # nodes = []
        all_edges = []

        prev_package = None

        all_packages = self.database.query("SELECT DISTINCT package FROM edge_file_id")["package"]

        for package in all_packages:
            edges = self.database.query(
                f"""
                SELECT
                edge_types.type_desc as type, src, dst
                FROM
                edge_file_id
                LEFT JOIN edges ON edge_file_id.id = edges.id
                LEFT JOIN edge_types ON edges.type = edge_types.type_id
                WHERE edge_file_id.package = '{package}'
                """
            )

            node_id_for_query = ",".join(map(str, set(edges["src"]) | set(edges["dst"])))
            package_nodes = self.database.query(
                f"""
                SELECT
                id, node_types.type_desc as type
                FROM
                nodes
                LEFT JOIN node_types ON nodes.type = node_types.type_id
                WHERE nodes.id IN ({node_id_for_query})
                """
            )

            print()

        # for edges in self.database.query(
        #     """
        #     SELECT
        #     edges_with_type_src_node.type as type,
        #     edges_with_type_src_node.src as src,
        #     edges_with_type_src_node.dst as dst,
        #     edges_with_type_src_node.src_type as src_type,
        #     dst_node_types.type_desc as dst_type
        #     FROM
        #     (
        #         SELECT
        #         edges_with_type.type as type,
        #         edges_with_type.src as src,
        #         edges_with_type.dst as dst,
        #         src_node_types.type_desc as src_type
        #         FROM
        #         (
        #             SELECT
        #             edge_types.type_desc as type, src, dst
        #             FROM
        #             edge_file_id
        #             LEFT JOIN edges ON edge_file_id.id = edges.id
        #             LEFT JOIN edge_types ON edges.type = edge_types.type_id
        #             ORDER BY edge_file_id.package
        #         ) as edges_with_type
        #         LEFT JOIN nodes as src_nodes ON src = src_nodes.id
        #         LEFT JOIN node_types as src_node_types ON src_nodes.type = src_node_types.type_id
        #     ) as edges_with_type_src_node
        #     LEFT JOIN nodes as dst_nodes ON dst = dst_nodes.id
        #     LEFT JOIN node_types as dst_node_types ON dst_nodes.type = dst_node_types.type_id
        #     """,
        #     chunksize=10000
        # ):
        #
        #     start_with = 0
        #
        #     for ind, package in enumerate(edges["package"]):
        #         if prev_package is None:
        #             prev_package = package
        #         else:
        #             if prev_package != package:
        #                 all_edges.append(
        #                     edges.iloc[start_with: ind-1]
        #                 )
        #                 # yield pd.concat(all_edges)
        #                 start_with = ind
        #                 prev_package = package
        #
        #     all_edges.append(
        #         edges.iloc[start_with: -1]
        #     )
        #     # yield pd.concat(all_edges)




class n4jGraphStorage:
    def __init__(self):
        import neo4j
        self.database = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "1111"))
        self.session = self.database.session()
        self.tx = self.session.begin_transaction()

    def query(self, query):
        return self.tx.run(query)

    def get_node_type_descriptions(self):
        results = self.query("call db.labels()")["type_desc"]
        return [r["label"] for r in results]

    def get_edge_type_descriptions(self):
        results = self.query("match ()-[r]->() return distinct type(r)")
        return [r["type(r)"] for r in results]

    def get_edge_types(self):
        table = self.database.query("SELECT * from edge_types")
        return dict(zip(table["type_id"], table["type_desc"]))

    def get_node_types(self):
        table = self.database.query("SELECT * from node_types")
        return dict(zip(table["type_id"], table["type_desc"]))

    def iterate_nodes_with_chunks(self):
        return self.database.query("SELECT * FROM nodes", chunksize=10000)

    def get_inbound_neighbors(self, ids):
        ids_query = ",".join(f'"{id_}"' for id_ in ids)
        results = self.query(f"MATCH (n)<--(connected) WHERE n.id in [{ids_query}] RETURN n, connected")
        return [r["connected"]["id"] for r in results]

    def get_subgraph_from_node_ids(self, ids):
        ids_query = ",".join(f'"{id_}"' for id_ in ids)
        results = self.query(f"MATCH p=(n)-->(m) WHERE n.id in [{ids_query}] and m.id in [{ids_query}] RETURN p")
        # nodes = self.database.query(f"SELECT id, type FROM nodes WHERE id IN ({ids_query})")
        # edges = self.database.query(f"SELECT type, src, dst FROM edges WHERE dst IN ({ids_query}) and src IN ({ids_query})")
        nodes = []
        edges = []
        for r in results:
            r = r["p"]
            nodes.append({
                "id": int(r.start_node["id"]),
                "type": next(iter(r.start_node.labels))
            })
            nodes.append({
                "id": int(r.end_node["id"]),
                "type": next(iter(r.end_node.labels))
            })
            edges.append({
                "type": r.relationships[0].type,
                "src": nodes[-2]["id"],
                "dst": nodes[-1]["id"]
            })

        nodes = pd.DataFrame.from_records(nodes, columns=["id", "type"]).drop_duplicates()
        edges = pd.DataFrame.from_records(edges, columns=["type", "src", "dst"]).drop_duplicates()
        return nodes, edges


if __name__ == "__main__":
    graph_storage = OnDiskGraphStorage("/Users/LTV/Downloads/v2_subsample_v4_new_ast2/with_ast/dataset.db")
    graph_storage.import_from_files("/Users/LTV/Downloads/v2_subsample_v4_new_ast2/with_ast")