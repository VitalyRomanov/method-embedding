import tempfile
from functools import partial
from os.path import join

import pandas as pd
from tqdm import tqdm
import diskcache as dc

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.SQLiteStorage import SQLiteStorage
from SourceCodeTools.code.data.dataset.partition_strategies import SGPartitionStrategies
from SourceCodeTools.code.data.file_utils import unpersist


class NodeTypes:
    def __init__(self, node2type_id, type_id2desc):
        self.node2type_id = node2type_id
        self.type_id2desc = type_id2desc

    def __getitem__(self, node_id):
        node_type_id = self.node2type_id[node_id]
        return self.type_id2desc[node_type_id]


class OnDiskGraphStorage:
    def __init__(self, path):
        self.path = path
        self.database = SQLiteStorage(path)
        self.cache_path = tempfile.TemporaryDirectory(suffix="OnDiskGraphStorage")
        self._cache = dc.Cache(self.cache_path.name)

    def write_to_cache(self, key, obj):
        self._cache[key] = obj

    def load_from_cache(self, key):
        if key in self._cache:
            return self._cache[key]
        else:
            return None

    @staticmethod
    def get_cache_key(how, group):
        return f"{how.name}_{group}"

    def _write_non_empty_table(self, table, table_name, create_index, **kwargs):
        if len(table) > 0:
            self.database.add_records(
                table=table,
                table_name=table_name,
                create_index=create_index,
                **kwargs
            )

    def _write_mapping(self, type_map, key_name, val_name, table_name):
        type_map_list = []
        for key, val in type_map.items():
            type_map_list.append({
                key_name: val,
                val_name: key
            })

        self._write_non_empty_table(
            table=pd.DataFrame.from_records(type_map_list),
            table_name=table_name,
            create_index=[key_name, val_name]
        )

    def _write_type_map(self, type_map, table_name):
        self._write_mapping(type_map, "type_id", "type_desc", table_name)

    def _write_package_map(self, package_map, table_name):
        self._write_mapping(package_map, "package_id", "package_desc", table_name)

    def _import_nodes(self, path):

        type_map = {}

        def update_types(nodes):
            for type_ in nodes["type"]:
                if type_ not in type_map:
                    type_map[type_] = len(type_map)

        for nodes in tqdm(read_nodes(path, as_chunks=True), desc="Importing nodes"):
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
        package_map = {}

        def update_mapping(seq, mapping):
            for val in seq:
                if val not in mapping:
                    mapping[val] = len(mapping)

        for edges in tqdm(read_edges(path, as_chunks=True), desc="Importing edges"):
            update_mapping(edges["type"], type_map)
            update_mapping(edges["package"].dropna(), package_map)
            edges["type"] = edges["type"].apply(type_map.get)
            edges["package"] = edges["package"].apply(package_map.get)
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
                    "package": "INT NOT NULL",
                }
            )

            rest_columns = [col for col in edges.columns if col not in {"id", "src", "dst", "type", "file_id", "package"}]
            self._write_non_empty_table(
                table=edges[["id"] + rest_columns],
                table_name="edge_hierarchy",
                create_index=["id", "mentioned_in"],
                dtype={
                    "id": "INT PRIMARY KEY",
                    "mentioned_in": "INT",
                }
            )

        self._write_type_map(type_map, "edge_types")
        self._write_package_map(package_map, "packages")

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
        # table = self.database.query("""
        # SELECT
        # id as id, type_desc as type
        # from
        # nodes
        # LEFT JOIN node_types on nodes.type = node_types.type_id
        # """)
        node_id2type_id = self.database.query("SELECT id, type as type_id from nodes")
        if not hasattr(self, "type_id2desc"):
            self.type_id2desc = self.database.query("SELECT type_id, type_desc from node_types")

        node_types = NodeTypes(dict(zip(node_id2type_id["id"], node_id2type_id["type_id"])),
                               dict(zip(self.type_id2desc["type_id"], self.type_id2desc["type_desc"])))

        return node_types

    def get_nodes(self, type_filter=None, **kwargs):
        query_str = """
        select id, type_desc as type, name 
        from nodes
        join node_types on nodes.type = node_types.type_id
        
        """
        if type_filter is not None:
            ntypes_str = ",".join((f"'{f}'" for f in type_filter))
            query_str += f"where type_desc in ({ntypes_str})"
        return self.database.query(query_str, **kwargs)

    def get_edges(self, type_filter=None, **kwargs):
        query_str = """
        select id, type_desc as type, src, dst 
        from edges
        join edge_types on edges.type = edge_types.type_id
        
        """
        if type_filter is not None:
            etypes_str = ",".join((f"'{f}'" for f in type_filter))
            query_str += f"where type_desc in ({etypes_str})"
        return self.database.query(query_str, **kwargs)

    def get_nodes_with_subwords(self):
        query_str = """
        select id, name 
        from nodes
        join (
            select dst 
            from edges
            join edge_types on edges.type = edge_types.type_id
            where type_desc = 'subword'
        ) as subword_dst
        on nodes.id = subword_dst.dst
        """
        return self.database.query(query_str)

    def get_nodes_for_classification(self):
        query_str = """
        select distinct dst as src, type_desc as dst
        from edges
        join nodes on edges.dst = nodes.id
        join node_types on nodes.type = node_types.type_id
        """
        return self.database.query(query_str)

    # def iterate_nodes_with_chunks(self):
    #     return self.database.query("SELECT * FROM nodes", chunksize=10000)

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

    def get_nodes_for_edges(self, edges):
        node_id_for_query = ",".join(map(str, set(edges["src"]) | set(edges["dst"])))
        nodes = self.database.query(
            f"""
                                SELECT
                                id, node_types.type_desc as type, name
                                FROM
                                nodes
                                LEFT JOIN node_types ON nodes.type = node_types.type_id
                                WHERE nodes.id IN ({node_id_for_query})
                                """
        )
        return nodes

    def get_subgraph_for_package(self, package_id):
        edges = self.database.query(
            f"""
                        SELECT
                        edges.id as id, edge_types.type_desc as type, src, dst
                        FROM
                        edge_file_id
                        LEFT JOIN edges ON edge_file_id.id = edges.id
                        LEFT JOIN edge_types ON edges.type = edge_types.type_id
                        WHERE edge_file_id.package = '{package_id}'
                        """
        )

        package_nodes = self.get_nodes_for_edges(edges)
        return package_nodes, edges

    def get_all_packages(self):
        return self.database.query("SELECT package_id, package_desc FROM packages")["package_id"]

    # def iterate_packages(self, all_packages=None):
    #     if all_packages is None:
    #         all_packages = self.database.query("SELECT package_id, package_desc FROM packages")["package_id"]
    #
    #     for package_id in all_packages:
    #         yield package_id, *self.get_subgraph_for_package(package_id)

    def get_subgraph_for_file(self, file_id):
        edges = self.database.query(
            f"""
                        SELECT
                        edges.id as id, edge_types.type_desc as type, src, dst
                        FROM
                        edge_file_id
                        LEFT JOIN edges ON edge_file_id.id = edges.id
                        LEFT JOIN edge_types ON edges.type = edge_types.type_id
                        WHERE edge_file_id.file_id = '{file_id}'
                        """
        )

        file_nodes = self.get_nodes_for_edges(edges)
        return file_nodes, edges

    def get_all_files(self):
        return self.database.query("SELECT distinct file_id FROM edge_file_id")["file_id"]

    # def iterate_files(self, all_files=None):
    #     if all_files is None:
    #         all_files = self.database.query("SELECT distinct file_id FROM edge_file_id")["file_id"]
    #
    #     for file_id in all_files:
    #         yield file_id, *self.get_subgraph_for_file(file_id)

    def get_subgraph_for_mention(self, mention):
        edges = self.database.query(
            f"""
                        SELECT
                        edges.id as id, edge_types.type_desc as type, src, dst
                        FROM
                        edge_hierarchy
                        LEFT JOIN edges ON edge_hierarchy.id = edges.id
                        LEFT JOIN edge_types ON edges.type = edge_types.type_id
                        WHERE edge_hierarchy.mentioned_in = '{mention}'
                        """
        )

        mention_nodes = self.get_nodes_for_edges(edges)
        return mention_nodes, edges

    def get_all_mentions(self):
        return self.database.query("SELECT distinct mentioned_in FROM edge_hierarchy")["mentioned_in"]

    # def iterate_mentions(self, all_mentions=None):
    #     if all_mentions is None:
    #         all_mentions = self.database.query("SELECT distinct mentioned_in FROM edge_hierarchy")["mentioned_in"]
    #
    #     for mention in all_mentions:
    #         yield mention, *self.get_subgraph_for_mention(mention)

    def iterate_subgraphs(self, how, groups):

        if how == SGPartitionStrategies.package:
            load_fn = self.get_subgraph_for_package
            if groups is None:
                groups = self.get_all_packages()
            # return self.iterate_packages(all_packages=groups)
        elif how == SGPartitionStrategies.file:
            load_fn = self.get_subgraph_for_file
            if groups is None:
                groups = self.get_all_files()
            # return self.iterate_files(all_files=groups)
        elif how == SGPartitionStrategies.mention:
            load_fn = self.get_subgraph_for_mention
            if groups is None:
                groups = self.get_all_mentions()
            # return self.iterate_mentions(all_mentions=groups)
        else:
            raise ValueError()

        for group in groups:
            cache_key = self.get_cache_key(how, group)
            cached = self.load_from_cache(cache_key)
            if cached is None:
                nodes, edges = load_fn(group)
                self.write_to_cache(cache_key, (nodes, edges))
            else:
                nodes, edges = cached
            yield group, nodes, edges

    # @staticmethod
    # def get_temp_schema(table, table_name):
    #     return pd.io.sql.get_schema(table, table_name).replace("CREATE TABLE", "CREATE TEMPORARY TABLE")

    def _create_tmp_node_ids_list(self, node_ids):
        table_name = "tmp_partition_nodes"
        # node_ids = node_ids.to_frame().rename({"src": "node_id"}, axis=1)
        # self.database.replace_records(node_ids, table_name, schema=self.get_temp_schema(node_ids, table_name))
        # # self.database.execute("CREATE INDEX temp.MyIndex ON MyTable(MyField)")
        # return table_name

        self.database.conn.execute(f"CREATE TEMPORARY TABLE {table_name}(node_ids INTEGER)")
        self.database.conn.executemany(
            f"insert into {table_name}(node_ids) values (?)", list(map(lambda x: (x,), node_ids))
        )
        self.database.conn.execute(
            f"CREATE INDEX IF NOT EXISTS temp.idx_tmp_partition_nodes ON {table_name}(node_ids)"
        )
        self.database.conn.commit()
        return table_name

    def _drop_tmp_node_ids_list(self, table_name):
        self.database.conn.execute(f"DROP TABLE {table_name}")
        self.database.conn.commit()

    def get_info_for_node_ids(self, node_ids, field):
        if field == SGPartitionStrategies.package:
            column_name = "package"
        elif field == SGPartitionStrategies.file:
            column_name = "file_id"
        elif field == SGPartitionStrategies.mention:
            column_name = "mentioned_in"
        else:
            raise ValueError()

        node_ids_table_name = self._create_tmp_node_ids_list(node_ids, )

        # node_id_str = ",".join(map(str, node_ids))
        query_str = f"""
        select distinct src, {column_name} from (
            select src, package, file_id, mentioned_in
            from {node_ids_table_name}
            inner join edges on edges.src = {node_ids_table_name}.node_ids
            inner join edge_file_id on edges.id = edge_file_id.id
            join edge_hierarchy on edges.id = edge_hierarchy.id
            union
            select dst as src, package, file_id, mentioned_in
            from {node_ids_table_name}
            inner join edges on edges.dst = {node_ids_table_name}.node_ids
            inner join edge_file_id on edges.id = edge_file_id.id
            join edge_hierarchy on edges.id = edge_hierarchy.id
        ) where {column_name} not null order by {column_name}
        """

        results = self.database.query(query_str)

        self._drop_tmp_node_ids_list(node_ids_table_name)
        return results, column_name

# class N4jGraphStorage:
#     def __init__(self):
#         import neo4j
#         self.database = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "1111"))
#         self.session = self.database.session()
#         self.tx = self.session.begin_transaction()
#
#     def query(self, query):
#         return self.tx.run(query)
#
#     def get_node_type_descriptions(self):
#         results = self.query("call db.labels()")["type_desc"]
#         return [r["label"] for r in results]
#
#     def get_edge_type_descriptions(self):
#         results = self.query("match ()-[r]->() return distinct type(r)")
#         return [r["type(r)"] for r in results]
#
#     def get_edge_types(self):
#         table = self.database.query("SELECT * from edge_types")
#         return dict(zip(table["type_id"], table["type_desc"]))
#
#     def get_node_types(self):
#         table = self.database.query("SELECT * from node_types")
#         return dict(zip(table["type_id"], table["type_desc"]))
#
#     def iterate_nodes_with_chunks(self):
#         return self.database.query("SELECT * FROM nodes", chunksize=10000)
#
#     def get_inbound_neighbors(self, ids):
#         ids_query = ",".join(f'"{id_}"' for id_ in ids)
#         results = self.query(f"MATCH (n)<--(connected) WHERE n.id in [{ids_query}] RETURN n, connected")
#         return [r["connected"]["id"] for r in results]
#
#     def get_subgraph_from_node_ids(self, ids):
#         ids_query = ",".join(f'"{id_}"' for id_ in ids)
#         results = self.query(f"MATCH p=(n)-->(m) WHERE n.id in [{ids_query}] and m.id in [{ids_query}] RETURN p")
#         # nodes = self.database.query(f"SELECT id, type FROM nodes WHERE id IN ({ids_query})")
#         # edges = self.database.query(f"SELECT type, src, dst FROM edges WHERE dst IN ({ids_query}) and src IN ({ids_query})")
#         nodes = []
#         edges = []
#         for r in results:
#             r = r["p"]
#             nodes.append({
#                 "id": int(r.start_node["id"]),
#                 "type": next(iter(r.start_node.labels))
#             })
#             nodes.append({
#                 "id": int(r.end_node["id"]),
#                 "type": next(iter(r.end_node.labels))
#             })
#             edges.append({
#                 "type": r.relationships[0].type,
#                 "src": nodes[-2]["id"],
#                 "dst": nodes[-1]["id"]
#             })
#
#         nodes = pd.DataFrame.from_records(nodes, columns=["id", "type"]).drop_duplicates()
#         edges = pd.DataFrame.from_records(edges, columns=["type", "src", "dst"]).drop_duplicates()
#         return nodes, edges


if __name__ == "__main__":
    graph_storage = OnDiskGraphStorage("/Users/LTV/Downloads/v2_subsample_v4_new_ast2/with_ast/dataset.db")
    graph_storage.import_from_files("/Users/LTV/Downloads/v2_subsample_v4_new_ast2/with_ast")
