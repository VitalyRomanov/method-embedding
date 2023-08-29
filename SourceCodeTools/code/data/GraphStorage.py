import asyncio
import hashlib
import logging
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from os.path import join
from pathlib import Path
from threading import Thread

import pandas as pd
from tqdm import tqdm
import diskcache as dc

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.DBStorage import SQLiteStorage, AbstractDBStorage, PostgresStorage
from SourceCodeTools.code.data.dataset.partition_strategies import SGPartitionStrategies
from SourceCodeTools.code.data.file_utils import unpersist, read_mapping_from_json, write_mapping_to_json


class NodeTypes:
    def __init__(self, node2type_id, type_id2desc):
        self.node2type_id = node2type_id
        self.type_id2desc = type_id2desc

    def __getitem__(self, node_id):
        node_type_id = self.node2type_id[node_id]
        return self.type_id2desc[node_type_id]


def read_or_add_summary_field(field):
    def read_or_add_summary(function):
        def wrapper(self, summary_filename="summary.json"):
            dataset_summary_path = self.path.parent.joinpath(summary_filename)

            if dataset_summary_path.is_file():
                summary = read_mapping_from_json(dataset_summary_path)
            else:
                summary = dict()

            if field not in summary:
                summary[field] = function(self)
                write_mapping_to_json(summary, dataset_summary_path)
            return summary[field]

        return wrapper
    return read_or_add_summary


class AbstractGraphStorage(ABC):
    ...
    @abstractmethod
    def get_node_type_descriptions(self):
        ...

    @abstractmethod
    def get_edge_type_descriptions(self):
        ...

    @abstractmethod
    def get_num_nodes(self):
        ...

    @abstractmethod
    def get_num_edges(self):
        ...

    @abstractmethod
    def get_nodes(self, type_filter=None):
        ...

    @abstractmethod
    def get_edges(self, type_filter=None):
        ...

    @abstractmethod
    def iterate_subgraphs(self, how, ids):
        ...

    @abstractmethod
    def get_nodes_with_subwords(self):
        ...

    @abstractmethod
    def get_nodes_for_classification(self):
        ...

    @abstractmethod
    def get_info_for_subgraphs(self, subgraph_ids, field):
        ...

    @abstractmethod
    def get_info_for_node_ids(self, node_ids, field):
        ...

    @abstractmethod
    def get_info_for_edge_ids(self, edge_ids, field):
        ...

    @abstractmethod
    def get_node_types(self):
        ...

    # @abstractmethod
    # def get_edge_types(self):
    #     ...


class OnDiskGraphStorage(AbstractGraphStorage):
    storage_class = SQLiteStorage  # can change this between PostgresStorage and SQLiteStorage

    def __init__(self, path):
        self.path = Path(path)
        if self.storage_class == PostgresStorage:
            self.database = PostgresStorage("localhost")
        elif self.storage_class == SQLiteStorage:
            self.database = SQLiteStorage(path)

        self.cache_path = tempfile.TemporaryDirectory(suffix="OnDiskGraphStorage")
        self._cache = dc.Cache(self.cache_path.name)

    @classmethod
    def get_storage_file_name(cls, path):
        return cls.storage_class.get_storage_file_name(path)

    @classmethod
    def verify_imported(cls, path):
        return cls.storage_class.verify_imported(path)

    @classmethod
    def add_import_completed_flag(cls, path):
        cls.storage_class.add_import_completed_flag(path)

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

    def _write_node_file_id_table(self):
        self.database.execute(
            """
            create table node_file_id as 
            select distinct src as id, unique_file_id, file_id, package from (
                select src, package, file_id, unique_file_id
                from edges
                inner join edge_file_id on edges.id = edge_file_id.id
                union
                select dst as src, package, file_id, unique_file_id
                from edges
                inner join edge_file_id on edges.id = edge_file_id.id
            )
            """
        )

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
                    "id": AbstractDBStorage.DataTypes.INT_PRIMARY,
                    "type": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                    "name": AbstractDBStorage.DataTypes.TEXT_NOT_NULL
                }
            )

            self._write_non_empty_table(
                table=nodes[["id", "mentioned_in"]].dropna(),
                table_name="node_hierarchy",
                create_index=["id", "mentioned_in"],
                dtype={
                    "id": AbstractDBStorage.DataTypes.INT_PRIMARY,
                    "mentioned_in": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                }
            )

            self._write_non_empty_table(
                table=nodes[["id", "string"]].dropna(),
                table_name="node_strings",
                create_index=["id"],
                dtype={
                    "id": AbstractDBStorage.DataTypes.INT_PRIMARY,
                    "string": AbstractDBStorage.DataTypes.TEXT_NOT_NULL,
                }
            )

        self._write_type_map(type_map, "node_types")

    def _import_edges(self, path):
        type_map = {}
        package_map = {}
        file_id_map = {}

        def update_mapping(seq, mapping):
            for val in seq:
                if val not in mapping:
                    mapping[val] = len(mapping)

        def add_unique_file_id(table):
            table["unique_file_id"] = list(
                map(
                    lambda x: int(hashlib.md5(x.encode('utf-8')).hexdigest()[:16], 16) % 2147483000,
                    map(
                        lambda x: f"{x[0]}_{x[1]}",
                        zip(table["file_id"], table["package"])
                    )
                )
            )

        for edges in tqdm(read_edges(path, as_chunks=True), desc="Importing edges"):
            if "package" not in edges.columns:
                edges["package"] = edges["file_id"]

            add_unique_file_id(edges)

            update_mapping(edges["type"], type_map)
            update_mapping(edges["package"].dropna(), package_map)
            # update_mapping(edges["unique_file_id"], file_id_map)

            edges["type"] = edges["type"].apply(type_map.get)
            edges["package"] = edges["package"].apply(package_map.get)
            # edges["unique_file_id"] = edges["unique_file_id"].apply(file_id_map.get)

            edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1, inplace=True)

            self._write_non_empty_table(
                table=edges[["id", "type", "src", "dst"]],
                table_name="edges",
                create_index=["id", "type", "src", "dst"],
                dtype={
                    "id": AbstractDBStorage.DataTypes.INT_PRIMARY,
                    "type": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                    "src": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                    "dst": AbstractDBStorage.DataTypes.INT_NOT_NULL
                }
            )

            self._write_non_empty_table(
                table=edges[["id", "unique_file_id", "file_id", "package"]].dropna(),
                table_name="edge_file_id",
                create_index=["id", "unique_file_id", "file_id", "package"],
                dtype={
                    "id": AbstractDBStorage.DataTypes.INT_PRIMARY,
                    "unique_file_id": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                    "file_id": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                    "package": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                }
            )

            rest_columns = [
                col for col in edges.columns
                if col not in {"id", "src", "dst", "type", "unique_file_id", "file_id", "package"}
            ]

            self._write_non_empty_table(
                table=edges[["id"] + rest_columns],
                table_name="edge_hierarchy",
                create_index=["id", "mentioned_in"],
                dtype={
                    "id": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                    "mentioned_in": AbstractDBStorage.DataTypes.INT,
                }
            )

        self._write_type_map(type_map, "edge_types")
        self._write_package_map(package_map, "packages")
        self._write_node_file_id_table()

    def _import_filecontent(self, path):
        for filecontent in unpersist(path, chunksize=100000):
            self._write_non_empty_table(
                table=filecontent[["id", "content", "package"]].dropna(),
                table_name="filecontent",
                create_index=["id", "package"],
                dtype={
                    "id": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                    "content": AbstractDBStorage.DataTypes.TEXT_NOT_NULL,
                    "package": AbstractDBStorage.DataTypes.INT_NOT_NULL,
                }
            )

    def import_from_files(self, path_to_dataset):
        get_path = partial(join, path_to_dataset)
        self._import_nodes(get_path("common_nodes.json.bz2"))
        self._import_edges(get_path("common_edges.json.bz2"))
        # self._import_filecontent(get_path("common_filecontent.json.bz2"))

    @read_or_add_summary_field("num_nodes")
    def get_num_nodes(self):  # max() is used here to make it compatible with DGL
        return int(self.database.query("SELECT max(id) FROM nodes").iloc[0,0] + 1)

    @read_or_add_summary_field("num_edges")
    def get_num_edges(self):
        return int(self.database.query("SELECT count(id) FROM edges").iloc[0,0])

    def get_node_type_descriptions(self):
        return self.database.query("SELECT type_desc from node_types")["type_desc"]

    def get_edge_type_descriptions(self):
        return self.database.query("SELECT type_desc from edge_types")["type_desc"]

    def get_edge_types(self):
        table = self.database.query("SELECT * from edge_types")
        return dict(zip(table["type_id"], table["type_desc"]))

    def get_node_types(self):
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
                                """,
        )
        return nodes

    # def get_nodes_for_edges(self, edges):
    #     nodes = defaultdict(list)
    #     seen = set()
    #     for src, dst, src_type, dst_type, src_name, dst_name in edges[["src", "dst", "src_type", "dst_type", "src_name", "dst_name"]].values:
    #         if src not in seen:
    #             nodes["id"].append(src)
    #             nodes["type"].append(src_type)
    #             nodes["name"].append(src_name)
    #         if dst not in seen:
    #             nodes["id"].append(dst)
    #             nodes["type"].append(dst_type)
    #             nodes["name"].append(dst_name)
    #     return pd.DataFrame(nodes)

    def get_subgraph_for_package(self, package_id):
        edges = self.database.query(
            f"""
                        SELECT
                        edges.id as id, edge_types.type_desc as type, src, dst,
                        src_node_types.type_desc as src_type,
                        dst_node_types.type_desc as dst_type,
                        src_nodes.name as src_name,
                        dst_nodes.name as dst_name
                        FROM
                        edge_file_id
                        LEFT JOIN edges ON edge_file_id.id = edges.id
                        LEFT JOIN edge_types ON edges.type = edge_types.type_id
                        JOIN nodes as src_nodes ON src_nodes.id = edges.src
                        JOIN nodes as dst_nodes ON dst_nodes.id = edges.dst
                        JOIN node_types as src_node_types ON src_nodes.type = src_node_types.type_id
                        JOIN node_types as dst_node_types ON dst_nodes.type = dst_node_types.type_id
                        WHERE edge_file_id.package = '{package_id}'
                        """
        )

        package_nodes = self.get_nodes_for_edges(edges)
        return package_nodes, edges

    def get_all_packages(self):
        return self.database.query("SELECT package_id, package_desc FROM packages")["package_id"]

    def get_subgraph_for_file(self, file_id):
        edges = self.database.query(
            f"""
                        SELECT
                        edges.id as id, edge_types.type_desc as type, src, dst,
                        src_node_types.type_desc as src_type,
                        dst_node_types.type_desc as dst_type,
                        src_nodes.name as src_name,
                        dst_nodes.name as dst_name
                        FROM
                        edge_file_id
                        LEFT JOIN edges ON edge_file_id.id = edges.id
                        LEFT JOIN edge_types ON edges.type = edge_types.type_id
                        JOIN nodes as src_nodes ON src_nodes.id = edges.src
                        JOIN nodes as dst_nodes ON dst_nodes.id = edges.dst
                        JOIN node_types as src_node_types ON src_nodes.type = src_node_types.type_id
                        JOIN node_types as dst_node_types ON dst_nodes.type = dst_node_types.type_id
                        WHERE edge_file_id.unique_file_id = '{file_id}'
                        """
        )

        file_nodes = self.get_nodes_for_edges(edges)
        return file_nodes, edges

    def get_all_files(self):
        return self.database.query("SELECT distinct unique_file_id FROM edge_file_id")["unique_file_id"]

    def get_subgraph_for_mention(self, mention):
        edges = self.database.query(
            f"""
                        SELECT
                        edges.id as id, edge_types.type_desc as type, src, dst,
                        src_node_types.type_desc as src_type,
                        dst_node_types.type_desc as dst_type,
                        src_nodes.name as src_name,
                        dst_nodes.name as dst_name
                        FROM
                        edge_hierarchy
                        LEFT JOIN edges ON edge_hierarchy.id = edges.id
                        LEFT JOIN edge_types ON edges.type = edge_types.type_id
                        JOIN nodes as src_nodes ON src_nodes.id = edges.src
                        JOIN nodes as dst_nodes ON dst_nodes.id = edges.dst
                        JOIN node_types as src_node_types ON src_nodes.type = src_node_types.type_id
                        JOIN node_types as dst_node_types ON dst_nodes.type = dst_node_types.type_id
                        WHERE edge_hierarchy.mentioned_in = '{mention}'
                        """
        )

        mention_nodes = self.get_nodes_for_edges(edges)
        return mention_nodes, edges

    def get_all_mentions(self):
        return self.database.query("SELECT distinct mentioned_in FROM edge_hierarchy").dropna()["mentioned_in"].astype("int64")

    def iterate_subgraphs(self, how, groups):

        if how == SGPartitionStrategies.package:
            load_fn = self.get_subgraph_for_package
            if groups is None:
                groups = self.get_all_packages()
        elif how == SGPartitionStrategies.file:
            load_fn = self.get_subgraph_for_file
            if groups is None:
                groups = self.get_all_files()
        elif how == SGPartitionStrategies.mention:
            load_fn = self.get_subgraph_for_mention
            if groups is None:
                groups = self.get_all_mentions()
        else:
            raise ValueError()

        for group in groups:
            cache_key = self.get_cache_key(how, group)
            cached = None  # self.load_from_cache(cache_key)
            if cached is None:
                nodes, edges = load_fn(group)
                # self.write_to_cache(cache_key, (nodes, edges))
            else:
                nodes, edges = cached
            yield group, nodes, edges

    # @staticmethod
    # def get_temp_schema(table, table_name):
    #     return pd.io.sql.get_schema(table, table_name).replace("CREATE TABLE", "CREATE TEMPORARY TABLE")

    def _create_tmp_ids_list(self, ids):
        table_name = "tmp_partition_ids"
        list_of_ids = ",".join(f"({val})" for val in ids)
        insertion_query = f"insert into {table_name}(ids) values {list_of_ids}"

        self.database.execute(f"DROP TABLE IF EXISTS {table_name}", commit=False)
        self.database.execute(f"CREATE TEMPORARY TABLE {table_name}(ids INTEGER)", commit=False)
        self.database.execute(insertion_query, commit=False)
        self.database.execute(
            f"CREATE INDEX IF NOT EXISTS idx_tmp_{table_name} ON {table_name}(ids)",
            commit=True
        )
        return table_name

    def _drop_tmp_ids_list(self, table_name):
        self.database.execute(f"DROP TABLE {table_name}")

    def get_info_for_subgraphs(self, subgraph_ids, field):
        info_column = subgraph_ids.copy()

        if field == SGPartitionStrategies.package:
            column_name = "package"
            package_data = self.database.query(
                f"""
                SELECT distinct package_desc, package_id from packages
                """
            )

            package_2_package_id = dict(zip(package_data["package_desc"], package_data["package_id"]))
            info_column = info_column.map(package_2_package_id.get)
        elif field == SGPartitionStrategies.file:
            column_name = "unique_file_id"

            file_id_data = self.database.query(
                f"""
                SELECT distinct file_id, unique_file_id from edge_file_id
                """
            )

            assert file_id_data["file_id"].nunique() == file_id_data["unique_file_id"].nunique()

            file_is_2_unique_file_id = dict(zip(file_id_data["file_id"], file_id_data["unique_file_id"]))

            info_column = info_column.map(file_is_2_unique_file_id.get)
        elif field == SGPartitionStrategies.mention:
            column_name = "mentioned_in"
        else:
            raise ValueError()

        results = subgraph_ids.to_frame()
        results[column_name] = info_column
        return results, column_name

    def get_info_for_node_ids(self, node_ids, field):
        if field == SGPartitionStrategies.package:
            column_name = "package"
        elif field == SGPartitionStrategies.file:
            column_name = "unique_file_id"
        elif field == SGPartitionStrategies.mention:
            column_name = "mentioned_in"
        else:
            raise ValueError()

        ids_table_name = self._create_tmp_ids_list(node_ids)
        # results = self.database.query(
        #     f"""
        #     select distinct src, {column_name} from (
        #         select src, package, unique_file_id, mentioned_in
        #         from {node_ids_table_name}
        #         inner join edges on edges.src = {node_ids_table_name}.node_ids
        #         inner join edge_file_id on edges.id = edge_file_id.id
        #         join edge_hierarchy on edges.id = edge_hierarchy.id
        #         union
        #         select dst as src, package, unique_file_id, mentioned_in
        #         from {node_ids_table_name}
        #         inner join edges on edges.dst = {node_ids_table_name}.node_ids
        #         inner join edge_file_id on edges.id = edge_file_id.id
        #         join edge_hierarchy on edges.id = edge_hierarchy.id
        #     ) as node_info where {column_name} is not null order by {column_name}
        #     """
        # )
        results = self.database.query(
            f"""
            select distinct node_file_id.id, {column_name}
            from {ids_table_name}
            inner join node_file_id on node_file_id.id = {ids_table_name}.ids
            join node_hierarchy on node_file_id.id = node_hierarchy.id
            """
        )

        self._drop_tmp_ids_list(ids_table_name)
        return results, column_name

    def get_info_for_edge_ids(self, edge_ids, field):
        if field == SGPartitionStrategies.package:
            column_name = "package"
        elif field == SGPartitionStrategies.file:
            column_name = "unique_file_id"
        elif field == SGPartitionStrategies.mention:
            column_name = "mentioned_in"
        else:
            raise ValueError()

        ids_table_name = self._create_tmp_ids_list(edge_ids)

        results = self.database.query(
            f"""
            select distinct edge_file_id.id, {column_name}
            from {ids_table_name}
            inner join edge_file_id on edge_file_id.id = {ids_table_name}.ids
            join edge_hierarchy on edge_file_id.id = edge_hierarchy.id
            """
        )

        self._drop_tmp_ids_list(ids_table_name)
        return results, column_name


class OnDiskGraphStorageWithFastIteration(OnDiskGraphStorage):
    def get_iteration_groups_if_necessary(self, how, groups):
        if groups is None:
            if how == SGPartitionStrategies.package:
                groups = self.get_all_packages()
            elif how == SGPartitionStrategies.file:
                groups = self.get_all_files()
            elif how == SGPartitionStrategies.mention:
                groups = self.get_all_mentions()
            else:
                raise ValueError()
        return groups

    def get_iteration_request_and_columns_and_group_encoder(self, how, groups):
        requested_partition_groups = ",".join(map(str, groups))
        if how == SGPartitionStrategies.package:
            requested_partition_groups = ",".join(map(repr, groups))
            query_str = f"""
                                SELECT
                                edges.id as id, edge_types.type_desc as type, src, dst,  
--                                 src_node_types.type_desc as src_type,
--                                 dst_node_types.type_desc as dst_type,
--                                 src_nodes.name as src_name,
--                                 dst_nodes.name as dst_name,
                                unique_file_id, package
                                FROM
                                edges
                                INNER JOIN edge_file_id ON edge_file_id.id = edges.id
                                INNER JOIN edge_types ON edges.type = edge_types.type_id
--                                 JOIN nodes as src_nodes ON src_nodes.id = edges.src
--                                 JOIN nodes as dst_nodes ON dst_nodes.id = edges.dst
--                                 JOIN node_types as src_node_types ON src_nodes.type = src_node_types.type_id
--                                 JOIN node_types as dst_node_types ON dst_nodes.type = dst_node_types.type_id
                                WHERE package in ({requested_partition_groups})
                                """
            partition_columns = ["package"]
            group_encoder = lambda x: x[0]
        elif how == SGPartitionStrategies.file:
            query_str = f"""
                                SELECT
                                edges.id as id, edge_types.type_desc as type, src, dst, 
--                                 src_node_types.type_desc as src_type,
--                                 dst_node_types.type_desc as dst_type,
--                                 src_nodes.name as src_name,
--                                 dst_nodes.name as dst_name,
                                unique_file_id, file_id, package
                                FROM
                                edges
                                INNER JOIN edge_file_id ON edge_file_id.id = edges.id
                                INNER JOIN edge_types ON edges.type = edge_types.type_id
--                                 JOIN nodes as src_nodes ON src_nodes.id = edges.src
--                                 JOIN nodes as dst_nodes ON dst_nodes.id = edges.dst
--                                 JOIN node_types as src_node_types ON src_nodes.type = src_node_types.type_id
--                                 JOIN node_types as dst_node_types ON dst_nodes.type = dst_node_types.type_id
                                WHERE unique_file_id in ({requested_partition_groups})
                                """
            partition_columns = ["unique_file_id"]  # , "file_id"]
            group_encoder = lambda x: x  # x[1]
        elif how == SGPartitionStrategies.mention:
            query_str = f"""
                                SELECT
                                edges.id as id, edge_types.type_desc as type, src, dst,
--                                 src_node_types.type_desc as src_type,
--                                 dst_node_types.type_desc as dst_type,
--                                 src_nodes.name as src_name,
--                                 dst_nodes.name as dst_name,
                                mentioned_in
                                FROM
                                edges
                                INNER JOIN edge_hierarchy ON edge_hierarchy.id = edges.id
                                INNER JOIN edge_types ON edges.type = edge_types.type_id
--                                 JOIN nodes as src_nodes ON src_nodes.id = edges.src
--                                 JOIN nodes as dst_nodes ON dst_nodes.id = edges.dst
--                                 JOIN node_types as src_node_types ON src_nodes.type = src_node_types.type_id
--                                 JOIN node_types as dst_node_types ON dst_nodes.type = dst_node_types.type_id
                                WHERE mentioned_in in ({requested_partition_groups})
                                """
            partition_columns = ["mentioned_in"]
            group_encoder = lambda x: x[0]
        else:
            raise ValueError()

        return query_str, partition_columns, group_encoder

    @staticmethod
    def get_next_chunk(iterator, output_loc: asyncio.Future=None):
        try:
            chunk = next(iterator)
        except StopIteration:
            chunk = None
        if output_loc is not None:
            output_loc.set_result(chunk)
        return chunk

    def iterate_subgraphs(self, how, groups):
        groups = self.get_iteration_groups_if_necessary(how, groups)
        query_str, partition_columns, group_encoder = self.get_iteration_request_and_columns_and_group_encoder(how, groups)

        iterator = iter(self.database.query(query_str, chunksize=10000))

        current_part = None
        prev_partition_value = None

        def merge_current_part(current_part, new_part):
            if current_part is None:
                return new_part
            return pd.concat([current_part, new_part])

        # next_edges = self.get_next_chunk(iterator)
        next_edges = asyncio.Future()
        # logging.info("Starting new thread to receive a chunk")
        th = Thread(target=self.get_next_chunk, name="Get chunk", args=(iterator, next_edges))
        th.start()

        served = 0

        while True:
            try:
                # edges = next_edges
                # logging.info("Joining the thread with a chunk")
                th.join()
                edges = next_edges.result()
                if edges is None:
                    raise StopIteration
                # logging.info(f"New chunk has {len(edges)} edges")
                # next_edges = self.get_next_chunk(iterator)
                next_edges = asyncio.Future()
                # logging.info("Starting new thread to receive a chunk")
                th = Thread(target=self.get_next_chunk, name="Get chunk", args=(iterator, next_edges))
                th.start()
                start = 0

                partition_c = edges[partition_columns]

                # logging.info(f"Looking through the current chunk")
                for ind, val in enumerate(partition_c.values.tolist()):
                    if val != prev_partition_value and prev_partition_value is not None:
                        end = ind
                        if end > 0:
                            current_part = merge_current_part(current_part, edges[start: end])
                        partition_edges = current_part
                        partition_nodes = self.get_nodes_for_edges(partition_edges)
                        # logging.info(f"Found subgraph with {len(partition_nodes)} nodes and {len(partition_edges)} edges")
                        # if served >= 50:
                        #     raise StopIteration
                        served += 1
                        yield group_encoder(prev_partition_value), partition_nodes, partition_edges
                        current_part = None
                        start = end
                    prev_partition_value = val
                # logging.info(f"Finished with the current chunk")
                if start != len(edges):
                    current_part = merge_current_part(current_part, edges[start: len(edges)])

            except StopIteration:
                # logging.info(f"Last chunk")
                partition_edges = current_part  # may be empty
                partition_nodes = self.get_nodes_for_edges(partition_edges)
                # logging.info(f"Found subgraph with {len(partition_nodes)} nodes and {len(partition_edges)} edges")
                yield group_encoder(prev_partition_value), partition_nodes, partition_edges
                break


class OnDiskGraphStorageWithFastIterationNoPandas(OnDiskGraphStorageWithFastIteration):
    def get_iteration_request_and_columns_and_group_encoder(self, how, groups):
        requested_partition_groups = ",".join(map(str, groups))
        if how == SGPartitionStrategies.package:
            requested_partition_groups = ",".join(map(repr, groups))
            query_str = f"""
                                SELECT
                                edges.id as id, edge_types.type_desc as type, src, dst,  
                                package
                                FROM
                                edges
                                INNER JOIN edge_file_id ON edge_file_id.id = edges.id
                                INNER JOIN edge_types ON edges.type = edge_types.type_id
                                WHERE package in ({requested_partition_groups})
                                """
            partition_columns = ["package"]
            group_encoder = lambda x: x[0]
        elif how == SGPartitionStrategies.file:
            query_str = f"""
                                SELECT
                                edges.id as id, edge_types.type_desc as type, src, dst, 
                                unique_file_id 
                                FROM
                                edges
                                INNER JOIN edge_file_id ON edge_file_id.id = edges.id
                                INNER JOIN edge_types ON edges.type = edge_types.type_id
                                WHERE unique_file_id in ({requested_partition_groups})
                                """
            partition_columns = ["unique_file_id"]  # , "file_id"]
            group_encoder = lambda x: x  # x[1]
        elif how == SGPartitionStrategies.mention:
            query_str = f"""
                                SELECT
                                edges.id as id, edge_types.type_desc as type, src, dst,
                                mentioned_in
                                FROM
                                edges
                                INNER JOIN edge_hierarchy ON edge_hierarchy.id = edges.id
                                INNER JOIN edge_types ON edges.type = edge_types.type_id
                                WHERE mentioned_in in ({requested_partition_groups})
                                """
            partition_columns = ["mentioned_in"]
            group_encoder = lambda x: x[0]
        else:
            raise ValueError()

        return query_str, partition_columns, group_encoder

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
                                """,
            as_table=False,
            column_names=["id", "type", "name"]
        )
        return nodes

    def iterate_subgraphs(self, how, groups):
        groups = self.get_iteration_groups_if_necessary(how, groups)
        query_str, partition_columns, group_encoder = self.get_iteration_request_and_columns_and_group_encoder(how, groups)

        iterator = iter(self.database.query(
            query_str, chunksize=10000, as_table=False,
            column_names=["id", "type", "src", "dst", partition_columns[0]]
        ))

        current_part = None
        prev_partition_value = None

        def merge_current_part(current_part, new_part):
            if current_part is None:
                return new_part
            return current_part.extend(new_part)

        next_edges = self.get_next_chunk(iterator)
        # next_edges = asyncio.Future()
        # # logging.info("Starting new thread to receive a chunk")
        # th = Thread(target=self.get_next_chunk, name="Get chunk", args=(iterator, next_edges))
        # th.start()

        served = 0

        while True:
            try:
                edges = next_edges
                # logging.info("Joining the thread with a chunk")
                # th.join()
                # edges = next_edges.result()
                if edges is None:
                    raise StopIteration
                # logging.info(f"New chunk has {len(edges)} edges")
                next_edges = self.get_next_chunk(iterator)
                # next_edges = asyncio.Future()
                # # logging.info("Starting new thread to receive a chunk")
                # th = Thread(target=self.get_next_chunk, name="Get chunk", args=(iterator, next_edges))
                # th.start()
                start = 0

                # partition_c = edges[partition_columns]

                # logging.info(f"Looking through the current chunk")
                # for ind, val in enumerate(partition_c.values.tolist()):
                for ind, val in enumerate(edges[partition_columns[0]]):
                    if val != prev_partition_value and prev_partition_value is not None:
                        end = ind
                        if end > 0:
                            current_part = merge_current_part(current_part, edges[start: end])
                        partition_edges = current_part
                        partition_nodes = self.get_nodes_for_edges(partition_edges)
                        # logging.info(f"Found subgraph with {len(partition_nodes)} nodes and {len(partition_edges)} edges")
                        # if served >= 50:
                        #     raise StopIteration
                        served += 1
                        yield group_encoder(prev_partition_value), partition_nodes, partition_edges
                        current_part = None
                        start = end
                    prev_partition_value = val
                # logging.info(f"Finished with the current chunk")
                if start != len(edges):
                    current_part = merge_current_part(current_part, edges[start: len(edges)])

            except StopIteration:
                # logging.info(f"Last chunk")
                if current_part is not None:
                    partition_edges = current_part
                    partition_nodes = self.get_nodes_for_edges(partition_edges)
                    # logging.info(f"Found subgraph with {len(partition_nodes)} nodes and {len(partition_edges)} edges")
                    yield group_encoder(prev_partition_value), partition_nodes, partition_edges
                break


class InMemoryGraphStorage(AbstractGraphStorage):
    def __init__(self, nodes, edges, add_type_nodes=False):
        if add_type_nodes:
            nodes, edges = self._add_type_nodes(nodes, edges)
        self._nodes = nodes
        self._edges = edges

    def get_nodes_for_classification(self):
        pass

    def get_info_for_subgraphs(self, subgraph_ids, field):
        pass

    def get_node_types(self):
        pass

    @staticmethod
    def _add_type_nodes(nodes, edges):
        node_new_id = nodes["id"].max() + 1
        edge_new_id = edges["id"].max() + 1

        new_nodes = []
        new_edges = []
        added_type_nodes = {}

        node_slice = nodes[["id", "type"]].values

        for id, type in node_slice:
            if type not in added_type_nodes:
                added_type_nodes[type] = node_new_id
                node_new_id += 1

                new_nodes.append({
                    "id": added_type_nodes[type],
                    "name": type,
                    "type": "type_node",
                })

            new_edges.append({
                "id": edge_new_id,
                "type": "node_type",
                "src": added_type_nodes[type],
                "dst": id,
            })
            edge_new_id += 1

        new_nodes, new_edges = pd.DataFrame(new_nodes), pd.DataFrame(new_edges)

        return nodes.append(new_nodes), edges.append(new_edges)

    @classmethod
    def get_node_type_descriptions(cls):
        return [
            "module", "global_variable", "non_indexed_symbol", "class", "function", "class_field", "class_method",
            "subword", "mention", "Module", "ImportFrom", "alias", "Import", "Attribute", "#attr#", "Call", "Subscript",
            "Index", "FunctionDef", "arg", "arguments", "Compare", "Op", "Constant", "BoolOp", "If", "BinOp", "Tuple",
            "Assign", "Raise", "Return", "#keyword#", "keyword", "For", "List", "Try", "ExceptHandler", "UnaryOp",
            "ListComp", "comprehension", "ClassDef", "IfExp", "Slice", "CtlFlow", "CtlFlowInstance", "Dict",
            "GeneratorExp", "Yield", "While", "Starred", "Global", "withitem", "With", "AugAssign", "Delete", "Lambda",
            "DictComp", "Set", "Assert", "SetComp", "ExtSlice", "AsyncFunctionDef", "Await", "AnnAssign", "JoinedStr",
            "AsyncWith", "Nonlocal", "ast_Literal", "YieldFrom", "AsyncFor"
        ]

    @classmethod
    def get_edge_type_descriptions(cls):
        return [
            "defines", "uses", "imports", "calls", "uses_type", "inheritance", "defined_in", "used_by", "imported_by",
            "called_by", "type_used_by", "inherited_by", "subword", "module", "global_mention", "module_rev", "name",
            "name_rev", "names", "names_rev", "defined_in_module", "defined_in_module_rev", "next", "prev", "value",
            "value_rev", "attr", "func", "func_rev", "slice", "slice_rev", "args", "args_rev", "arg", "arg_rev", "left",
            "left_rev", "ops", "comparators", "values", "values_rev", "op", "test", "test_rev", "elts", "elts_rev",
            "right", "right_rev", "targets", "targets_rev", "executed_if_true", "executed_if_true_rev", "exc",
            "exc_rev", "defined_in_function", "defined_in_function_rev", "function_name", "function_name_rev",
            "keywords", "keywords_rev", "kwarg", "kwarg_rev", "executed_if_false", "executed_if_false_rev", "target",
            "target_rev", "iter", "iter_rev", "executed_in_for", "executed_in_for_rev", "comparators_rev",
            "executed_in_try", "executed_in_try_rev", "type", "type_rev", "executed_with_try_handler",
            "executed_with_try_handler_rev", "operand", "operand_rev", "elt", "elt_rev", "generators", "generators_rev",
            "defined_in_class", "defined_in_class_rev", "class_name", "class_name_rev", "decorator_list",
            "decorator_list_rev", "if_true", "if_true_rev", "if_false", "if_false_rev", "lower", "upper", "upper_rev",
            "control_flow", "lower_rev", "executed_while_true", "executed_while_true_rev", "asname", "asname_rev",
            "keys", "vararg", "vararg_rev", "executed_in_try_final", "executed_in_try_final_rev", "ifs", "ifs_rev",
            "context_expr", "context_expr_rev", "optional_vars", "optional_vars_rev", "items", "items_rev",
            "executed_inside_with", "executed_inside_with_rev", "lambda", "lambda_rev", "key", "key_rev",
            "executed_in_try_else", "executed_in_try_else_rev", "msg", "msg_rev", "keys_rev", "step", "step_rev",
            "executed_in_for_orelse", "executed_in_for_orelse_rev", "dims", "dims_rev", "kwonlyargs", "kwonlyargs_rev",
            "cause", "cause_rev"
        ]

    def get_num_nodes(self):
        return len(self._nodes)

    def get_num_edges(self):
        return len(self._edges)

    def get_nodes(self, type_filter=None):
        if type_filter is not None:
            return self._nodes.query("type in @type_filter", local_dict={"type_filter": type_filter})
        else:
            return self._nodes.copy()

    def get_edges(self, type_filter=None):
        if type_filter is not None:
            return self._edges.query("type in @type_filter", local_dict={"type_filter": type_filter})
        else:
            return self._edges.copy()

    def iterate_subgraphs(self, how, ids):
        yield 0, self._nodes.copy(), self._edges.copy()

    def get_nodes_with_subwords(self):
        return list(set(self._edges.query("type == 'subword'")["dst"]))

    def get_info_for_node_ids(self, node_ids, group_by):
        pd.DataFrame({"ids": node_ids, "group": [0] * len(node_ids)})
        node_info = self._nodes[["id"]]
        node_info["group"] = 0
        return node_info

    def get_info_for_edge_ids(self, edge_ids, group_by):
        edge_info = self._edges.query("id in @edge_ids", local_dict={"edge_ids": edge_ids})[["id"]].rename({"id": "src"}, axis=1)
        edge_info["group"] = 0
        return edge_info, "group"


class InMemoryGraphStorageNoPandas(InMemoryGraphStorage):
    def __init__(self, nodes, edges, add_type_nodes=False):
        if add_type_nodes:
            nodes, edges = self._add_type_nodes(nodes, edges)
        self._nodes = nodes
        self._edges = edges

    def get_nodes_for_classification(self):
        pass

    def get_info_for_subgraphs(self, subgraph_ids, field):
        pass

    def get_node_types(self):
        pass

    @staticmethod
    def _add_type_nodes(nodes, edges):
        node_new_id = nodes["id"].max() + 1
        edge_new_id = edges["id"].max() + 1

        new_nodes = []
        new_edges = []
        added_type_nodes = {}

        node_slice = nodes[["id", "type"]].values

        for id, type in node_slice:
            if type not in added_type_nodes:
                added_type_nodes[type] = node_new_id
                node_new_id += 1

                new_nodes.append({
                    "id": added_type_nodes[type],
                    "name": type,
                    "type": "type_node",
                })

            new_edges.append({
                "id": edge_new_id,
                "type": "node_type",
                "src": added_type_nodes[type],
                "dst": id,
            })
            edge_new_id += 1

        new_nodes, new_edges = pd.DataFrame(new_nodes), pd.DataFrame(new_edges)

        return nodes.append(new_nodes), edges.append(new_edges)

    def get_num_nodes(self):
        return len(self._nodes)

    def get_num_edges(self):
        return len(self._edges)

    def get_nodes(self, type_filter=None):
        if type_filter is not None:
            return self._nodes.query("type in @type_filter", local_dict={"type_filter": type_filter})
        else:
            return self._nodes

    def get_edges(self, type_filter=None):
        if type_filter is not None:
            return self._edges.query("type in @type_filter", local_dict={"type_filter": type_filter})
        else:
            return self._edges

    def iterate_subgraphs(self, how, ids):
        yield 0, self._nodes, self._edges

    def get_nodes_with_subwords(self):
        return list(set(self._edges.query("type == 'subword'")["dst"]))

    def get_info_for_node_ids(self, node_ids, group_by):
        pd.DataFrame({"ids": node_ids, "group": [0] * len(node_ids)})
        node_info = self._nodes[["id"]]
        node_info["group"] = 0
        return node_info

    def get_info_for_edge_ids(self, edge_ids, group_by):
        edge_info = self._edges.query("id in @edge_ids", local_dict={"edge_ids": edge_ids})[["id"]].rename({"id": "src"}, axis=1)
        edge_info["group"] = 0
        return edge_info, "group"




def test_subgraph_iteration():
    storage = OnDiskGraphStorageWithFastIterationNoPandas("/Users/LTV/Downloads/NitroShare/codeseatchnet_dedicated_type_pred/dataset.db")
    for subgraph in tqdm(storage.iterate_subgraphs(SGPartitionStrategies["file"], None)):
        pass


if __name__ == "__main__":
    test_subgraph_iteration()