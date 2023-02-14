from collections import Counter
from pathlib import Path

from tqdm import tqdm

from SourceCodeTools.code.data.dataset.partition_strategies import SGPartitionStrategies
from SourceCodeTools.code.data.file_utils import unpersist, read_mapping_from_json, write_mapping_to_json


def is_int(value):
    try:
        int(value)
        return True
    except:
        return False


def read_or_add_summary_field(field):
    def read_or_add_summary(function):
        def wrapper(self, summary_filename="summary.json"):
            dataset_summary_path = self._path.joinpath(summary_filename)

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


class LocationIterator:
    def __init__(self, dataset_path, location_filepath):
        self._dataset_path = dataset_path
        self._location_filepath = location_filepath
        self._opened_file = None

    def _read_num_locations(self):
        with open(self._location_filepath, "r") as src:
            self._num_locations = int(src.readline().strip())

    def __iter__(self):
        if self._opened_file is not None:
            if self._opened_file.closed is False:
                self._opened_file.close()

        self._opened_file = open(self._location_filepath, "r")
        self._opened_file.readline()  # first line is the number of locations
        return self

    def __next__(self):
        return self._dataset_path.joinpath(self._opened_file.readline().strip())

    def __len__(self):
        return self._num_locations

    @staticmethod
    def write_locations(locations, path):
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as sink:
            sink.write(f"{len(locations)}\n")
            for loc in locations:
                sink.write(f"{str(loc)}\n")


class FileSystemStorage:
    def __init__(self, path):
        self._path = Path(path)
        self._partition_cache = dict()

    @staticmethod
    def get_cache_key(how, group):
        return f"{how.name}_{group}"

    @staticmethod
    def labels_present(path: Path, labels_filename, force_labels):
        if  labels_filename is None:
            return True

        labels_filename = path.joinpath(labels_filename)
        if force_labels and not labels_filename.is_file():
            return False
        return True

    @staticmethod
    def attach_labels_from(table, path, labels_filename):
        if labels_filename is not None:
            return table.merge(unpersist(path.joinpath(labels_filename)), left_on="id", right_on="src")
        else:
            return table

    def _determine_partition(self, path):
        current_mention = path.name

        if current_mention not in self._partition_cache:
            while True:
                metadata_path = path.joinpath("metadata.json")
                if metadata_path.is_file():
                    mt = read_mapping_from_json(metadata_path)
                    if "partition" in mt:
                        self._partition_cache[current_mention] = mt["partition"]
                        break
                path = path.parent
                if path.name in self._partition_cache:
                    self._partition_cache[current_mention] = self._partition_cache[path.name]
                    break
                if not is_int(path.name):
                    raise Exception("No information about partition")

        return self._partition_cache[current_mention]

    def belongs_to_partition(self, path, partition):
        return self._determine_partition(path) == partition

    def _load_graph(
            self, path: Path, node_labels_filename=None, edge_labels_filename=None, force_labels=True, partition=None
    ):
        nodes_path = path.joinpath("nodes.parquet")
        edges_path = path.joinpath("edges.parquet")

        if partition is not None and not self.belongs_to_partition(path, partition):
            return None

        nodes = None
        edges = None
        if (
                nodes_path.is_file() and
                self.labels_present(path, node_labels_filename, force_labels) and
                self.labels_present(path, edge_labels_filename, force_labels)
        ):
            nodes = self.attach_labels_from(unpersist(nodes_path), path, node_labels_filename)
            edges = self.attach_labels_from(unpersist(edges_path), path, edge_labels_filename)

        def extend_if_needed(table, table_):
            if table is None:
                return table_
            elif table_ is None:
                return table
            else:
                return table.append(table_)

        for nested_graph in path.iterdir():
            if nested_graph.is_dir():
                subgraph = self._load_graph(nested_graph, node_labels_filename, edge_labels_filename)
                nodes = extend_if_needed(nodes, subgraph["nodes"]).drop_duplicates(subset="id")
                edges = extend_if_needed(edges, subgraph["edges"])

        return {
            "path": path, "nodes": nodes, "edges": edges
        }

    def _stream_packages(self):
        for package_path in self._path.iterdir():
            if package_path.is_dir():
                yield package_path

    def _stream_files(self):
        for package_path in self._stream_packages():
            for file_path in package_path.iterdir():
                if file_path.is_dir():
                    yield file_path

    def _stream_leaf_mentions(self, leaf_file="nodes.parquet"):
        for file_path in self._stream_files():
            for mention_path in file_path.iterdir():
                if mention_path.is_dir():
                    for leaf_mention in self._iterate_leafs(mention_path, leaf_file):
                        yield leaf_mention

    def stream_locations_from_file(self, file_path):
        with open(file_path, "r") as location_src:
            num_locations = int(location_src.readline())
            for line in location_src:
                yield self._path.joinpath(line.strip())

    def _iterate_leafs(self, path, leaf_file="nodes.parquet"):
        is_leaf = True
        for subdir in path.iterdir():
            if subdir.is_dir():
                for leaf_mention in self._iterate_leafs(subdir, leaf_file):
                    if not leaf_mention.joinpath(leaf_file).is_file():
                        continue
                    is_leaf = False
                    yield leaf_mention

        if is_leaf == True and path.joinpath(leaf_file).is_file():
            yield path

    def _iterate_all(self, path, need_file="nodes.parquet"):
        if path.joinpath(need_file).is_file():
            yield path

        for subdir in path.iterdir():
            if subdir.is_dir():
                for loc in self._iterate_all(subdir, need_file):
                    yield loc

    # - type distribution seems to be odd. not as many floats as before. maybe need to include those hidden packages.
    # - probably do not need to include only leaf mentions because type edges are incapsulated in a mention
    # - do not exclude type edges by default.
    # - it is probably enough to mark packages that have labels, do not store all mention locations
    def collect_objective_summary(self, labels_filename):
        summary_path = self._path.joinpath(".summary").joinpath("summary_" + labels_filename.split(".")[0] + ".json")
        labels_locations = self._path.joinpath(".summary").joinpath("location_" + labels_filename.split(".")[0] + "txt")
        locations = []

        if labels_locations.is_file():
            locations = None

        summary = {
            "train": Counter(),
            "test": Counter(),
            "val": Counter()
        }
        for loc_path in tqdm(self._stream_leaf_mentions(labels_filename)):
            if locations is not None:
                locations.append(loc_path.relative_to(self._path))
            partition = self._determine_partition(loc_path)
            targets = Counter(unpersist(loc_path.joinpath(labels_filename))["dst"])
            summary[partition] |= targets

        summary = {
            "train": dict(summary["train"].most_common()),
            "test": dict(summary["test"].most_common()),
            "val": dict(summary["val"].most_common()),
        }

        summary_path.parent.mkdir(exist_ok=True)
        write_mapping_to_json(summary, summary_path)
        if locations is not None:
            LocationIterator.write_locations(locations, labels_locations)

    def iterate_packages(
            self, node_labels_file=None, edge_labels_file=None, force_labels=True, objective_type=None
    ):
        # partitions are assigned for files and are irrelevant for packages
        for package in self._stream_packages():
            yield self._load_graph(package, node_labels_file, edge_labels_file, force_labels)

    def iterate_files(
            self, node_labels_file=None, edge_labels_file=None, force_labels=True, partition=None, objective_type=None
    ):
        for file in self._stream_files():
            graph = self._load_graph(file, node_labels_file, edge_labels_file, force_labels, partition)
            if graph is not None:
                yield graph

    def iterate_leaf_mentions(
            self, node_labels_file=None, edge_labels_file=None, force_labels=True, partition=None, objective_type=None
    ):
        for mention in self._stream_leaf_mentions():
            graph = self._load_graph(mention, node_labels_file, edge_labels_file, force_labels, partition)
            if graph is not None:
                yield graph

    def iterate_all_mentions(
            self, node_labels_file=None, edge_labels_file=None, force_labels=True, partition=None, objective_type=None
    ):
        for file_path in self._stream_files():
            for mention in self._iterate_all(file_path):
                graph = self._load_graph(mention, node_labels_file, edge_labels_file, force_labels, partition)
                if graph is not None:
                    yield graph

    def _stream_fs_files(self, need_file):
        locations_path = self._path.joinpath(".summary").joinpath("location_" + need_file.split(".")[0] + "txt")
        # labels_locations = self._path.joinpath(".summary").joinpath("location_" + need_file.split(".")[0] + "txt")
        #
        # if labels_locations.is_file():
        #     location_iterator = LocationIterator(self._path, labels_locations)
        # else:
        def iterate_all():
            for package_path in self._stream_packages():
                for loc in self._iterate_all(package_path, need_file=need_file):
                    file_path = loc.joinpath(need_file)
                    yield file_path
                    # yield unpersist(file_path)

        location_iterator = iterate_all()

        for location in location_iterator:
            yield location

    def _stream_nodes(self):
        for nodes in self._stream_fs_files(need_file="nodes.parquet"):
            yield unpersist(nodes)
            # yield nodes

    def _stream_edges(self):
        for edges in self._stream_fs_files(need_file="edges.parquet"):
            yield unpersist(edges)
            # yield nodes

    @read_or_add_summary_field("number_of_nodes")
    def get_num_nodes(self):
        unique_nodes = set()
        for nodes in tqdm(self._stream_nodes(), desc="Computing number of nodes"):
            unique_nodes.update(nodes["id"])
        return len(unique_nodes)

    @read_or_add_summary_field("number_of_edges")
    def get_num_edges(self):
        unique_edges = set()
        for edges in tqdm(self._stream_edges(), desc="Computing number of edges"):
            unique_edges.update(edges["id"])
        return len(unique_edges)

    @read_or_add_summary_field("node_types")
    def get_node_types(self):
        unique_node_types = set()
        for nodes in tqdm(self._stream_nodes(), desc="Computing unique node types"):
            unique_node_types.update(nodes["type"])
        unique_node_types = list(unique_node_types)
        unique_node_types.sort()
        return unique_node_types

    @read_or_add_summary_field("edge_types")
    def get_edge_types(self):
        unique_edge_types = set()
        for edges in tqdm(self._stream_edges(), desc="Computing unique edge types"):
            unique_edge_types.update(edges["type"])
        unique_edge_types = list(unique_edge_types)
        unique_edge_types.sort()
        return unique_edge_types

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

    def get_subgraph_for_file(self, file_id):
        edges = self.database.query(
            f"""
                        SELECT
                        edges.id as id, edge_types.type_desc as type, src, dst
                        FROM
                        edge_file_id
                        LEFT JOIN edges ON edge_file_id.id = edges.id
                        LEFT JOIN edge_types ON edges.type = edge_types.type_id
                        WHERE edge_file_id.unique_file_id = '{file_id}'
                        """
        )

        file_nodes = self.get_nodes_for_edges(edges)
        return file_nodes, edges

    def get_all_files(self):
        return self.database.query("SELECT distinct unique_file_id FROM edge_file_id").values

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

    def iterate_subgraphs(self, how, partition=None, node_labels_file=None, edge_labels_file=None):

        if how == SGPartitionStrategies.package:
            for subgraph in self.iterate_packages(node_labels_file, edge_labels_file, partition):
                yield subgraph
        elif how == SGPartitionStrategies.file:
            for subgraph in self.iterate_files(node_labels_file, edge_labels_file, partition):
                yield subgraph
        elif how == SGPartitionStrategies.mention:
            for subgraph in self.iterate_all_mentions(node_labels_file, edge_labels_file, partition):
                yield subgraph
        else:
            raise ValueError()

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

        node_ids_table_name = self._create_tmp_node_ids_list(node_ids, )
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
            from {node_ids_table_name}
            inner join node_file_id on node_file_id.id = {node_ids_table_name}.node_ids
            join node_hierarchy on node_file_id.id = node_hierarchy.id
            """
        )

        self._drop_tmp_node_ids_list(node_ids_table_name)
        return results, column_name

if __name__ == "__main__":
    s = FileSystemStorage("/Users/LTV/Documents/popular_packages/graph")
    s.collect_objective_summary("type_ann_labels.parquet")
    # for subgraph in tqdm(s.iterate_all_mentions(partition="train", objective_type="classification"), desc="Iterating mentions"):
    #     pass
    # for p in s.iterate_all_mentions():
    # # for p in s._load_graph(s._path):
    #     print(p)