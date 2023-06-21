from pathlib import Path

from tqdm import tqdm

from SourceCodeTools.code.data.GraphStorage import AbstractGraphStorage
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
        self._prev_position = -1

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
        next_line = self._opened_file.readline().strip()
        next_position = self._opened_file.tell()
        if next_position == self._prev_position:
            raise StopIteration
        self._prev_position = next_position
        return self._dataset_path.joinpath(next_line)

    def __len__(self):
        return self._num_locations

    @staticmethod
    def write_locations(locations, path):
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as sink:
            sink.write(f"{len(locations)}\n")
            for loc in locations:
                sink.write(f"{str(loc)}\n")


class FileSystemStorage(AbstractGraphStorage):
    summary_dir = ".summary"

    def __init__(self, path):
        self._path = Path(path)
        assert self._path.is_dir(), f"Specified directory does not exist: {self._path}"
        self._partition_cache = dict()

    @staticmethod
    def get_cache_key(how, group):
        return f"{how.name}_{group}"

    @staticmethod
    def labels_present(path: Path, labels_filename, force_labels):
        if labels_filename is None:
            return True

        labels_present = False
        if isinstance(labels_filename, list):
            for file_ in labels_filename:
                labels_present = labels_present or path.joinpath(file_).is_file()
        else:
            raise NotImplementedError
            # labels_filename = path.joinpath(labels_filename)
            # if force_labels and not labels_filename.is_file():
            #     return False
        return labels_present

    @staticmethod
    def attach_labels_from(table, path, labels_filename):
        if labels_filename is not None:
            return table.merge(unpersist(path.joinpath(labels_filename)), left_on="id", right_on="src")
        else:
            return table

    def get_labels_from(self, path, labels_filename):
        labels = {}
        if labels_filename is None:
            return None
        for objective, labels_path in labels_filename.items():
            lpath = path.joinpath(labels_path)
            if lpath.is_file():
                labels[objective] = read_mapping_from_json(lpath)
        if len(labels) == 0:
            return None
        return labels

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
                    return None
                    # raise Exception("No information about partition")

        return self._partition_cache[current_mention]

    def belongs_to_partition(self, path, partition):
        return self._determine_partition(path) == partition

    def _load_graph(
            self, path: Path, node_labels_file=None, edge_labels_file=None, subgraph_labels_file=None,
            force_labels=True, partition=None
    ):
        nodes_path = path.joinpath("nodes.json")
        edges_path = path.joinpath("entry.json")

        if partition is not None and not self.belongs_to_partition(path, partition):
            if self._determine_partition(path) is not None:
                return None  # not a package

        node_labels = None
        edge_labels = None
        subgraph_labels = None
        entry = None

        if (
                nodes_path.is_file() and
                (
                        (node_labels_file is not None and self.labels_present(path, list(node_labels_file.values()), force_labels)) or
                        (edge_labels_file is not None and self.labels_present(path, list(edge_labels_file.values()), force_labels)) or
                        (subgraph_labels_file is not None and self.labels_present(path, list(subgraph_labels_file.values()), force_labels))
                )
        ):
            node_labels = self.get_labels_from(path, node_labels_file)
            edge_labels = self.get_labels_from(path, edge_labels_file)
            subgraph_labels = self.get_labels_from(path, edge_labels_file)

        if node_labels is not None or edge_labels is not None or subgraph_labels is not None:
            entry = read_mapping_from_json(edges_path)
            entry["node_labels"] = node_labels
            entry["edge_labels"] = edge_labels
            entry["subgraph_labels"] = subgraph_labels
            entry["node_names"] = dict(zip(map(lambda x: int(x), entry["node_names"].keys()), entry["node_names"].values()))

        # def extend_if_needed(table, table_):
        #     if table is None:
        #         return table_
        #     elif table_ is None:
        #         return table
        #     else:
        #         return table.append(table_)

        def update_if_needed(item, other):
            if item is None:
                return other
            else:
                if other is not None:
                    item.update(other)
                return item

        for nested_graph in path.iterdir():
            if nested_graph.is_dir():
                subgraph = self._load_graph(
                    nested_graph, node_labels_file, edge_labels_file, subgraph_labels_file, force_labels, partition
                )
                if entry is not None and subgraph is not None:
                    entry["edge_id"] = entry["edge_id"] + subgraph["edge_id"]
                    entry["src_id"] = entry["src_id"] + subgraph["src_id"]
                    entry["dst_id"] = entry["dst_id"] + subgraph["dst_id"]
                    entry["edge_type"] = entry["edge_type"] + subgraph["edge_type"]
                    entry["node_names"] = update_if_needed(entry["node_names"], subgraph["node_names"])
                    entry["node_labels"] = update_if_needed(entry["node_labels"], subgraph["node_labels"])
                    entry["edge_labels"] = update_if_needed(entry["edge_labels"], subgraph["edge_labels"])
                    entry["subgraph_labels"] = update_if_needed(entry["subgraph_labels"], subgraph["subgraph_labels"])
                    # will need to merge offsets here
                    # if subgraph has offsets need to make sure that the entry is an upper level
                    # and not another sibling
                elif entry is None and subgraph is not None:
                    entry = subgraph
                elif entry is not None and subgraph is None:
                    entry = entry
                elif entry is None and subgraph is None:
                    entry = entry

        if entry is not None:
            assert self.labels_in_graph(entry)
            assert len((set(entry["src_id"]) | set(entry["dst_id"])) - set(entry["node_names"].keys())) == 0
            entry.pop('normalized_node_offsets', None)
            entry.pop('normalized_edge_offsets', None)
            entry.pop('normalized_edge_offsets', None)
            entry.pop('type_annotations', None)
            entry.pop('returns', None)
            entry.pop('docstring', None)
            entry.pop('enclosing_span', None)

        return entry

    def labels_in_graph(self, entry):
        node_ids = set(entry["src_id"]) | set(entry["dst_id"])
        all_good = True
        if entry["node_labels"] is not None:
            for obj, map in entry["node_labels"].items():
                if "node_id" in map:
                    all_good = all_good and len(set(map["node_id"]) - node_ids) == 0  # assert here

        edge_ids = set(entry["edge_id"])
        if entry["edge_labels"] is not None:
            for obj, map in entry["edge_labels"].items():
                all_good = all_good and len(set(map["edge_id"]) - edge_ids) == 0  # assert here
        return all_good

    def _stream_packages(self):
        for package_path in self._path.iterdir():
            if package_path.is_dir():
                yield package_path

    def _stream_files(self):
        for package_path in self._stream_packages():
            for file_path in package_path.iterdir():
                if file_path.is_dir():
                    yield file_path

    def _stream_leaf_mentions(self, leaf_file="entry.json"):
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

    def _iterate_leafs(self, path, leaf_file="entry.json"):
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

    def _iterate_all(self, path, need_file="entry.json"):
        if path.joinpath(need_file).is_file():
            yield path

        for subdir in path.iterdir():
            if subdir.is_dir():
                for loc in self._iterate_all(subdir, need_file):
                    yield loc

    def get_objective_summary_path(self, objective_name):
        summary_path = self._path.joinpath(self.summary_dir).joinpath("summary_" + objective_name + ".json")
        return summary_path

    def get_objective_labels_location_path(self, objective_name):
        labels_locations = self._path.joinpath(self.summary_dir).joinpath("location_" + objective_name + ".txt")
        return labels_locations

    def is_level(self, path: Path, level):
        parts = path.relative_to(self._path).parts
        if len(parts) == 1:
            return level == "package"
        elif len(parts) == 2:
            return level == "file"
        elif len(parts) >= 3:
            return level == "mention"
        else:
            return False

    def get_level(self, path, level):
        parts = path.relative_to(self._path).parts
        if level == "package":
            level_ = 1
        elif level == "file":
            level_ = 3
        elif level == "mention":
            level_ = 4
        else:
            raise ValueError

        return self._path.joinpath(*parts[:level_])

    def get_paths_with_level(self, paths, level):
        assert level in {"file", "mention", "package"}

        yield from set(self.get_level(path, level) for path in paths)

    def collect_objective_info(self, objective_names):
        node_labels_file = {}
        edge_labels_file = {}
        subgraph_labels_file = {}
        locations = set()

        if objective_names is not None:
            for objective_name in objective_names:
                locations.update(LocationIterator(self._path, self.get_objective_labels_location_path(objective_name)))
                objective_summary = read_mapping_from_json(self.get_objective_summary_path(objective_name))
                if objective_summary["labels_for"] == "nodes":
                    node_labels_file[objective_name] = objective_summary["labels_filename"]
                elif objective_summary["labels_for"] == "edges":
                    edge_labels_file[objective_name] = objective_summary["labels_filename"]
                elif objective_summary["labels_for"] == "subgraph":
                    subgraph_labels_file[objective_name] = objective_summary["labels_filename"]
                else:
                    raise ValueError

        if len(node_labels_file) == 0:
            node_labels_file = None
        if len(edge_labels_file) == 0:
            edge_labels_file = None
        if len(subgraph_labels_file) == 0:
            subgraph_labels_file = None

        locations = list(locations)
        locations.sort()

        return {
            "node_labels_file": node_labels_file,
            "edge_labels_file": edge_labels_file,
            "subgraph_labels_file": subgraph_labels_file,
            "locations": locations
        }

    def iterate_level(
            self, force_labels=True, partition=None, objective_names=None, default_iterator_fn=None, level=None
    ):
        objective_info = self.collect_objective_info(objective_names)

        if len(objective_info["locations"]) > 0:
            location_iterator = self.get_paths_with_level(objective_info.pop("locations"), level)
        else:
            location_iterator = default_iterator_fn()

        for file in location_iterator:
            graph = self._load_graph(path=file, force_labels=force_labels, partition=partition, **objective_info)
            if graph is not None:
                yield graph

    def iterate_packages(
            self, force_labels=True, partition=None, objective_names=None,
    ):
        # partitions are assigned for files and are irrelevant for packages
        yield from self.iterate_level(
            force_labels, partition, objective_names, default_iterator_fn=self._stream_packages, level="package"
        )

    def iterate_files(
            self, force_labels=True, partition=None, objective_names=None,
    ):
        yield from self.iterate_level(
            force_labels, partition, objective_names, default_iterator_fn=self._stream_files, level="file"
        )

    def iterate_mentions(
            self, force_labels=True, partition=None, objective_names=None
    ):
        yield from self.iterate_level(
            force_labels, partition, objective_names, default_iterator_fn=self._stream_files, level="mention"
        )

    # def iterate_leaf_mentions(
    #         self, node_labels_file=None, edge_labels_file=None, force_labels=True, partition=None, objective_type=None
    # ):
    #     for mention in self._stream_leaf_mentions():
    #         graph = self._load_graph(mention, node_labels_file, edge_labels_file, force_labels, partition)
    #         if graph is not None:
    #             yield graph

    def iterate_all_mentions(
            self, node_labels_file=None, edge_labels_file=None, force_labels=True, partition=None, objective_names=None
    ):
        for file_path in self._stream_files():
            for mention in self._iterate_all(file_path):
                graph = self._load_graph(mention, node_labels_file, edge_labels_file, force_labels, partition)
                if graph is not None:
                    yield graph

    def _stream_fs_files(self, need_file):
        def iterate_all():
            for package_path in self._stream_packages():
                for loc in self._iterate_all(package_path, need_file=need_file):
                    file_path = loc.joinpath(need_file)
                    yield file_path

        yield from iterate_all()

    def _stream_nodes(self):
        for nodes in self._stream_fs_files(need_file="nodes.json"):
            yield read_mapping_from_json(nodes)

    def _stream_edges(self):
        for edges in self._stream_fs_files(need_file="entry.json"):
            yield read_mapping_from_json(edges)

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
    def get_node_type_descriptions(self):
        unique_node_types = set()
        for nodes in tqdm(self._stream_nodes(), desc="Computing unique node types"):
            for node in nodes:
                unique_node_types.add(node["type"])
        unique_node_types = list(unique_node_types)
        unique_node_types.sort()
        return unique_node_types

    @read_or_add_summary_field("edge_types")
    def get_edge_type_descriptions(self):
        unique_edge_types = set()
        for edges in tqdm(self._stream_edges(), desc="Computing unique edge types"):
            unique_edge_types.update(edges["edge_type"])
        unique_edge_types = list(unique_edge_types)
        unique_edge_types.sort()
        return unique_edge_types

    def get_nodes(self, type_filter=None, **kwargs):
        raise NotImplementedError

    def get_edges(self, type_filter=None, **kwargs):
        raise NotImplementedError

    def get_nodes_with_subwords(self):
        raise NotImplementedError

    def get_nodes_for_classification(self):
        raise NotImplementedError

    def get_inbound_neighbors(self, ids):
        raise NotImplementedError

    def get_subgraph_from_node_ids(self, ids):
        raise NotImplementedError

    def get_nodes_for_edges(self, edges):
        raise NotImplementedError

    def get_subgraph_for_package(self, package_id):
        raise NotImplementedError

    def get_all_packages(self):
        for path in self._stream_packages():
            yield path.name

    def get_subgraph_for_file(self, file_id):
        raise NotImplementedError

    def get_all_files(self):
        raise NotImplementedError

    def get_subgraph_for_mention(self, mention):
        raise NotImplementedError

    def get_all_mentions(self):
        raise NotImplementedError

    def iterate_subgraphs(self, how, partition=None, node_labels_file=None, edge_labels_file=None):
        if how == SGPartitionStrategies.package:
            for subgraph in self.iterate_packages(node_labels_file, edge_labels_file, partition):
                yield subgraph
        elif how == SGPartitionStrategies.file:
            for subgraph in self.iterate_files(node_labels_file, edge_labels_file, partition):
                yield subgraph
        elif how == SGPartitionStrategies.mention:
            for subgraph in self.iterate_mentions(node_labels_file, edge_labels_file, partition):
                yield subgraph
        else:
            raise ValueError()

    def get_info_for_subgraphs(self, subgraph_ids, field):
        raise NotImplementedError

    def get_info_for_node_ids(self, node_ids, field):
        raise NotImplementedError


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    s = FileSystemStorage(args.path)

    for g in tqdm(s.iterate_mentions(objective_names=["variable_misuse_node"], partition="val")):
        pass

    # for subgraph in tqdm(s.iterate_all_mentions(partition="train", objective_type="classification"), desc="Iterating mentions"):
    #     pass
    # for p in s.iterate_all_mentions():
    # # for p in s._load_graph(s._path):
    #     print(p)
