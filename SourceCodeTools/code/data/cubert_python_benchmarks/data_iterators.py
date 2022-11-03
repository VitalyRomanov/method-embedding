import json
from os.path import join


def read_data(dataset_path, partition):
    """
    Read data stored as JSON records.
    """
    assert partition in {"train", "val", "test"}
    data_path = join(dataset_path, f"var_misuse_seq_{partition}.json")

    # data = []
    for line in open(data_path, "r"):
        entry = json.loads(line)

        text = entry.pop("text")
        yield (text, entry)
        # data.append((text, entry))
    # return data


def iterate_data(dataset):
    for entry in dataset:
        text = entry.pop("text")
        yield (text, entry)


def get_num_lines(dataset_path, partition):
    assert partition in {"train", "val", "test"}
    data_path = join(dataset_path, f"var_misuse_seq_{partition}.json")

    return sum(1 for line in open(data_path))


class DataIterator:
    def __init__(self, data_path, partition_name):
        assert partition_name in {"train", "val", "test"}

        self._data_path = data_path
        self._partition_name = partition_name

        self._num_entries = get_num_lines(self._data_path, self._partition_name)

    def __iter__(self):
        return read_data(self._data_path, self._partition_name)

    def __len__(self):
        return self._num_entries


class DataIteratorMem:
    def __init__(self, dataset):
        self._data = dataset

    def __iter__(self):
        return iterate_data(self._data)

    def __len__(self):
        return len(self._data)