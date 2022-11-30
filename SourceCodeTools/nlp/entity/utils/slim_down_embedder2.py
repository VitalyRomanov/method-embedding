import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from SourceCodeTools.code.common import read_nodes
from SourceCodeTools.code.data.cubert_python_benchmarks.data_iterators import DataIterator
from SourceCodeTools.models.Embedder import Embedder
from SourceCodeTools.nlp.entity.utils.data import read_json_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("embedder")
    parser.add_argument("nodes")
    parser.add_argument("output")

    args = parser.parse_args()
    assert args.embedder.endswith("txt")

    def get_node_types():
        node_types = dict()
        for nodes in tqdm(read_nodes(args.nodes, as_chunks=True), desc="Reading node types"):
            node_types.update(dict(zip(nodes["id"], nodes["type"])))
        return node_types

    node_types = get_node_types()

    data_path = Path(args.data_path)
    train_data = DataIterator(data_path, "train")
    test_data = DataIterator(data_path, "test")

    def get_allowed_ids(data_iterator):
        allowed_ids = set()
        for text, entry in tqdm(data_iterator, desc="Reading data"):
            for _, _, node_id in entry["replacements"]:
                if node_types[node_id] in {"mention", "args", "arg"}:
                    allowed_ids.add(node_id)
        return allowed_ids

    allowed_ids = get_allowed_ids(train_data)
    allowed_ids.update(get_allowed_ids(test_data))

    with open(args.embedder, "r") as source:
        added = set()
        keys = dict()
        embs = list()

        for ind, line in enumerate(tqdm(source, desc="Reading embeddings")):
            parts = line.strip().split()
            key = int(parts.pop(0))
            emb = np.fromiter(map(str, parts), dtype=np.float32)
            if key in allowed_ids and key not in added:
                keys[key] = len(keys)
                embs.append(emb)

            if ind % 100000 == 0:
                print("Added", len(keys))

        print("After slimming:", len(keys))

    embs = Embedder(keys, np.vstack(embs))
    pickle.dump(embs, open(args.output, "wb"))


if __name__ == "__main__":
    main()
