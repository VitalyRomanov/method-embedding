import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from SourceCodeTools.models.Embedder import Embedder


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("path")
    # parser.add_argument("output_path")

    args = parser.parse_args()

    path = Path(args.path)
    train_stored = path.joinpath("type_pred_relational_train")
    test_stored = path.joinpath("type_pred_relational_test")

    embedding_keys = {}
    embeddings = []
    # repeated = 0
    without_node = 0
    total_anotations = 0

    repeated = defaultdict(list)

    def add_embeddings_from_dir(dir_path):
        nonlocal repeated, without_node, total_anotations

        for file in dir_path.iterdir():
            if not file.name.endswith(".pt") or file.name.startswith("saved_state"):
                continue

            data = torch.load(file)
            all_offsets = {(r[0], r[1]): r[2] for r in data[1]["replacements"]}
            entities = {(r[0], r[1]) for r in data[1]["entities"]}

            for e in entities:
                total_anotations += 1
                if not (e in all_offsets and all_offsets[e] in data[1]["embeddings"]):
                    without_node += 1
                # assert e in all_offsets and all_offsets[e] in data[1]["embeddings"]

            for key, emb in data[1]["embeddings"].items():
                if key not in embedding_keys:
                    embedding_keys[key] = len(embedding_keys)
                    embeddings.append(emb)
                else:
                    repeated[key].append(emb)
                    # repeated += 1

    add_embeddings_from_dir(train_stored)
    add_embeddings_from_dir(test_stored)

    embedder = Embedder(embedding_keys, np.stack(embeddings))

    output_path = path.joinpath("embedder_relational_transferred.pkl")

    print("Repeated nodes:", len(repeated))
    print("Total annotations:", total_anotations)
    print("Annotations without nodes:", without_node)

    pickle.dump(embedder, open(output_path, "wb"))


if __name__ == "__main__":
    main()