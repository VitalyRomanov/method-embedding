import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("relational_data_path")
    parser.add_argument("original_embedder")

    args = parser.parse_args()

    data_path = Path(args.relational_data_path)
    original_embedder_path = Path(args.original_embedder)

    output = data_path.joinpath("for_tensorboard")

    functions = data_path.joinpath("type_pred_relational_train")
    original_embedder = pickle.load(open(original_embedder_path, "rb"))

    dists = []

    for i in range(500):
        example_folder = output.joinpath(f"{i}")
        example_folder.mkdir(exist_ok=True, parents=True)

        emb_sink = open(example_folder.joinpath("embs.tsv"), "w")
        emb_orig_sink = open(example_folder.joinpath("embs_original.tsv"), "w")
        emb_meta = open(example_folder.joinpath("meta.tsv"), "w")
        emb_fn = open(example_folder.joinpath("fn.txt"), "w")

        relational_data = torch.load(functions.joinpath(f"{i}.pt"))
        emb_fn.write(f"{relational_data[0]}\n")

        orig = []
        new = []

        nodeid2name = dict(zip(relational_data[1]["nodes"]["id"], relational_data[1]["nodes"]["name"]))
        embedder = relational_data[1]["embeddings"]
        for node, name in nodeid2name.items():
            if node in embedder:
                orig_emb = original_embedder[node]
                orig.append(orig_emb)
                new_emb = embedder[node]
                new.append(new_emb)
                emb_str = '\t'.join(map(str, new_emb))
                orig_emb_str = '\t'.join(map(str, orig_emb))
                emb_orig_sink.write(f"{orig_emb_str}\n")
                emb_sink.write(f"{emb_str}\n")
                emb_meta.write(f"{name}\n")

        orig = np.vstack(orig)
        new = np.vstack(new)

        def normalize(vects):
            length = np.sqrt(np.sum(vects ** 2, axis=-1, keepdims=True))
            return vects / length

        orig = normalize(orig)
        new = normalize(new)

        orig_dist = np.sum(orig[np.newaxis, :, :] * orig[:, np.newaxis, :], axis=-1)
        new_dist = np.sum(new[np.newaxis, :, :] * new[:, np.newaxis, :], axis=-1)

        dist = float(np.sum(np.abs(new_dist - orig_dist)))

        with open(example_folder.joinpath("dist_diff.json"), "w") as dist_diff_sink:
            data = {
                "dist": dist
            }
            dist_diff_sink.write(f"{json.dumps(data)}\n")

        dists.append(dist)

        # print()

        emb_sink.close()
        emb_meta.close()
        emb_fn.close()

    with open(output.joinpath("dist_diff.json"), "w") as dist_diff_sink:
        data = {
            "dist": sum(dists) / len(dists)
        }
        dist_diff_sink.write(f"{json.dumps(data)}\n")

if __name__ == "__main__":
    main()