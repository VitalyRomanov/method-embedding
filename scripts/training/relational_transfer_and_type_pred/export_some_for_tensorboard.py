import argparse
import pickle
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("relational_data_path")

    args = parser.parse_args()

    data_path = Path(args.relational_data_path)

    output = data_path.joinpath("for_tensorboard")

    functions = data_path.joinpath("type_pred_relational_train")

    for i in range(10):
        example_folder = output.joinpath(f"{i}")
        example_folder.mkdir(exist_ok=True, parents=True)

        emb_sink = open(example_folder.joinpath("embs.tsv"), "w")
        emb_meta = open(example_folder.joinpath("meta.tsv"), "w")
        emb_fn = open(example_folder.joinpath("fn.txt"), "w")

        relational_data = torch.load(functions.joinpath(f"{i}.pt"))
        emb_fn.write(f"{relational_data[0]}\n")

        nodeid2name = dict(zip(relational_data[1]["nodes"]["id"], relational_data[1]["nodes"]["name"]))
        embedder = relational_data[1]["embeddings"]
        for node, name in nodeid2name.items():
            if node in embedder:
                emb_str = '\t'.join(map(str, embedder[node]))
                emb_sink.write(f"{emb_str}\n")
                emb_meta.write(f"{name}\n")

        emb_sink.close()
        emb_meta.close()
        emb_fn.close()

if __name__ == "__main__":
    main()