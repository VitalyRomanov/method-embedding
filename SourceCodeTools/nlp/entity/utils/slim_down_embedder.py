import argparse
import pickle
from pathlib import Path

import numpy as np

from SourceCodeTools.models.Embedder import Embedder
from SourceCodeTools.nlp.entity.utils.data import read_json_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("embedder")
    parser.add_argument("output")

    args = parser.parse_args()

    data_path = Path(args.data_path)
    train_data, test_data = read_json_data(
        data_path, normalize=True, allowed=None, include_replacements=True, include_only="entities",
        min_entity_count=0
    )

    embedder = pickle.load(open(args.embedder, "rb"))

    repls = []
    vects = []

    for t in train_data:
        for s, e, r in t[1]["replacements"]:
            try:
                vects.append(embedder[r])
                repls.append(r)
            except:
                continue

    embs = Embedder(dict(zip(repls, range(len(repls)))), np.vstack(vects))
    pickle.dump(embs, open(args.output, "wb"))


if __name__ == "__main__":
    main()
