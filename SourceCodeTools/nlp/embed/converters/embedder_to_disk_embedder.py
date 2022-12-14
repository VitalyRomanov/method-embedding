import pickle
import sys

import numpy as np
from nhkv import KVStore
from tqdm import tqdm


def main():
    embeddings_path = sys.argv[1]
    output_path = sys.argv[2]

    embedder = pickle.load(open(embeddings_path, "rb"))[0]

    store = KVStore(output_path)
    for key in tqdm(embedder.keys()):
        store[key] = np.ravel(embedder[key])
    store.save()


if __name__ == "__main__":
    main()