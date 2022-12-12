import sys

from nhkv import KVStore
import numpy as np
from tqdm import tqdm


def main():
    embeddings_file_path = sys.argv[1]
    output_path = sys.argv[2]

    storage = KVStore(output_path)

    with open(embeddings_file_path, "r") as source:
        for line in tqdm(source, desc="Reading embeddings"):
            parts = line.split()
            if len(parts) > 0:
                key = int(parts.pop(0))

            embedding = np.fromiter(map(float, parts), dtype=np.float32)
            storage[key] = embedding
        storage.save()


if __name__ == "__main__":
    main()