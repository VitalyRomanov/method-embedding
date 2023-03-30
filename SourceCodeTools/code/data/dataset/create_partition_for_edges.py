import argparse
import json
from os.path import join

import pandas as pd

from SourceCodeTools.code.common import read_edges
from SourceCodeTools.code.data.dataset.create_partition import add_splits
from SourceCodeTools.code.data.file_utils import persist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("working_directory")
    parser.add_argument("output_path")

    args = parser.parse_args()

    exclude = set()
    all_edges = []
    for edges in read_edges(join(args.working_directory, "common_edges.json.bz2"), as_chunks=True):
        all_edges.append(edges[["id"]])

    partition = add_splits(
        items=pd.concat(all_edges),
        train_frac=0.8,
        force_test=None,
        exclude_ids=exclude
    )

    persist(partition, args.output_path)


if __name__ == "__main__":
    main()