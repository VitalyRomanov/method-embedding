import bz2
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def write_chunk(temp, column_order, first_part):
        data = pd.DataFrame.from_records(temp, columns=column_order)
        data.rename({"function": "filecontent"}, axis=1, inplace=True)
        if first_part is True:
            data.to_csv(output_path, index=False)
        else:
            data.to_csv(output_path, index=False, mode="a", header=False)

def convert(dataset_path, output_path):
    assert dataset_path.name.endswith("bz2")

    file = bz2.open(dataset_path, mode="rt")

    temp = []
    id_ = 0

    column_order = None
    first_part = True

    for line in tqdm(file):
        record = json.loads(line)

        if record.pop("parsing_error"):
            continue

        if column_order is None:
            column_order = list(record.keys())
            column_order.insert(0, "id")

        record["id"] = id_
        id_ += 1

        temp.append(record)

        if len(temp) > 10000:
            write_chunk(temp, column_order, first_part)
            first_part = False
            temp.clear()

    if len(temp) > 0:
        write_chunk(temp, column_order, first_part)

    # dataset = pd.DataFrame.from_records(json.loads(line) for line in)
    # dataset.rename({"function": "filecontent"}, axis=1, inplace=True)
    # dataset["id"] = range(len(dataset))
    #
    # dataset.to_pickle(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Convert CuBERT dataset file from jsonl format to pandas table comparible with AST graph builder")

    parser.add_argument("dataset_path", help="Path to jsonl file (compressed bz2)")
    parser.add_argument("output_path")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    convert(dataset_path, output_path)