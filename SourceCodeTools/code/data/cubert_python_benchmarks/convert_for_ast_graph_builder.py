import bz2
# import hashlib
import json
from pathlib import Path
from random import random, seed

import pandas as pd
from tqdm import tqdm


def write_chunk(temp, column_order, first_part, output_path):
        data = pd.DataFrame.from_records(temp, columns=column_order)
        data.rename({"function": "filecontent"}, axis=1, inplace=True)
        if first_part is True:
            data.to_csv(output_path, index=False)
        else:
            data.to_csv(output_path, index=False, mode="a", header=False)


def convert_bzip(dataset_path, output_path, keep_frac):
    assert dataset_path.name.endswith("bz2")

    dataset = bz2.open(dataset_path, mode="rt")

    convert(dataset, output_path, keep_frac)


def convert_json(dataset_path, output_path, keep_frac):
    assert dataset_path.name.endswith("json")

    dataset = open(dataset_path, mode="r")

    convert(dataset, output_path, keep_frac)


# def compute_id(function_text):
#     return int(hashlib.md5(function_text.encode('utf8')).hexdigest(), 16) % 4294967296  # 2**32


def convert(dataset, output_path, keep_frac):

    temp = []
    id_ = 0

    column_order = None
    first_part = True

    original = False
    write_to_output = False
    seed(42)

    for ind, line in enumerate(tqdm(dataset)):
        record = json.loads(line)

        if record.pop("parsing_error"):
            continue

        if column_order is None:
            column_order = list(record.keys())
            column_order.insert(0, "id")

        if original is False and random() < keep_frac:
            write_to_output = True

        if write_to_output:
            record["id"] = id_  # compute_id(record["fn_path"] + record["function"])
            id_ += 1

            temp.append(record)

        if original is True:
            assert record["label"] == "Correct"
            original = False
            write_to_output = False
        else:
            assert record["label"] == "Variable misuse"
            original = True

        if len(temp) > 10000:
            write_chunk(temp, column_order, first_part, output_path)
            first_part = False
            temp.clear()

    if len(temp) > 0:
        write_chunk(temp, column_order, first_part, output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Convert CuBERT dataset file from json format to pandas table comparible with AST graph builder")

    parser.add_argument("dataset_path", help="Path to json file (compressed bz2)")
    parser.add_argument("output_path")
    parser.add_argument("--keep_frac", default=1.0, type=float, help="")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    if dataset_path.name.endswith("bz2"):
        convert_bzip(dataset_path, output_path, keep_frac=args.keep_frac)
    else:
        convert_json(dataset_path, output_path, keep_frac=args.keep_frac)