import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

from tqdm import tqdm


def parse_tensorboard(path):
    scalars = ["F1/Test", "Precision/Test", "Recall/Test"]
    data = defaultdict(list)

    if path == '/Users/LTV/Downloads/NitroShare/events/with_ast/RGGAN-2021-11-01-22-10-11-956335_edgepred_500/type_annotation/2022-04-27_14-58-50.953629/0/events.out.tfevents.1651060736.adilkhan-desktop.2057317.0.v2':
        print()
    for item in summary_iterator(path):
        try:
            tag = item.summary.value[0].tag
            if tag in scalars:
                try:
                    value = tensor_util.MakeNdarray(item.summary.value[0].tensor).item()
                except TypeError:
                    value = item.summary.value[0].simple_value
                data[tag].append(value)#(item.step, value))
        except IndexError:
            pass

    df = pd.DataFrame(data).ewm(0.98).mean()
    maxidx = df["F1/Test"].idxmax()
    max_val = df.loc[maxidx]

    if max_val["F1/Test"] == 0:
        print()
    return max_val["Precision/Test"], max_val["Recall/Test"], max_val["F1/Test"]


relevant_params = [
    "learning_rate",
    "learning_rate_decay",
    "epochs",
    "suffix_prefix_buckets",
    "seq_len",
    "batch_size",
    "no_localization",
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path)

    statistics = defaultdict(lambda: defaultdict(list))

    dirs = []

    for ind, run in enumerate(tqdm(path.iterdir())):
        if not run.is_dir():
            continue
        dirs.append(str(run.absolute()))

    dirs = sorted(dirs)

    for run in dirs:
        run = Path(run)

        params_path = run.joinpath("params.json")

        if not params_path.is_file():
            params_path = run.joinpath("0").joinpath("params.json")

        if not params_path.is_file():
            print("Could not find", params_path)
            continue

        params = json.loads(open(params_path).read())

        key = tuple((pname, params[pname]) for pname in relevant_params)

        events_file = None
        for filein in params_path.parent.iterdir():
            if filein.name.startswith("events.out"):
                events_file = str(filein.absolute())

        if events_file is None:
            print("Could not find event file", params_path)
        max_p, max_r, max_f1 = parse_tensorboard(events_file)

        statistics[key]["test_p"].append(max_p)
        statistics[key]["test_r"].append(max_r)
        statistics[key]["test_f1"].append(max_f1)

    common = None
    for key in statistics:
        cc = set(i for i in key)
        if common is None:
            common = cc
        else:
            common = common.intersection(cc)

    datapoints = []

    for key, v in statistics.items():
        cc = set(i for i in key)
        k = tuple(cc - common)
        datapoint = {"key": k}

        for metric, values in v.items():
            datapoint["num_entries"] = len(values)
            values = np.array(values)
            datapoint[metric] = f"{np.mean(values):.4f}±{np.std(values):.3f}"

        datapoints.append(datapoint)
    df = pd.DataFrame.from_records(datapoints)

    print(args.path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()