import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util
import matplotlib.pyplot as plt

from tqdm import tqdm

from SourceCodeTools.models.training_config import load_config


def parse_tensorboard(path):
    scalars = ["Accuracy/test/TypeAnnPrediction_"]
    data = defaultdict(list)

    all_tags = set()
    for item in summary_iterator(path):
        try:
            tag = item.summary.value[0].tag
            all_tags.add(tag)
            if tag in scalars:
                try:
                    value = tensor_util.MakeNdarray(item.summary.value[0].tensor).item()
                except TypeError:
                    value = item.summary.value[0].simple_value
                data[tag].append(value)#(item.step, value))
        except IndexError:
            pass

    df = pd.DataFrame(data).ewm(0.98).mean()
    maxidx = df["Accuracy/test/TypeAnnPrediction_"].idxmax()
    max_val = df.loc[maxidx]

    if max_val["Accuracy/test/TypeAnnPrediction_"] == 0:
        print("wtf")

    acc_100 = df.iloc[99]["Accuracy/test/TypeAnnPrediction_"] if len(df) > 99 else None
    acc_200 = df.iloc[99]["Accuracy/test/TypeAnnPrediction_"] if len(df) > 199 else None
    acc_last = df.iloc[-1]["Accuracy/test/TypeAnnPrediction_"]


    return df,  max_val["Accuracy/test/TypeAnnPrediction_"], acc_100, acc_200, acc_last


relevant_params = [
    "filter_edges",
    "no_global_edges",
    "use_edge_types",
    "activation",
    "h_dim",
    "n_layers",
    "learning_rate",
    "restore_state",
    "pretraining_objective"
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
        for dd in run.iterdir():
            if dd.name.startswith("."):
                continue
            if not dd.is_dir():
                continue
            dirs.append(str(dd.absolute()))

    dirs = sorted(dirs)

    for ind, run in enumerate(dirs):

        run = Path(run)

        params_path = run.joinpath("config.yaml")
        if not params_path.is_file():
            continue
        params = load_config(params_path)
        params = {**params['DATASET'], **params['MODEL'], **params['TRAINING']}

        if params["filter_edges"] is not None:
            params["filter_edges"] = tuple(params["filter_edges"])

        pobjective = None
        if "name_pred" in str(run):
            pobjective = "name_pred"
        if "edge_pred" in str(run):
            pobjective = "edge_pred"
        params["pretraining_objective"] = pobjective
        key = tuple((pname, params[pname]) for pname in relevant_params)


        events_file = None
        for filein in params_path.parent.iterdir():
            if filein.name.startswith("events.out"):
                events_file = str(filein.absolute())
                break

        if events_file is None:
            print("Could not find event file", params_path)
        history, max_acc, acc_100, acc_200, final_acc = parse_tensorboard(events_file)

        statistics[key]["history"].append(history)
        statistics[key]["max_acc"].append(max_acc)
        statistics[key]["final_acc"].append(final_acc)
        statistics[key]["acc_100"].append(acc_100)
        statistics[key]["acc_200"].append(acc_200)

    common = None
    for key in statistics:
        cc = set(i for i in key)
        if common is None:
            common = cc
        else:
            common = common.intersection(cc)

    datapoints = []
    legend = []
    history_bank = []

    for key, v in statistics.items():
        cc = set(i for i in key)
        k = tuple(cc - common)
        # datapoint = {"key": k}
        datapoint = {}
        for p, pv in k:
            datapoint[p] = pv

        datapoint["history_id"] = len(datapoints)
        history = v.pop("history")[0].copy()
        history["id"] = datapoint["history_id"]

        history_bank.append(history)

        legend.append(str(datapoint))
        # plt.plot(v["history"][0].values)

        for metric, values in v.items():
            # if metric == "history" or metric == "history": continue
            # datapoint["num_entries"] = len(values)
            values = np.array(values)
            datapoint[metric] = f"{np.mean(values):.4f}" if values[0] is not None else None# f"{np.mean(values):.4f}±{np.std(values):.3f}"

        datapoints.append(datapoint)
    df = pd.DataFrame.from_records(datapoints)
    df.to_csv(Path(args.path).joinpath("type_pred_accuracy.csv"), index=False)

    # plt.legend(legend)
    # plt.savefig(Path(args.path).joinpath("history.svg"))
    history_bank = pd.concat(history_bank)
    history_bank.to_csv(Path(args.path).joinpath("history_bank.csv"), index=False)

    data = df.astype({"max_acc": "float32", "final_acc": "float32", "acc_100": "float32", "acc_200": "float32"})
    data = data.query("n_layers <= 5")

    def dependence_on_number_of_layers(df):
        # dependence on number of layers
        for key, grp in df.query("restore_state == False").groupby(['use_edge_types', 'filter_edges', 'restore_state',
           'no_global_edges']):
            grp.sort_values("n_layers", inplace=True)
            plt.plot(grp["n_layers"], grp["final_acc"])
        plt.show()

    def improvement_from_pretraining(df):
        legend = []
        plt.figure(figsize=(6,5))
        for pobj in set(df["pretraining_objective"].unique()) - {None}:
            dd = df[df["pretraining_objective"].apply(lambda x: x != pobj)]
            if pobj == "edge_pred":
                legend.append("Edge Prediction")
            if pobj == "name_pred":
                legend.append("Name Prediction")
            diffs = []
            for key, grp in dd.query("use_edge_types == False").groupby(['use_edge_types', 'filter_edges', 'n_layers',
               'no_global_edges']):
                assert len(grp) == 2
                # print(grp["restore_state"])
                grp.sort_values("restore_state", inplace=True)
                hid_1 = grp["history_id"].iloc[0]
                hid_2 = grp["history_id"].iloc[1]
                h_1 = history_bank.query(f"id == {hid_1}")["Accuracy/test/TypeAnnPrediction_"]
                h_2 = history_bank.query(f"id == {hid_2}")["Accuracy/test/TypeAnnPrediction_"]
                assert len(h_1) == len(h_2)
                diff = (h_2 - h_1).values.reshape(1, -1)
                diffs.append(diff)
            diffs = np.concatenate(diffs, axis=0)
            plt.plot(np.mean(diffs, axis=0))
        plt.legend(legend)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy Difference")
        plt.grid()
        # plt.savefig(Path(args.path).joinpath("pretraining_effect.svg"))
        plt.show()


    def ablation(df):
        inspect_params = {
            'use_edge_types': [False, True],
            'filter_edges': [None, ("subword",)],
            'no_global_edges': [False, True]
        }
        dd = df.query("restore_state == False")
        for key in inspect_params:
            options = inspect_params[key]
            t1 = dd[dd[key].apply(lambda x: x == options[1])]
            t2 = dd[dd[key].apply(lambda x: x != options[1])]

            merged = t1.merge(t2, on=list(set(inspect_params.keys()) - set([key]) | set(["n_layers"])))[["final_acc_x", "final_acc_y"]]
            merged = merged.dropna()
            diff = merged["final_acc_x"] - merged["final_acc_y"]
            print(key, len(merged), diff.mean(), diff.std(), t1["final_acc"].mean(), t1["final_acc"].std(), t2["final_acc"].mean(), t2["final_acc"].std())

    dependence_on_number_of_layers(data)
    improvement_from_pretraining(data)
    ablation(data)

    print()


if __name__ == "__main__":
    main()