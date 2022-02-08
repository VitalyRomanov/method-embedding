import json
import logging
import os
import shutil
from datetime import datetime
from os import mkdir
from os.path import isdir

from SourceCodeTools.code.data.dataset.Dataset import read_or_create_gnn_dataset
from SourceCodeTools.models.graph import RGGAN
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from SourceCodeTools.models.training_options import add_gnn_train_args


def detect_checkpoint_files(path):
    checkpoints = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if not os.path.isfile:
            continue

        if not file.startswith("saved_state_"):
            continue

        epoch = int(file.split("_")[2].split(".")[0])
        checkpoints.append((epoch, filepath))

    checckpoints = sorted(checkpoints, key=lambda x: x[0])

    return checckpoints


def main(models, args):
    for model, param_grid in models.items():
        for params in param_grid:

            if args.h_dim is None:
                params["h_dim"] = args.node_emb_size
            else:
                params["h_dim"] = args.h_dim

            date_time = str(datetime.now())
            print("\n\n")
            print(date_time)
            print(f"Model: {model.__name__}, Params: {params}")

            model_attempt = get_name(model, date_time)

            model_base = get_model_base(args, model_attempt)

            dataset = read_or_create_gnn_dataset(args=args, model_base=model_base)

            from SourceCodeTools.models.graph.train.sampling_multitask2 import evaluation_procedure

            checkpoints = detect_checkpoint_files(model_base)

            for epoch, ckpt_path in checkpoints:
                shutil.copy(ckpt_path, os.path.join(model_base, "saved_state.pt"))

                evaluation_procedure(dataset, model, params, args, model_base)



if __name__ == "__main__":

    import argparse
    # from train import add_train_args

    parser = argparse.ArgumentParser(description='Process some integers.')
    add_gnn_train_args(parser)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    model_out = args.model_output_dir

    try:
        saved_args = json.loads(open(os.path.join(model_out, "metadata.json")).read())
        models_ = {
            eval(saved_args.pop("name").split("-")[0]): [saved_args.pop("parameters")]
        }
    except:
        param_keys = 'activation', 'use_self_loop', 'num_steps', 'dropout', 'num_bases', 'lr'
        saved_args = json.loads(open(os.path.join(model_out, "params.json")).read())
        models_ = {
            RGGAN: [{key: saved_args[key] for key in param_keys}]
        }

    args.__dict__.update(saved_args)
    args.restore_state = True
    args.model_output_dir = model_out

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    main(models_, args)
