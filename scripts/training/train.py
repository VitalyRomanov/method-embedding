import json
import logging
from copy import copy
from datetime import datetime
from os import mkdir
from os.path import isdir, join

from SourceCodeTools.code.data.dataset.Dataset import read_or_create_gnn_dataset
from SourceCodeTools.models.graph import RGGAN
from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from SourceCodeTools.models.training_config import get_config, load_config, update_config, save_config
from SourceCodeTools.models.training_options import add_gnn_train_args, verify_arguments
from params import rggan_params


def train_grid(models, args):

    for model, param_grid in models.items():
        for params in param_grid:

            if args.h_dim is None:
                params["h_dim"] = args.node_emb_size
            else:
                params["h_dim"] = args.h_dim

            params["num_steps"] = args.n_layers

            date_time = str(datetime.now())
            print("\n\n")
            print(date_time)
            print(f"Model: {model.__name__}, Params: {params}")

            model_attempt = get_name(model, date_time)

            model_base = get_model_base(args, model_attempt)

            dataset = read_or_create_gnn_dataset(args=args, model_base=model_base)

            def write_params(args, params):
                args = copy(args.__dict__)
                args.update(params)
                args['activation'] = args['activation'].__name__
                with open(join(model_base, "params.json"), "w") as mdata:
                    mdata.write(json.dumps(args, indent=4))

            if not args.restore_state:
                write_params(args, params)

            from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure

            trainer, scores = \
                training_procedure(dataset, model, copy(params), args, model_base)

            trainer.save_checkpoint(model_base)

            print("Saving...")

            params['activation'] = params['activation'].__name__

            metadata = {
                "base": model_base,
                "name": model_attempt,
                "parameters": params,
                "layers": "embeddings.pkl",
                "mappings": "nodes.csv",
                "state": "state_dict.pt",
                "scores": scores,
                "time": date_time,
            }

            metadata.update(args.__dict__)

            # pickle.dump(dataset, open(join(model_base, "dataset.pkl"), "wb"))
            import pickle
            pickle.dump(trainer.get_embeddings(), open(join(model_base, metadata['layers']), "wb"))

            with open(join(model_base, "metadata.json"), "w") as mdata:
                mdata.write(json.dumps(metadata, indent=4))

            print("Done saving")


def main(args):

    args = copy(args.__dict__)
    config_path = args.pop("config")

    if config_path is None:
        config = get_config(**args)
    else:
        config = load_config(config_path)

    model = RGGAN

    date_time = str(datetime.now())
    print("\n\n")
    print(date_time)

    restore_state = config["TRAINING"]["restore_state"]

    model_attempt = get_name(model, date_time)

    model_base = get_model_base(config["TRAINING"], model_attempt)

    dataset = read_or_create_gnn_dataset(
        args={**config["DATASET"], **config["TOKENIZER"]},
        model_base=model_base, restore_state=restore_state
    )

    if not restore_state:
        save_config(config, join(model_base, "config.yaml"))

    trainer, scores = training_procedure(
        dataset,
        model_name=model,
        model_params=config["MODEL"],
        trainer_params=config["TRAINING"],
        tokenizer_path=config["TOKENIZER"]["tokenizer_path"],
        model_base_path=model_base
    )

    trainer.save_checkpoint(model_base)

    print("Saving...")

    metadata = {
        "base": model_base,
        "name": model_attempt,
        "layers": "embeddings.pkl",
        "mappings": "nodes.csv",
        "state": "state_dict.pt",
        "scores": scores,
        "time": date_time,
    }

    metadata["config"] = args

    # pickle.dump(dataset, open(join(model_base, "dataset.pkl"), "wb"))
    import pickle
    pickle.dump(trainer.get_embeddings(), open(join(model_base, metadata['layers']), "wb"))

    with open(join(model_base, "metadata.json"), "w") as mdata:
        mdata.write(json.dumps(metadata, indent=4))

    print("Done saving")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='')
    add_gnn_train_args(parser)

    args = parser.parse_args()
    verify_arguments(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    # models_ = {
    #     # GCNSampling: gcnsampling_params,
    #     # GATSampler: gatsampling_params,
    #     # RGCNSampling: rgcnsampling_params,
    #     # RGAN: rgcnsampling_params,
    #     RGGAN: rggan_params
    #
    # }

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    main(args)

    # main(models_, args)
