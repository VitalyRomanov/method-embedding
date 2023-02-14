import logging
from copy import copy
from datetime import datetime
from os import mkdir
from os.path import isdir
from pathlib import Path

import torch

from SourceCodeTools.code.data.dataset.Dataset import ProxyDataset
from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure
from SourceCodeTools.models.graph.train.sampling_relational_finetune import RelationalFinetuneTrainer
from SourceCodeTools.models.training_config import get_config, load_config
from SourceCodeTools.models.training_options import add_gnn_train_args, verify_arguments


def iterate_data(data_path):
    data = torch.load(data_path)
    for r in data:
        yield r


def train_model(args):

    args = copy(args.__dict__)
    config_path = args.pop("config")

    if config_path is None:
        config = get_config(**args)
    else:
        config = load_config(config_path)

    date_time = str(datetime.now())
    print("\n\n")
    print(date_time)

    model_base = Path(config["DATASET"]["data_path"])
    config["TRAINING"]["restore_state"] = True
    config["TRAINING"]["save_checkpoints"] = False
    config["MODEL"]["n_layers"] = 1
    config["DATASET"]["remove_reverse"] = True

    def process_data(data_path, out_datapath):
        for ind, (text, entry) in enumerate(iterate_data(data_path)):
            save_path = out_datapath.joinpath(f"{ind}.pt")
            if save_path.is_file():
                continue

            dataset = ProxyDataset(
                **config["DATASET"],
                **config["TOKENIZER"],
                storage_kwargs={
                    "nodes": entry["nodes"],
                    "edges": entry["edges"],
                },
                type_nodes=True
            )

            if dataset._graph_storage.get_num_nodes() == 0:
                continue

            ntypes, etypes = dataset.get_graph_types()
            config["TRAINING"]['ntypes'] = ntypes
            config["TRAINING"]['etypes'] = etypes

            trainer, scores, embedder = training_procedure(
                dataset,
                model_name=None,
                model_params=config["MODEL"],
                trainer_params=config["TRAINING"],
                tokenizer_path=config["TOKENIZER"]["tokenizer_path"],
                model_base_path=model_base,
                trainer=RelationalFinetuneTrainer
            )

            # trainer.save_checkpoint(out_datapath, f"{ind}_checkpoint.pt")
            entry["embeddings"] = embedder
            torch.save(
                (text, entry), save_path
            )

    train_path_out = model_base.joinpath("type_pred_relational_train")
    train_path_out.mkdir(exist_ok=True)
    process_data(model_base.joinpath("train_relaitonal_type_pred.pt"), train_path_out)

    test_path_out = model_base.joinpath("type_pred_relational_train")
    test_path_out.mkdir(exist_ok=True)
    process_data(model_base.joinpath("test_relaitonal_type_pred.pt"), test_path_out)

    # print("Saving...")
    #
    # metadata = {
    #     "base": model_base,
    #     "name": model_attempt,
    #     "layers": "embeddings.pkl",
    #     "mappings": "nodes.csv",
    #     "state": "state_dict.pt",
    #     "scores": scores,
    #     "time": date_time,
    # }
    #
    # metadata["config"] = args
    #
    # # pickle.dump(embedder, open(join(model_base, metadata['layers']), "wb"))
    #
    # with open(join(model_base, "metadata.json"), "w") as mdata:
    #     mdata.write(json.dumps(metadata, indent=4))
    #
    # print("Done saving")


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
    #     # RGCN: rgcnsampling_params,
    #     # RGAN: rgcnsampling_params,
    #     RGGAN: rggan_params
    #
    # }

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    train_model(args)
