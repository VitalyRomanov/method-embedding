import json
import logging
from copy import copy
from datetime import datetime
from os import mkdir
from os.path import isdir, join

from SourceCodeTools.cli_arguments import GraphTrainerArgumentParser
from SourceCodeTools.code.data.dataset.Dataset import read_or_create_gnn_dataset
from SourceCodeTools.models.graph import RGGAN
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from params import rggan_params


def main(models, args):
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

            if args.external_dataset is not None:
                external_args = copy(args)
                external_args.data_path = external_args.external_dataset
                external_args.external_model_base = get_model_base(external_args, model_attempt, force_new=True)
                def load_external_dataset():
                    return external_args, read_or_create_gnn_dataset(args=external_args,
                                                                     model_base=external_args.external_model_base,
                                                                     force_new=True)
            else:
                load_external_dataset = None

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
                training_procedure(dataset, model, copy(params), args, model_base, load_external_dataset=load_external_dataset)

            if load_external_dataset is not None:
                model_base = external_args.external_model_base

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





if __name__ == "__main__":

    args = GraphTrainerArgumentParser().parse()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    models_ = {
        # GCNSampling: gcnsampling_params,
        # GATSampler: gatsampling_params,
        # RGCN: rgcnsampling_params,
        # RGAN: rgcnsampling_params,
        RGGAN: rggan_params

    }

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    main(models_, args)
