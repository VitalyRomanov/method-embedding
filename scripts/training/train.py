import json
import logging
from copy import copy
from datetime import datetime
from os import mkdir
from os.path import isdir, join

from SourceCodeTools.code.data.sourcetrail.Dataset import read_or_create_dataset
from SourceCodeTools.models.graph import RGCNSampling, RGAN, RGGAN
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from SourceCodeTools.models.training_options import add_gnn_train_args, verify_arguments
from params import rgcnsampling_params, rggan_params


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

            dataset = read_or_create_dataset(args=args, model_base=model_base)

            def write_params(args, params):
                args = copy(args.__dict__)
                args.update(params)
                args['activation'] = args['activation'].__name__
                with open(join(model_base, "params.json"), "w") as mdata:
                    mdata.write(json.dumps(args, indent=4))

            write_params(args, params)

            if args.training_mode == "multitask":

                # if args.intermediate_supervision:
                #     # params['use_self_loop'] = True  # ????
                #     from SourceCodeTools.models.graph.train.sampling_multitask_intermediate_supervision import training_procedure
                # else:
                from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure

                trainer, scores = \
                    training_procedure(dataset, model, copy(params), args, model_base)

                trainer.save_checkpoint(model_base)
            else:
                raise ValueError("Unknown training mode:", args.training_mode)

            print("Saving...", end="")

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

            print("done")





if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    add_gnn_train_args(parser)

    args = parser.parse_args()
    verify_arguments(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    models_ = {
        # GCNSampling: gcnsampling_params,
        # GATSampler: gatsampling_params,
        # RGCNSampling: rgcnsampling_params,
        # RGAN: rgcnsampling_params,
        RGGAN: rggan_params

    }

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    main(models_, args)
