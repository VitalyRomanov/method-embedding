import json
import logging
from copy import copy
from datetime import datetime
from os import mkdir
from os.path import isdir, join

from SourceCodeTools.code.data.sourcetrail.Dataset import read_or_create_dataset
from SourceCodeTools.models.graph import RGCNSampling, RGAN, RGGAN
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
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


def add_data_arguments(parser):
    parser.add_argument('--data_path', '-d', dest='data_path', default=None, help='Path to the files')
    parser.add_argument('--train_frac', dest='train_frac', default=0.9, type=float, help='')
    parser.add_argument('--filter_edges', dest='filter_edges', default=None, help='Edges filtered before training')
    parser.add_argument('--min_count_for_objectives', dest='min_count_for_objectives', default=5, type=int, help='')
    parser.add_argument('--packages_file', dest='packages_file', default=None, type=str, help='')
    parser.add_argument('--self_loops', action='store_true')
    parser.add_argument('--use_node_types', action='store_true')
    parser.add_argument('--use_edge_types', action='store_true')
    parser.add_argument('--restore_state', action='store_true')
    parser.add_argument('--no_global_edges', action='store_true')
    parser.add_argument('--remove_reverse', action='store_true')


def add_pretraining_arguments(parser):
    parser.add_argument('--pretrained', '-p', dest='pretrained', default=None, help='')
    parser.add_argument('--tokenizer', '-t', dest='tokenizer', default=None, help='')
    parser.add_argument('--pretraining_phase', dest='pretraining_phase', default=0, type=int, help='')


def add_training_arguments(parser):
    parser.add_argument('--embedding_table_size', dest='embedding_table_size', default=200000, type=int, help='Batch size')
    parser.add_argument('--random_seed', dest='random_seed', default=None, type=int, help='')

    parser.add_argument('--node_emb_size', dest='node_emb_size', default=100, type=int, help='')
    parser.add_argument('--elem_emb_size', dest='elem_emb_size', default=100, type=int, help='')
    parser.add_argument('--num_per_neigh', dest='num_per_neigh', default=10, type=int, help='')
    parser.add_argument('--neg_sampling_factor', dest='neg_sampling_factor', default=3, type=int, help='')

    parser.add_argument('--use_layer_scheduling', action='store_true')
    parser.add_argument('--schedule_layers_every', dest='schedule_layers_every', default=10, type=int, help='')

    parser.add_argument('--epochs', dest='epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size')

    parser.add_argument("--h_dim", dest="h_dim", default=None, type=int)
    parser.add_argument("--n_layers", dest="n_layers", default=5, type=int)
    parser.add_argument("--objectives", dest="objectives", default=None, type=str)


def add_scoring_arguments(parser):
    parser.add_argument('--measure_ndcg', action='store_true')
    parser.add_argument('--dilate_ndcg', dest='dilate_ndcg', default=200, type=int, help='')


def add_performance_arguments(parser):
    parser.add_argument('--no_checkpoints', dest="save_checkpoints", action='store_false')

    parser.add_argument('--use_gcn_checkpoint', action='store_true')
    parser.add_argument('--use_att_checkpoint', action='store_true')
    parser.add_argument('--use_gru_checkpoint', action='store_true')


def add_train_args(parser):
    parser.add_argument(
        '--training_mode', '-tr', dest='training_mode', default=None,
        help='Selects one of training procedures [multitask]'
    )

    add_data_arguments(parser)
    add_pretraining_arguments(parser)
    add_training_arguments(parser)
    add_scoring_arguments(parser)
    add_performance_arguments(parser)

    parser.add_argument('--note', dest='note', default="", help='Note, added to metadata')
    parser.add_argument('model_output_dir', help='Location of the final model')

    # parser.add_argument('--intermediate_supervision', action='store_true')
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int, help='')


def verify_arguments(args):
    pass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    add_train_args(parser)

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
