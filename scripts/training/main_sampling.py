import json
import logging
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

            date_time = str(datetime.now())
            print("\n\n")
            print(date_time)
            print(f"Model: {model.__name__}, Params: {params}")

            model_attempt = get_name(model, date_time)

            model_base = get_model_base(args, model_attempt)

            dataset = read_or_create_dataset(args=args, model_base=model_base)

            if args.training_mode == "multitask":

                if args.intermediate_supervision:
                    # params['use_self_loop'] = True  # ????
                    from SourceCodeTools.models.graph.train.sampling_multitask_intermediate_supervision import training_procedure
                else:
                    from SourceCodeTools.models.graph.train.sampling_multitask import training_procedure

                trainer, scores = \
                    training_procedure(dataset, model, params, args, model_base)

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
    parser.add_argument('--training_mode', '-tr', dest='training_mode', default=None,
                        help='Selects one of training procedures '
                             '[multitask]')
    parser.add_argument('--data_path', '-d', dest='data_path', default=None,
                        help='Path to the files')
    parser.add_argument('--pretrained', '-p', dest='pretrained', default=None,
                        help='')
    parser.add_argument('--tokenizer', '-t', dest='tokenizer', default=None,
                        help='')
    parser.add_argument('--pretraining_phase', dest='pretraining_phase', default=1, type=int,
                        help='')
    # parser.add_argument('--node_path', '-n', dest='node_path', default=None,
    #                     help='Path to the file with nodes')
    # parser.add_argument('--edge_path', '-e', dest='edge_path', default=None,
    #                     help='Path to the file with edges')
    parser.add_argument('--train_frac', dest='train_frac', default=0.6, type=float,
                        help='')
    # parser.add_argument('--call_seq_file', dest='call_seq_file', default=None,
    #                     help='Path to the file with edges that represent API call sequence. '
    #                          'Used only with training mode \'predict_next_function\'')
    # parser.add_argument('--node_name_file', dest='node_name_file', default=None,
    #                     help='Path to the file with edges that show function names')
    # parser.add_argument('--var_use_file', dest='var_use_file', default=None,
    #                     help='Path to the file with edges that show variable names')
    parser.add_argument('--filter_edges', dest='filter_edges', default=None,
                        help='Edges filtered before training')
    parser.add_argument('--node_emb_size', dest='node_emb_size', default=100, type=int,
                        help='')
    parser.add_argument('--elem_emb_size', dest='elem_emb_size', default=100, type=int,
                        help='')
    parser.add_argument('--num_per_neigh', dest='num_per_neigh', default=10, type=int,
                        help='')
    parser.add_argument('--neg_sampling_factor', dest='neg_sampling_factor', default=3, type=int,
                        help='')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--note', dest='note', default="",
                        help='Note, added to metadata')
    parser.add_argument('model_output_dir',
                        help='Location of the final model')
    parser.add_argument('--use_node_types', action='store_true')
    parser.add_argument('--use_edge_types', action='store_true')
    parser.add_argument('--restore_state', action='store_true')
    parser.add_argument('--self_loops', action='store_true')
    parser.add_argument('--override_labels', action='store_true')
    parser.add_argument('--intermediate_supervision', action='store_true')
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int,
                        help='')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(module)s:%(lineno)d:%(message)s")

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
