from SourceCodeTools.graph.model import GCNSampling, GATSampler, RGCNSampling
from SourceCodeTools.graph.model.train.utils import get_name, get_model_base
import sys
from datetime import datetime
from params import gcnsampling_params, gatsampling_params, rgcnsampling_params
import pandas
import pickle
import json
from os import mkdir
from os.path import isdir, join
import torch
from SourceCodeTools.data.sourcetrail.Dataset import SourceGraphDataset, read_or_create_dataset


def main(models, args):

    for model, param_grid in models.items():
        for params in param_grid:



            dateTime = str(datetime.now())
            print("\n\n")
            print(dateTime)
            print(f"Model: {model.__name__}, Params: {params}")

            model_attempt = get_name(model, dateTime)

            MODEL_BASE = get_model_base(args, model_attempt)

            dataset = read_or_create_dataset(args=args, model_base=MODEL_BASE, model_name=model.__name__)



            if args.training_mode == 'node_classifier':

                from SourceCodeTools.graph.model.train.sampling_node_classifier import training_procedure

                m, scores = training_procedure(dataset, model, params, args.epochs, args)

            elif args.training_mode == "multitask":

                from SourceCodeTools.graph.model.train.sampling_multitask import training_procedure

                m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, scores = \
                    training_procedure(dataset, model, params, args.epochs, args)

                torch.save(
                    {
                        'elem_embeder_fname': ee_fname.state_dict(),
                        'elem_embeder_varuse': ee_varuse.state_dict(),
                        'elem_embeder_apicall': ee_apicall.state_dict(),
                        'link_predictor_fname': lp_fname.state_dict(),
                        'link_predictor_varuse': lp_varuse.state_dict(),
                        'link_predictor_apicall': lp_apicall.state_dict(),
                    },
                    join(MODEL_BASE, "multitask.pt")
                )
            else:
                raise ValueError("Unknown training mode:", args.training_mode)

            print("Saving...", end="")

            params['activation'] = params['activation'].__name__

            metadata = {
                "base": MODEL_BASE,
                "name": model_attempt,
                "parameters": params,
                "layers": "embeddings.pkl",
                "mappings": "nodes.csv",
                "state": "state_dict.pt",
                "scores": scores,
                "time": dateTime,
            }.update(args.__dict__)

            pickle.dump(m.get_embeddings(dataset.global_id_map), open(join(MODEL_BASE, metadata['layers']), "wb"))
            pickle.dump(dataset, open(join(MODEL_BASE, "dataset.pkl"), "wb"))

            with open(join(MODEL_BASE, "metadata.json"), "w") as mdata:
                mdata.write(json.dumps(metadata, indent=4))

            torch.save(
                {
                    'model_state_dict': m.state_dict(),
                    'splits': dataset.splits
                },
                join(MODEL_BASE, metadata['state'])
            )

            dataset.nodes.to_csv(join(MODEL_BASE, "nodes.csv"), index=False)
            dataset.edges.to_csv(join(MODEL_BASE, "edges.csv"), index=False)
            dataset.held.to_csv(join(MODEL_BASE, "held.csv"), index=False)

            print("done")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--training_mode', dest='training_mode', default=None,
                        help='Selects one of training procedures [node_classifier|vector_sim|vector_sim_classifier|predict_next_function|multitask]')
    parser.add_argument('--node_path', dest='node_path', default=None,
                        help='Path to the file with nodes')
    parser.add_argument('--edge_path', dest='edge_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--holdout', dest='holdout', default=None,
                        help='Pre-generated holdout set')
    parser.add_argument('--train_frac', dest='train_frac', default=0.6, type=float,
                        help='Pre-generated holdout set')
    parser.add_argument('--call_seq_file', dest='call_seq_file', default=None,
                        help='Path to the file with edges that represent API call sequence. Used only with training mode \'predict_next_function\'')
    parser.add_argument('--fname_file', dest='fname_file', default=None,
                        help='Path to the file with edges that show function names')
    parser.add_argument('--varuse_file', dest='varuse_file', default=None,
                        help='Path to the file with edges that show variable names')
    parser.add_argument('--data_file', dest='data_file', default=None,
                        help='Path to the file with edges that are used for training')
    parser.add_argument('--filter_edges', dest='filter_edges', default=None,
                        help='Edges filtered before training')
    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--note', dest='note', default="",
                        help='Note, added to metadata')
    parser.add_argument('model_output_dir', dest='model_output_dir',
                        help='Location of the final model')
    parser.add_argument('--use_node_types', action='store_true')
    parser.add_argument('--restore_state', action='store_true')
    parser.add_argument('--self_loops', action='store_true')
    parser.add_argument('--override_labels', action='store_true')
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int,
                        help='')

    args = parser.parse_args()

    models_ = {
        # GCNSampling: gcnsampling_params,
        # GATSampler: gatsampling_params,
        RGCNSampling: rgcnsampling_params
    }

    # data_paths = pandas.read_csv("../../graph-network/deprecated/data_paths.tsv", sep="\t")
    # MODELS_PATH = "../../graph-network/models"
    # MODELS_PATH = args.model_output_dir
    # EPOCHS = args.epochs

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    main(models_, args)