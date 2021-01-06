import json
import logging
import pickle
from datetime import datetime
from os import mkdir
from os.path import isdir, join

import torch

from SourceCodeTools.data.sourcetrail.Dataset import read_or_create_dataset
from SourceCodeTools.graph.model import GCNSampling, GATSampler, RGCNSampling
from SourceCodeTools.graph.model.train.utils import get_name, get_model_base
from params import rgcnsampling_params


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

            if args.training_mode == 'node_classifier':

                from SourceCodeTools.graph.model.train.sampling_node_classifier import training_procedure

                m, scores = training_procedure(dataset, model, params, args.epochs, args, model_base)

            elif args.training_mode == "multitask":

                if args.intermediate_supervision:
                    params['use_self_loop'] = True
                    from SourceCodeTools.graph.model.train.sampling_multitask_intermediate_supervision import training_procedure
                else:
                    from SourceCodeTools.graph.model.train.sampling_multitask import training_procedure

                m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, scores = \
                    training_procedure(dataset, model, params, args.epochs, args, model_base)

                if args.intermediate_supervision:
                    torch.save(
                        {
                            'elem_embeder_fname': ee_fname.state_dict(),
                            'elem_embeder_varuse': ee_varuse.state_dict(),
                            'elem_embeder_apicall': ee_apicall.state_dict(),
                            'link_predictor_fname': [lp.state_dict() for lp in lp_fname],
                            'link_predictor_varuse': [lp.state_dict() for lp in lp_varuse],
                            'link_predictor_apicall': [lp.state_dict() for lp in lp_apicall],
                        },
                        join(model_base, "multitask.pt")
                    )
                else:
                    torch.save(
                        {
                            'elem_embeder_fname': ee_fname.state_dict(),
                            'elem_embeder_varuse': ee_varuse.state_dict(),
                            'elem_embeder_apicall': ee_apicall.state_dict(),
                            'link_predictor_fname': lp_fname.state_dict(),
                            'link_predictor_varuse': lp_varuse.state_dict(),
                            'link_predictor_apicall': lp_apicall.state_dict(),
                        },
                        join(model_base, "multitask.pt")
                    )
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

            pickle.dump(m.get_embeddings(dataset.global_id_map), open(join(model_base, metadata['layers']), "wb"))
            pickle.dump(dataset, open(join(model_base, "dataset.pkl"), "wb"))

            with open(join(model_base, "metadata.json"), "w") as mdata:
                mdata.write(json.dumps(metadata, indent=4))

            torch.save(
                {
                    'model_state_dict': m.state_dict(),
                    'splits': dataset.splits
                },
                join(model_base, metadata['state'])
            )

            dataset.nodes.to_csv(join(model_base, "nodes.csv"), index=False)
            dataset.edges.to_csv(join(model_base, "edges.csv"), index=False)

            print("done")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--training_mode', dest='training_mode', default=None,
                        help='Selects one of training procedures '
                             '[node_classifier|vector_sim|vector_sim_classifier|predict_next_function|multitask]')
    parser.add_argument('--node_path', dest='node_path', default=None,
                        help='Path to the file with nodes')
    parser.add_argument('--edge_path', dest='edge_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--holdout', dest='holdout', default=None,
                        help='Pre-generated holdout set')
    parser.add_argument('--train_frac', dest='train_frac', default=0.6, type=float,
                        help='Pre-generated holdout set')
    parser.add_argument('--call_seq_file', dest='call_seq_file', default=None,
                        help='Path to the file with edges that represent API call sequence. '
                             'Used only with training mode \'predict_next_function\'')
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
    parser.add_argument('model_output_dir',
                        help='Location of the final model')
    parser.add_argument('--use_node_types', action='store_true')
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
        RGCNSampling: rgcnsampling_params
    }

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)

    main(models_, args)
