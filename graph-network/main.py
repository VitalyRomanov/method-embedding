# %%
import sys
from models import GAT, RGCN
from datetime import datetime
from params import gat_params, rgcn_params
import pandas
import pickle
import json
from os import mkdir
from os.path import isdir, join
import torch
from Dataset import SourceGraphDataset


def get_name(model, timestamp):
    return "{} {}".format(model.__name__, timestamp).replace(":", "-").replace(" ", "-").replace(".", "-")


def main(nodes_path, edges_path, models, desc, args):
    """

    :param nodes_path:
    :param edges_path:
    :param models:
    :param desc:
    :param args:
    :return:
    """

    for model, param_grid in models.items():
        for params in param_grid:

            LABELS_FROM = "type"

            dateTime = str(datetime.now())
            print("\n\n")
            print(dateTime)
            print("Model: {}, Params: {}, Desc: {}".format(model.__name__, params, desc))

            if model.__name__ == "GAT":
                dataset = SourceGraphDataset(nodes_path, edges_path, label_from=LABELS_FROM,
                                             restore_state=args.restore_state)
            elif model.__name__ == "RGCN":
                dataset = SourceGraphDataset(nodes_path,
                                             edges_path,
                                             label_from=LABELS_FROM,
                                             node_types=args.use_node_types,
                                             edge_types=True,
                                             restore_state=args.restore_state
                                             )
            else:
                raise Exception("Unknown model: {}".format(model.__name__))

            model_attempt = get_name(model, dateTime)

            MODEL_BASE = join(MODELS_PATH, model_attempt)

            if not isdir(MODEL_BASE):
                mkdir(MODEL_BASE)

            if args.training_mode == 'node_classifier':

                from train_node_classifier import training_procedure

                m, scores = training_procedure(dataset, model, params, EPOCHS, args.restore_state)

            elif args.training_mode == "vector_sim":

                from train_vector_sim import training_procedure

                m, ee, scores = training_procedure(dataset, model, params, EPOCHS, args.restore_state)

                torch.save(
                    {
                        'elem_embeder': ee.state_dict(),
                    },
                    join(MODEL_BASE, "vector_sim.pt")
                )

            elif args.training_mode == "vector_sim_classifier":

                from train_vector_sim_with_classifier import training_procedure

                m, ee, lp, scores = training_procedure(dataset, model, params, EPOCHS, args.data_file,
                                                       args.restore_state)

                torch.save(
                    {
                        'elem_embeder': ee.state_dict(),
                        'link_predictor': lp.state_dict(),
                    },
                    join(MODEL_BASE, "vector_sim_with_classifier.pt")
                )

            elif args.training_mode == "predict_next_function":

                from train_vector_sim_next_call import training_procedure

                m, ee, lp, scores = training_procedure(dataset, model, params, EPOCHS, args.call_seq_file,
                                                       args.restore_state)

                torch.save(
                    {
                        'elem_embeder': ee.state_dict(),
                        'link_predictor': lp.state_dict(),
                    },
                    join(MODEL_BASE, "vector_sim_next_call.pt")
                )
            elif args.training_mode == "multitask":

                from train_multitask import training_procedure

                m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, scores = \
                    training_procedure(dataset, model, params, EPOCHS, args.call_seq_file, args.fname_file,
                                       args.varuse_file, args.restore_state)

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
                "description": desc,
                "training_mode": args.training_mode,
                "datafile": args.data_file,
                "call_seq": args.call_seq_file,
                "fname_file": args.fname_file,
                "varuse_file": args.varuse_file
            }

            pickle.dump(m.get_embeddings(dataset.global_id_map), open(join(metadata['base'], metadata['layers']), "wb"))

            with open(join(metadata['base'], "metadata.json"), "w") as mdata:
                mdata.write(json.dumps(metadata, indent=4))

            torch.save(
                {
                    'model_state_dict': m.state_dict(),
                    'splits': dataset.splits
                },
                join(metadata['base'], metadata['state'])
            )

            dataset.nodes.to_csv(join(metadata['base'], "nodes.csv"), index=False)
            dataset.edges.to_csv(join(metadata['base'], "edges.csv"), index=False)
            dataset.held.to_csv(join(metadata['base'], "held.csv"), index=False)

            print("done")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--training_mode', dest='training_mode', default=None,
                        help='Selects one of training procedures [node_classifier|vector_sim|vector_sim_classifier|predict_next_function]')
    parser.add_argument('--call_seq_file', dest='call_seq_file', default=None,
                        help='Path to the file with edges that represent API call sequence. Used only with training mode \'predict_next_function\'')
    parser.add_argument('--node_path', dest='node_path', default=None,
                        help='Path to the file with nodes')
    parser.add_argument('--edge_path', dest='edge_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--fname_file', dest='fname_file', default=None,
                        help='Path to the file with edges that show function names')
    parser.add_argument('--varuse_file', dest='varuse_file', default=None,
                        help='Path to the file with edges that show variable names')
    parser.add_argument('--data_file', dest='data_file', default=None,
                        help='Path to the file with edges that are used for training')
    parser.add_argument('--use_node_types', action='store_true')
    parser.add_argument('--restore_state', action='store_true')

    args = parser.parse_args()

    models_ = {
        # GAT: gat_params,
        RGCN: rgcn_params
    }

    data_paths = pandas.read_csv("data_paths.tsv", sep="\t")
    MODELS_PATH = "models"
    EPOCHS = 40

    if not isdir(MODELS_PATH):
        mkdir(MODELS_PATH)

    main(args.node_path, args.edge_path, models_, "full", args)

    # for ind, row in data_paths.iterrows():
    #
    #     if ind == 1: break
    #     node_path = row.nodes
    #     edge_path = row.edges_train
    #     desc_ = row.desc
    #     # node_path = "/Volumes/External/dev/method-embeddings/res/python/normalized_sourcetrail_nodes.csv"
    #     # edge_path = "/Volumes/External/dev/method-embeddings/res/python/edges.csv"
    #     # node_path = "/home/ltv/data/datasets/source_code/python-source-graph/02_largest_component/nodes_component_0.csv.bz2"
    #     # edge_path = "/home/ltv/data/datasets/source_code/python-source-graph/02_largest_component/edges_component_0.csv.bz2"
    #     # node_path = "/home/ltv/data/datasets/source_code/sample-python/normalized_sourcetrail_nodes.csv"
    #     # edge_path = "/home/ltv/data/datasets/source_code/sample-python/edges.csv"
    #
    #     # nodes_, edges_ = load_data(node_path, edge_path)
    #
    #     main(args.node_path, args.edge_path, models_, desc_, args)
    #     break
