#%%
import sys
from data import load_data
from graphtools import *
import numpy as np
from models import GAT, RGCN, train, final_evaluation
from datetime import datetime
from params import gat_params, rgcn_params
import pandas
import pickle
import json
from os import mkdir
from os.path import isdir, join
import torch
from Dataset import SourceGraphDataset

# node_path = "/home/ltv/data/datasets/source_code/python-source-graph/normalized_sourcetrail_nodes.csv"
# edge_path = "/home/ltv/data/datasets/source_code/python-source-graph/non-ambiguous_edges.csv"
# node_path = "/home/ltv/data/datasets/source_code/sample-python/normalized_sourcetrail_nodes.csv"
# edge_path = "/home/ltv/data/datasets/source_code/sample-python/edges.csv"
# node_path = "/Volumes/External/dev/method-embeddings/res/python/normalized_sourcetrail_nodes.csv"
# edge_path = "/Volumes/External/dev/method-embeddings/res/python/edges.csv"


#%%
# print("Loading data...", end="")
# nodes, edges = load_data(node_path, edge_path)
# print("done")
# # print("Creating graph...", end="")
# # g, labels = create_graph(nodes, edges)
# # print("done")
#
# print("Using GPU:", torch.cuda.is_available())


#%%

def get_name(model, timestamp):
    return "{} {}".format(model.__name__, timestamp).replace(":","-").replace(" ","-").replace(".","-")



def main(nodes_path, edges_path, models, desc):

    for model, param_grid in models.items():
        for params in param_grid:

            LABELS_FROM = "type"

            dateTime = str(datetime.now())
            print("\n\n")
            print(dateTime)
            print("Model: {}, Params: {}, Desc: {}".format(model.__name__, params, desc))

            if model.__name__ == "GAT":
                dataset = SourceGraphDataset(nodes_path, edges_path, label_from=LABELS_FROM)
            elif model.__name__ == "RGCN":
                dataset = SourceGraphDataset(nodes_path,
                                  edges_path,
                                  label_from=LABELS_FROM,
                                  node_types=False,
                                  edge_types=True
                                  )
            else:
                raise Exception("Unknown model: {}".format(model.__name__))

            m = model(dataset.g,
                      num_classes=dataset.num_classes,
                      activation=torch.nn.functional.leaky_relu,
                      **params)

            try:
                train(m, dataset.labels, dataset.splits, EPOCHS)
            except KeyboardInterrupt:
                print("Training interrupted")
            finally:
                m.eval()
                scores = final_evaluation(m, dataset.labels, dataset.splits)

            print("Saving...", end="")

            model_attempt = get_name(model, dateTime)

            MODEL_BASE = join(MODELS_PATH, model_attempt)

            if not isdir(MODEL_BASE):
                mkdir(MODEL_BASE)

            metadata = {
                "base": MODEL_BASE,
                "name": model_attempt,
                "parameters": params,
                "layers": "embeddings.pkl",
                "mappings": "nodes.csv",
                "state": "state_dict.pt",
                "scores": scores,
                "time": dateTime,
                "description": desc
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

    models_ = {
        GAT: gat_params,
        # RGCN: rgcn_params
    }

    data_paths = pandas.read_csv("data_paths.tsv", sep="\t")
    MODELS_PATH = "models"
    EPOCHS = 150

    if not isdir(MODELS_PATH):
        mkdir(MODELS_PATH)

    for ind, row in data_paths.iterrows():

        if ind == 1: break
        node_path = row.nodes
        edge_path = row.edges_train
        desc_ = row.desc
        # node_path = "/Volumes/External/dev/method-embeddings/res/python/normalized_sourcetrail_nodes.csv"
        # edge_path = "/Volumes/External/dev/method-embeddings/res/python/edges.csv"
        node_path = "/home/ltv/data/datasets/source_code/python-source-graph/02_largest_component/nodes_component_0.csv.bz2"
        edge_path = "/home/ltv/data/datasets/source_code/python-source-graph/02_largest_component/edges_component_0.csv.bz2"
        # node_path = "/home/ltv/data/datasets/source_code/sample-python/normalized_sourcetrail_nodes.csv"
        # edge_path = "/home/ltv/data/datasets/source_code/sample-python/edges.csv"

        # nodes_, edges_ = load_data(node_path, edge_path)

        main(node_path, edge_path, models_, desc_)
        break
