#%%
import sys
from data import load_data
from graphtools import *
import numpy as np
from models import GAT, RGCN, train, final_evaluation, get_train_test_val_indices
from datetime import datetime
from params import gat_params, rgcn_params
import pandas
import pickle
import json
from os import mkdir
from os.path import isdir, join
import torch

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


def main(nodes, edges, models, desc):

    for model, param_grid in models.items():
        for params in param_grid:

            dateTime = str(datetime.now())
            print("\n\n")
            print(dateTime)
            print("Model: {}, Params: {}, Desc: {}".format(model.__name__, params, desc))

            if model.__name__ == "GAT":
                g, labels, node_mappings = create_graph(nodes, edges)
            elif model.__name__ == "RGCN":
                g, labels, node_mappings = create_hetero_graph(nodes, edges)
            else:
                raise Exception("Unknown model: {}".format(model.__name__))

            m = model(g,
                      num_classes=np.unique(labels).size,
                      activation=torch.nn.functional.leaky_relu,
                      **params)

            splits = get_train_test_val_indices(labels)
            try:
                train(m, labels, EPOCHS, splits)
            except KeyboardInterrupt:
                print("Training interrupted")
            finally:
                m.eval()
                scores = final_evaluation(m, labels, splits)

            print("Saving...", end="")

            model_attempt = "{} {}".format(model.__name__, dateTime).repalce(":","-").repalce(" ","-").repalce(".","-")

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

            pickle.dump(m.get_embeddings(node_mappings), open(join(metadata['base'], metadata['layers']), "wb"))

            # node_mappings.to_csv(join(metadata['base'], metadata['mappings']), index=False)

            with open(join(metadata['base'], "metadata.json"), "w") as mdata:
                mdata.write(json.dumps(metadata, indent=4))

            torch.save(
                {
                    'model_state_dict': m.state_dict(),
                    'splits': splits
                },
                join(metadata['base'], metadata['state'])
            )

            # with open("mode_file_log.log", "a") as filelog:
            #     filelog.write("%s\t%s\n" % (model_filename, repr(metadata)))

            print("done")


if __name__ == "__main__":

    models_ = {
        GAT: gat_params,
        RGCN: rgcn_params
    }

    data_paths = pandas.read_csv("data_paths.tsv", sep="\t")
    MODELS_PATH = "models"
    EPOCHS = 300

    if not isdir(MODELS_PATH):
        mkdir(MODELS_PATH)

    for ind, row in data_paths.iterrows():

        if ind == 3: break
        node_path = row.nodes
        edge_path = row.edges_train
        desc_ = row.desc

        nodes_, edges_ = load_data(node_path, edge_path)

        main(nodes_, edges_, models_, desc_)
