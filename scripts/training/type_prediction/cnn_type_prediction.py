from __future__ import unicode_literals, print_function

import os
import pickle
from copy import copy

from SourceCodeTools.cli_arguments import TypePredictorTrainerArgumentParser
from SourceCodeTools.nlp.entity.utils import get_unique_entities

from SourceCodeTools.nlp.entity.utils.data import read_json_data
from SourceCodeTools.nlp.trainers.cnn_entity_trainer import ModelTrainer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_pkl_emb(path):
    """
    Load graph embeddings from a pickle file. Embeddings are stored in class Embedder or in a list of Embedders. The
        last embedder in the list is returned.
    :param path: path to graph embeddings stored as Embedder pickle
    :return: Embedder object
    """
    embedder = pickle.load(open(path, "rb"))
    if isinstance(embedder, list):
        embedder = embedder[-1]
    return embedder


def save_entities(path, entities):
    with open(os.path.join(path, 'entities.txt'), "w") as entitiesfile:
        for e in entities:
            entitiesfile.write(f"{e}\n")


def filter_labels(dataset, allowed=None, field=None):
    if allowed is None:
        return dataset
    dataset = copy(dataset)
    for sent, annotations in dataset:
        annotations["entities"] = [e for e in annotations["entities"] if e[2] in allowed]
    return dataset


def find_example(dataset, needed_label):
    for sent, annotations in dataset:
        for start, end, e in annotations["entities"]:
            if e == needed_label:
                print(f"{sent}: {sent[start: end]}")


if __name__ == "__main__":
    config = TypePredictorTrainerArgumentParser().parse()

    if config["DATASET"]["restrict_allowed"]:
        allowed = {
            'str', 'Optional', 'int', 'Any', 'Union', 'bool', 'Callable', 'Dict', 'bytes', 'float', 'Description',
            'List', 'Sequence', 'Namespace', 'T', 'Type', 'object', 'HTTPServerRequest', 'Future', "Matcher"
        }
    else:
        allowed = None

    train_data, test_data = read_json_data(
        config["DATASET"]["data_path"], normalize=True, allowed=allowed, include_replacements=True, include_only="entities",
        min_entity_count=config["DATASET"]["min_entity_count"]
    )

    config["MODEL"].update({
        'h_sizes': [100, 100, 100],
        'dense_size': 60,
        'pos_emb_size': 50,
        'cnn_win_size': 7,
        'suffix_prefix_dims': 70,
    })

    # for model_params in cnn_params:
    trainer = ModelTrainer(
        train_data, test_data, config
    )
    trainer.train_model()
