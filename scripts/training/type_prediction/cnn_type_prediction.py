from __future__ import unicode_literals, print_function

import logging
import os
import pickle
from copy import copy
from os.path import isfile, isdir

from SourceCodeTools.cli_arguments import TypePredictorTrainerArguments
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


# def write_config(trial_dir, params, extra_params=None):
#     config_path = os.path.join(trial_dir, "model_config.conf")
#
#     import configparser
#
#     params = copy(params)
#     if extra_params is not None:
#         params.update(extra_params)
#
#     config = configparser.ConfigParser()
#     config['DEFAULT'] = params
#
#     with open(config_path, 'w') as configfile:
#         config.write(configfile)


def get_type_prediction_arguments():
    parser = TypePredictorTrainerArguments()
    args = parser.parse()

    if args.finetune is False and args.pretraining_epochs > 0:
        logging.info(
            f"Fine-tuning is disabled, but the the number of pretraining epochs is {args.pretraining_epochs}. Setting pretraining epochs to 0.")
        args.pretraining_epochs = 0

    if args.graph_emb_path is not None and not (isfile(args.graph_emb_path) or isdir(args.graph_emb_path)):
        logging.warning(f"File with graph embeddings does not exist: {args.graph_emb_path}")
        args.graph_emb_path = None
    return args


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
    args = get_type_prediction_arguments()

    output_dir = args.model_output
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if args.restrict_allowed:
        allowed = {
            'str', 'Optional', 'int', 'Any', 'Union', 'bool', 'Callable', 'Dict', 'bytes', 'float', 'Description',
            'List', 'Sequence', 'Namespace', 'T', 'Type', 'object', 'HTTPServerRequest', 'Future', "Matcher"
        }
    else:
        allowed = None

    train_data, test_data = read_json_data(
        args.data_path, normalize=True, allowed=allowed, include_replacements=True, include_only="entities",
        min_entity_count=args.min_entity_count, random_seed=args.random_seed
    )

    unique_entities = get_unique_entities(train_data, field="entities")
    save_entities(output_dir, unique_entities)

    trainer_params = copy(args.__dict__)

    model_params = {
        'h_sizes': [100, 100, 100],
        'dense_size': 60,
        'pos_emb_size': 50,
        'cnn_win_size': 7,
        'suffix_prefix_dims': 70,
    }

    # for model_params in cnn_params:
    trainer = ModelTrainer(
        train_data, test_data, model_params, trainer_params
    )
    trainer.train_model()
