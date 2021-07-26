from __future__ import unicode_literals, print_function

import json
import logging
import os
import pickle
from copy import copy
from datetime import datetime

import tensorflow

from SourceCodeTools.nlp.batchers import PythonBatcher
from SourceCodeTools.nlp.entity import parse_biluo
from SourceCodeTools.nlp.entity.tf_models.params import cnn_params
from SourceCodeTools.nlp.entity.utils import get_unique_entities
from SourceCodeTools.nlp.entity.utils.data import read_data


def load_pkl_emb(path):
    """
    Load graph embeddings from a pickle file. Embeddigns are stored in class Embedder or in a list of Embedders. The
        last embedder in the list is returned.
    :param path: path to graph embeddigs stored as Embedder pickle
    :return: Embedder object
    """
    embedder = pickle.load(open(path, "rb"))
    if isinstance(embedder, list):
        embedder = embedder[-1]
    return embedder


def scorer(pred, labels, tagmap, eps=1e-8):
    """
    Compute f1 score, precision, and recall from BILUO labels
    :param pred: predicted BILUO labels
    :param labels: ground truth BILUO labels
    :param tagmap:
    :param eps:
    :return:
    """
    # TODO
    # the scores can be underestimated because ground truth does not contain all possible labels
    # this results in higher reported false alarm rate
    pred_biluo = [tagmap.inverse(p) for p in pred]
    labels_biluo = [tagmap.inverse(p) for p in labels]

    pred_spans = set(parse_biluo(pred_biluo))
    true_spans = set(parse_biluo(labels_biluo))

    tp = len(pred_spans.intersection(true_spans))
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def write_config(trial_dir, params, extra_params=None):
    config_path = os.path.join(trial_dir, "model_config.conf")

    import configparser

    params = copy(params)
    if extra_params is not None:
        params.update(extra_params)

    config = configparser.ConfigParser()
    config['DEFAULT'] = params

    with open(config_path, 'w') as configfile:
        config.write(configfile)


class ModelTrainer:
    def __init__(self, train_data, test_data, params, graph_emb_path=None, word_emb_path=None,
            output_dir=None, epochs=30, batch_size=32, seq_len=100, finetune=False, trials=1):
        self.set_batcher_class()
        self.set_model_class()

        self.train_data = train_data
        self.test_data = test_data
        self.model_params = params
        self.graph_emb_path = graph_emb_path
        self.word_emb_path = word_emb_path
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.finetune = finetune
        self.trials = trials
        self.seq_len = seq_len

    def set_batcher_class(self):
        self.batcher = PythonBatcher

    def set_model_class(self):
        from SourceCodeTools.nlp.entity.tf_models.tf_model import TypePredictor
        self.model = TypePredictor

    def get_batcher(self, *args, **kwards):
        return self.batcher(*args, **kwards)

    def get_model(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, *args, **kwargs):
        from SourceCodeTools.nlp.entity.tf_models.tf_model import train
        return train(*args, summary_writer=self.summary_writer, **kwargs)

    def create_summary_writer(self, path):
        self.summary_writer = tensorflow.summary.create_file_writer(path)

    # def write_summary(self, value_name, value, step):
    #     with self.summary_writer.as_default():
    #         tensorflow.summary.scalar(value_name, value, step=step)

    def train_model(self):

        graph_emb = load_pkl_emb(self.graph_emb_path)
        word_emb = load_pkl_emb(self.word_emb_path)

        suffix_prefix_buckets = params.pop("suffix_prefix_buckets")

        train_batcher = self.get_batcher(
            train_data, self.batch_size, seq_len=self.seq_len, graphmap=graph_emb.ind, wordmap=word_emb.ind, tagmap=None,
            class_weights=False, element_hash_size=suffix_prefix_buckets
        )
        test_batcher = self.get_batcher(
            test_data, self.batch_size, seq_len=self.seq_len, graphmap=graph_emb.ind, wordmap=word_emb.ind,
            tagmap=train_batcher.tagmap,  # use the same mapping
            class_weights=False, element_hash_size=suffix_prefix_buckets  # class_weights are not used for testing
        )

        print(f"\n\n{params}")
        lr = params.pop("learning_rate")
        lr_decay = params.pop("learning_rate_decay")

        param_dir = os.path.join(output_dir, str(datetime.now()))
        os.mkdir(param_dir)

        for trial_ind in range(self.trials):
            trial_dir = os.path.join(param_dir, repr(trial_ind))
            os.mkdir(trial_dir)
            self.create_summary_writer(trial_dir)

            model = self.get_model(
                word_emb, graph_emb, train_embeddings=self.finetune, suffix_prefix_buckets=suffix_prefix_buckets,
                num_classes=train_batcher.num_classes(), seq_len=self.seq_len, **params
            )

            train_losses, train_f1, test_losses, test_f1 = self.train(
                model=model, train_batches=train_batcher, test_batches=test_batcher,
                epochs=self.epochs, learning_rate=lr, scorer=lambda pred, true: scorer(pred, true, train_batcher.tagmap),
                learning_rate_decay=lr_decay, finetune=self.finetune
            )

            checkpoint_path = os.path.join(trial_dir, "checkpoint")
            model.save_weights(checkpoint_path)

            metadata = {
                "train_losses": train_losses,
                "train_f1": train_f1,
                "test_losses": test_losses,
                "test_f1": test_f1,
                "learning_rate": lr,
                "learning_rate_decay": lr_decay,
                "epochs": self.epochs,
                "suffix_prefix_buckets": suffix_prefix_buckets,
                "seq_len": self.seq_len
            }

            # write_config(trial_dir, params, extra_params={"suffix_prefix_buckets": suffix_prefix_buckets, "seq_len": seq_len})

            metadata.update(params)

            with open(os.path.join(trial_dir, "params.json"), "w") as metadata_sink:
                metadata_sink.write(json.dumps(metadata, indent=4))

            pickle.dump(train_batcher.tagmap, open(os.path.join(trial_dir, "tag_types.pkl"), "wb"))


def get_type_prediction_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', dest='data_path', default=None,
                        help='Path to the dataset file')
    parser.add_argument('--graph_emb_path', dest='graph_emb_path', default=None,
                        help='Path to the file with graph embeddings')
    parser.add_argument('--word_emb_path', dest='word_emb_path', default=None,
                        help='Path to the file with token embeddings')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, type=float,
                        help='')
    parser.add_argument('--learning_rate_decay', dest='learning_rate_decay', default=1.0, type=float,
                        help='')
    parser.add_argument('--random_seed', dest='random_seed', default=None, type=int,
                        help='')
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='')
    parser.add_argument('--max_seq_len', dest='max_seq_len', default=100, type=int,
                        help='')
    parser.add_argument('--pretraining_epochs', dest='pretraining_epochs', default=0, type=int,
                        help='')
    parser.add_argument('--epochs', dest='epochs', default=500, type=int,
                        help='')
    parser.add_argument('--trials', dest='trials', default=1, type=int,
                        help='')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('model_output',
                        help='')

    args = parser.parse_args()

    if args.finetune is False and args.pretraining_epochs > 0:
        logging.info(f"Finetuning is disabled, but the the number of pretraining epochs is {args.pretraining_epochs}. Setting pretraining epochs to 0.")
        args.pretraining_epochs = 0

    if not os.path.isfile(args.graph_emb_path):
        logging.warning(f"File with graph embeddings does not exist: {args.graph_emb_path}")
        args.graph_emb_path = None
    return args


def save_entities(path, entities):
    with open(os.path.join(path, 'entities.txt'), "w") as entitiesfile:
        for e in entities:
            entitiesfile.write(f"{e}\n")


if __name__ == "__main__":
    args = get_type_prediction_arguments()

    output_dir = args.model_output
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
    #            'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}

    train_data, test_data = read_data(
        open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True, include_only="entities",
        min_entity_count=1, random_seed=args.random_seed
    )

    unique_entities = get_unique_entities(train_data, field="entities")
    save_entities(output_dir, unique_entities)

    for params in cnn_params:
        trainer = ModelTrainer(
            train_data, test_data, params, graph_emb_path=args.graph_emb_path, word_emb_path=args.word_emb_path,
            output_dir=output_dir, epochs=args.epochs, batch_size=args.batch_size,
            finetune=args.finetune, trials=args.trials, seq_len=args.max_seq_len,
        )
        trainer.train_model()
