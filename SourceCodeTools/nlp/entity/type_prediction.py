from __future__ import unicode_literals, print_function

import json
import logging
import os
import pickle
from copy import copy
from datetime import datetime

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from pathlib import Path

import tensorflow

from SourceCodeTools.nlp.batchers import PythonBatcher
from SourceCodeTools.nlp.entity import parse_biluo
from SourceCodeTools.nlp.entity.tf_models.params import cnn_params
from SourceCodeTools.nlp.entity.utils import get_unique_entities, overlap
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


def compute_precision_recall_f1(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def localized_f1(pred_spans, true_spans, eps=1e-8):

    tp = 0.
    fp = 0.
    fn = 0.

    for pred, true in zip(pred_spans, true_spans):
        if true != "O":
            if true == pred:
                tp += 1
            else:
                if pred == "O":
                    fn += 1
                else:
                    fp += 1

    return compute_precision_recall_f1(tp, fp, fn)


def span_f1(pred_spans, true_spans, eps=1e-8):
    tp = len(pred_spans.intersection(true_spans))
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    return compute_precision_recall_f1(tp, fp, fn)


def scorer(pred, labels, tagmap, no_localization=False, eps=1e-8):
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

    if not no_localization:
        pred_spans = set(parse_biluo(pred_biluo))
        true_spans = set(parse_biluo(labels_biluo))

        precision, recall, f1 = span_f1(pred_spans, true_spans, eps=eps)
    else:
        precision, recall, f1 = localized_f1(pred_biluo, labels_biluo, eps=eps)

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
            output_dir=None, epochs=30, batch_size=32, seq_len=100, finetune=False, trials=1,
            no_localization=False, ckpt_path=None, no_graph=False):
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
        self.no_localization = no_localization
        self.ckpt_path = ckpt_path
        self.no_graph = no_graph

    def set_batcher_class(self):
        self.batcher = PythonBatcher

    def set_model_class(self):
        from SourceCodeTools.nlp.entity.tf_models.tf_model import TypePredictor
        self.model = TypePredictor

    def get_batcher(self, *args, **kwargs):
        return self.batcher(*args, **kwargs)

    def get_model(self, *args, **kwargs):
        model = self.model(*args, **kwargs)
        if self.ckpt_path is not None:
            model.load_weights(os.path.join(self.ckpt_path, "checkpoint"))
        return model

    def train(self, *args, **kwargs):
        from SourceCodeTools.nlp.entity.tf_models.tf_model import train
        return train(*args, summary_writer=self.summary_writer, **kwargs)

    def test(self, *args, **kwargs):
        from SourceCodeTools.nlp.entity.tf_models.tf_model import test
        return test(*args, **kwargs)

    def create_summary_writer(self, path):
        self.summary_writer = tensorflow.summary.create_file_writer(path)

    # def write_summary(self, value_name, value, step):
    #     with self.summary_writer.as_default():
    #         tensorflow.summary.scalar(value_name, value, step=step)

    def get_dataloaders(self, word_emb, graph_emb, suffix_prefix_buckets, **kwargs):

        if self.ckpt_path is not None:
            tagmap = pickle.load(open(os.path.join(self.ckpt_path, "tag_types.pkl"), "rb"))
        else:
            tagmap = None

        train_batcher = self.get_batcher(
            self.train_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb.ind if graph_emb is not None else None,
            wordmap=word_emb.ind, tagmap=tagmap, tokenizer="codebert",
            class_weights=False, element_hash_size=suffix_prefix_buckets, no_localization=self.no_localization,
            **kwargs
        )
        test_batcher = self.get_batcher(
            self.test_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb.ind if graph_emb is not None else None,
            wordmap=word_emb.ind, tokenizer="codebert",
            tagmap=train_batcher.tagmap,  # use the same mapping
            class_weights=False, element_hash_size=suffix_prefix_buckets,  # class_weights are not used for testing
            no_localization=self.no_localization, **kwargs
        )
        return train_batcher, test_batcher

    def train_model(self):

        graph_emb = load_pkl_emb(self.graph_emb_path) if self.graph_emb_path is not None else None
        word_emb = load_pkl_emb(self.word_emb_path)

        suffix_prefix_buckets = params.pop("suffix_prefix_buckets")

        train_batcher, test_batcher = self.get_dataloaders(word_emb, graph_emb, suffix_prefix_buckets)

        print(f"\n\n{params}")
        lr = params.pop("learning_rate")
        lr_decay = params.pop("learning_rate_decay")

        timestamp = str(datetime.now()).replace(":","-").replace(" ","_")
        param_dir = os.path.join(output_dir, timestamp)
        os.mkdir(param_dir)

        for trial_ind in range(self.trials):
            trial_dir = os.path.join(param_dir, repr(trial_ind))
            logging.info(f"Running trial: {timestamp}")
            os.mkdir(trial_dir)
            self.create_summary_writer(trial_dir)

            model = self.get_model(
                word_emb, graph_emb, train_embeddings=self.finetune, suffix_prefix_buckets=suffix_prefix_buckets,
                num_classes=train_batcher.num_classes, seq_len=self.seq_len, no_graph=self.no_graph, **params
            )

            def save_ckpt_fn():
                checkpoint_path = os.path.join(trial_dir, "checkpoint")
                model.save_weights(checkpoint_path)

            train_losses, train_f1, test_losses, test_f1 = self.train(
                model=model, train_batches=train_batcher, test_batches=test_batcher,
                epochs=self.epochs, learning_rate=lr, scorer=lambda pred, true: scorer(pred, true, train_batcher.tagmap, no_localization=self.no_localization),
                learning_rate_decay=lr_decay, finetune=self.finetune, save_ckpt_fn=save_ckpt_fn, no_localization=self.no_localization
            )

            # checkpoint_path = os.path.join(trial_dir, "checkpoint")
            # model.save_weights(checkpoint_path)

            metadata = {
                "train_losses": train_losses,
                "train_f1": train_f1,
                "test_losses": test_losses,
                "test_f1": test_f1,
                "learning_rate": lr,
                "learning_rate_decay": lr_decay,
                "epochs": self.epochs,
                "suffix_prefix_buckets": suffix_prefix_buckets,
                "seq_len": self.seq_len,
                "batch_size": self.batch_size,
                "no_localization": self.no_localization,
                "no_graph": self.no_graph,
                "finetune": self.finetune,
                "word_emb_path": self.word_emb_path,
                "graph_emb_path": self.graph_emb_path
            }

            print("Maximum f1:", max(test_f1))

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
    parser.add_argument('--type_ann_edges', dest='type_ann_edges', default=None,
                        help='Path to type annotation edges')
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
    parser.add_argument('--min_entity_count', dest='min_entity_count', default=3, type=int,
                        help='')
    parser.add_argument('--pretraining_epochs', dest='pretraining_epochs', default=0, type=int,
                        help='')
    parser.add_argument('--ckpt_path', dest='ckpt_path', default=None, type=str,
                        help='')
    parser.add_argument('--epochs', dest='epochs', default=500, type=int,
                        help='')
    parser.add_argument('--trials', dest='trials', default=1, type=int,
                        help='')
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int,
                        help='Does not work with Tensorflow backend')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--no_localization', action='store_true')
    parser.add_argument('--restrict_allowed', action='store_true', default=False)
    parser.add_argument('--no_graph', action='store_true', default=False)
    parser.add_argument('model_output',
                        help='')

    args = parser.parse_args()

    if args.finetune is False and args.pretraining_epochs > 0:
        logging.info(f"Finetuning is disabled, but the the number of pretraining epochs is {args.pretraining_epochs}. Setting pretraining epochs to 0.")
        args.pretraining_epochs = 0

    if args.graph_emb_path is not None and not os.path.isfile(args.graph_emb_path):
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

    # allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
    #            'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}
    if args.restrict_allowed:
        allowed = {
            'str', 'Optional', 'int', 'Any', 'Union', 'bool', 'Callable', 'Dict', 'bytes', 'float', 'Description',
            'List', 'Sequence', 'Namespace', 'T', 'Type', 'object', 'HTTPServerRequest', 'Future', "Matcher"
        }
    else:
        allowed = None

    # train_data, test_data = read_data(
    #     open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True, include_only="entities",
    #     min_entity_count=args.min_entity_count, random_seed=args.random_seed
    # )

    dataset_dir = Path(args.data_path).parent
    train_data = filter_labels(
        pickle.load(open(dataset_dir.joinpath("type_prediction_dataset_no_defaults_train.pkl"), "rb")),
        allowed=allowed
    )
    test_data = filter_labels(
        pickle.load(open(dataset_dir.joinpath("type_prediction_dataset_no_defaults_test.pkl"), "rb")),
        allowed=allowed
    )

    unique_entities = get_unique_entities(train_data, field="entities")
    save_entities(output_dir, unique_entities)

    for params in cnn_params:
        trainer = ModelTrainer(
            train_data, test_data, params, graph_emb_path=args.graph_emb_path, word_emb_path=args.word_emb_path,
            output_dir=output_dir, epochs=args.epochs, batch_size=args.batch_size,
            finetune=args.finetune, trials=args.trials, seq_len=args.max_seq_len, no_localization=args.no_localization,
            ckpt_path=args.ckpt_path, no_graph=args.no_graph
        )
        trainer.train_model()
