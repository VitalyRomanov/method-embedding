from __future__ import unicode_literals, print_function

import json
import logging
import os
import pickle
import tensorflow
from collections import defaultdict
from copy import copy
from datetime import datetime
from pathlib import Path

from time import time

from SourceCodeTools.cli_arguments import TypePredictorTrainerArguments
from SourceCodeTools.code.data.file_utils import write_mapping_to_json
from SourceCodeTools.nlp.entity.entity_scores import entity_scorer
from SourceCodeTools.nlp.batchers import PythonBatcher
from SourceCodeTools.nlp.entity.utils import get_unique_entities
from tqdm import tqdm


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


class ModelTrainer:
    def __init__(
            self, train_data, test_data, model_params, trainer_params
    ):
        self.set_batcher_class()
        self.set_model_class()

        self.model_params = model_params
        self.trainer_params = trainer_params

        self.train_data = train_data
        self.test_data = test_data

        self.set_gpu()

    @property
    def learning_rate(self):
        return self.trainer_params["learning_rate"]

    @property
    def learning_rate_decay(self):
        return self.trainer_params["learning_rate_decay"]

    @property
    def suffix_prefix_buckets(self):
        return self.trainer_params["suffix_prefix_buckets"]

    @property
    def data_path(self):
        return self.trainer_params["data_path"]

    @property
    def graph_emb_path(self):
        return self.trainer_params["graph_emb_path"]

    @property
    def seq_len(self):
        return self.trainer_params["max_seq_len"]

    @property
    def batch_size(self):
        return self.trainer_params["batch_size"]

    @property
    def word_emb_path(self):
        return self.trainer_params["word_emb_path"]

    @property
    def mask_unlabeled_declarations(self):
        return self.trainer_params["mask_unlabeled_declarations"]

    # @property
    # def trials(self):
    #     return self.trainer_params["trials"]

    @property
    def finetune(self):
        return self.trainer_params["finetune"]

    @property
    def max_seq_len(self):
        return self.trainer_params["max_seq_len"]

    @property
    def no_graph(self):
        return self.trainer_params["no_graph"]

    @property
    def no_localization(self):
        return self.trainer_params["no_localization"]

    @property
    def epochs(self):
        return self.trainer_params["epochs"]

    @property
    def ckpt_path(self):
        return self.trainer_params["ckpt_path"]

    @property
    def gpu_id(self):
        return self.trainer_params["gpu"]

    @property
    def classes_for(self):
        return "tags"

    @property
    def vocab_mapping(self):
        if hasattr(self, "_vocab_mapping"):
            return self._vocab_mapping
        else:
            return None

    @property
    def best_score_metric(self):
        return "F1"

    def set_gpu(self):
        if self.gpu_id == -1:
            self.use_cuda = False
            self.device = "cpu"
        else:
            self.use_cuda = True
            self.device = f"cuda:{self.gpu_id}"

    def set_batcher_class(self):
        self.batcher = PythonBatcher

    def set_model_class(self):
        from SourceCodeTools.nlp.entity.tf_models.tf_model import TypePredictor
        self.model = TypePredictor

    def get_training_dir(self):
        if not hasattr(self, "_timestamp"):
            self._timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_")
        return Path(self.trainer_params["model_output"]).joinpath(self._timestamp)

    def get_trial_dir(self, trial_ind):
        return self.get_training_dir().joinpath(repr(trial_ind))

    def get_batcher(self, *args, **kwargs):
        return self.batcher(*args, **kwargs)

    def get_model(self, *args, **kwargs):
        model = self.model(*args, **kwargs)
        if self.ckpt_path is not None:
            ckpt_path = os.path.join(self.ckpt_path, "checkpoint")
            model = self.load_checkpoint(model, ckpt_path)
        return model

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
            mask_unlabeled_declarations=self.mask_unlabeled_declarations, **kwargs
        )
        test_batcher = self.get_batcher(
            self.test_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb.ind if graph_emb is not None else None,
            wordmap=word_emb.ind, tokenizer="codebert",
            tagmap=train_batcher.tagmap,  # use the same mapping
            class_weights=False, element_hash_size=suffix_prefix_buckets,  # class_weights are not used for testing
            no_localization=self.no_localization,
            mask_unlabeled_declarations=self.mask_unlabeled_declarations, **kwargs
        )
        return train_batcher, test_batcher

    def save_checkpoint(self, model, path):
        model.save_weights(path)

    def save_params(self, path, params, **kwargs):
        params = copy(params)
        params.update(kwargs)
        write_mapping_to_json(params, path.joinpath("params.json"))

    def load_checkpoint(self, model, path):
        model = model.load_weights(path)
        return model

    def _load_grap_embs(self):
        return load_pkl_emb(self.graph_emb_path) if self.graph_emb_path is not None else None

    def _load_word_embs(self):
        return load_pkl_emb(self.word_emb_path)

    def _create_summary_writer(self, path):
        self.summary_writer = tensorflow.summary.create_file_writer(str(path))

    def _write_to_summary(self, label, value, step):
        tensorflow.summary.scalar(label, value, step=step)

    def _create_optimizer(self, model):
        self._lr = tensorflow.Variable(self.learning_rate, trainable=False)
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self._lr)

    def _lr_scheduler_step(self):
        self._lr.assign(self._lr * self.learning_rate_decay)

    @classmethod
    def _format_batch(cls, batch, device):
        pass

    @classmethod
    def compute_loss_and_scores(
            cls, model, token_ids, prefix, suffix, graph_ids, labels, lengths, extra_mask=None, class_weights=None,
            scorer=None, finetune=False, vocab_mapping=None, training=False
    ):
        seq_mask = tensorflow.sequence_mask(lengths, token_ids.shape[1])
        logits = model(token_ids, prefix, suffix, graph_ids, target=None, training=training, mask=seq_mask)
        loss = model.loss(logits, labels, mask=seq_mask, class_weights=class_weights, extra_mask=extra_mask)
        # token_acc = tf.reduce_sum(tf.cast(tf.argmax(logits, axis=-1) == labels, tf.float32)) / (token_ids.shape[0] * token_ids.shape[1])
        scores = model.score(logits, labels, mask=seq_mask, scorer=scorer, extra_mask=extra_mask)

        scores["loss"] = loss
        return scores

    @classmethod
    def make_step(
            cls, model, optimizer, token_ids, prefix, suffix, graph_ids, labels, lengths, extra_mask=None,
            class_weights=None, scorer=None, finetune=False, train=False, **kwargs
    ):
        """
        Make a train step
        :param model: TypePrediction model instance
        :param optimizer: tf optimizer
        :param token_ids: ids for tokens, shape (?, seq_len)
        :param prefix: ids for prefixes, shape (?, seq_len)
        :param suffix: ids for suffixes, shape (?, seq_len)
        :param graph_ids: ids for graph nodes, shape (?, seq_len)
        :param labels: ids for labels, shape (?, )
        :param lengths: actual sequence lengths, shape (?, )
        :param extra_mask: additional mask to hide tokens that should be labeled, but are not labeled, shape (?, seq_len)
        :param class_weights: weight of each token, shape (?, seq_len)
        :param scorer: scorer function, takes `pred_labels` and `true_labels` as arguments
        :param finetune: whether to train embeddings
        :param train:
        :return: values for loss, precision, recall and f1-score
        """
        if train is True:
            with tensorflow.GradientTape() as tape:
                scores = cls.compute_loss_and_scores(
                    model, token_ids, prefix, suffix, graph_ids, labels, lengths, extra_mask=extra_mask,
                    class_weights=class_weights, scorer=scorer, finetune=finetune, training=train
                )
                gradients = tape.gradient(scores["loss"], model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            scores = cls.compute_loss_and_scores(
                model, token_ids, prefix, suffix, graph_ids, labels, lengths, extra_mask=extra_mask,
                class_weights=class_weights, scorer=scorer, finetune=finetune, training=train
            )

        scores["loss"] = float(scores["loss"])

        return scores

    def iterate_batches(self, model, batches, epoch, num_train_batches, train_scores, scorer, train=True):
        scores_for_averaging = defaultdict(list)

        batch_count = 0

        for ind, batch in enumerate(tqdm(batches, desc=f"Epoch {epoch}")):
            self._format_batch(batch, self.device)
            scores = self.make_step(
                model=model, optimizer=self.optimizer, token_ids=batch['tok_ids'],
                prefix=batch['prefix'], suffix=batch['suffix'],
                graph_ids=batch['graph_ids'] if 'graph_ids' in batch else None,
                labels=batch['tags'], lengths=batch['lens'],
                extra_mask=batch['no_loc_mask'] if self.no_localization else batch['hide_mask'],
                # class_weights=batch['class_weights'],
                scorer=scorer, finetune=self.finetune and epoch / self.epochs > 0.6,
                vocab_mapping=self.vocab_mapping,
                train=train
            )

            batch_count += 1

            scores["batch_size"] = batch['tok_ids'].shape[0]
            for score, value in scores.items():
                self._write_to_summary(f"{score}/{'Train' if train else 'Test'}", value,
                                       epoch * num_train_batches + ind)
                scores_for_averaging[score].append(value)
            train_scores.append(scores_for_averaging)

        return num_train_batches

    def iterate_epochs(self, train_batches, test_batches, epochs, model, scorer, save_ckpt_fn):

        num_train_batches = len(train_batches)
        num_test_batches = len(test_batches)
        train_scores = []
        test_scores = []

        train_average_scores = []
        test_average_scores = []

        best_score = 0.

        try:
            for e in range(epochs):

                start = time()

                num_train_batches = self.iterate_batches(
                    model, train_batches, e, num_train_batches, train_scores, scorer, train=True
                )
                num_test_batches = self.iterate_batches(
                    model, test_batches, e, num_test_batches, test_scores, scorer, train=False
                )

                epoch_time = time() - start

                def print_scores(scores, average_scores, partition):
                    for score, value in scores.items():
                        if score == "batch_size":
                            continue
                        avg_value = sum(value) / len(value)
                        average_scores[score] = avg_value
                        print(f"{partition} {score}: {avg_value: .4f}", end=" ")
                        self._write_to_summary(f"Average {score}/{partition}", avg_value, e)

                train_average_scores.append({})
                test_average_scores.append({})

                print(f"\nEpoch: {e}, {epoch_time: .2f} s", end=" ")
                print_scores(train_scores[-1], train_average_scores[-1], "Train")
                print_scores(test_scores[-1], test_average_scores[-1], "Test")
                print("\n")

                if save_ckpt_fn is not None and test_average_scores[-1][self.best_score_metric] > best_score:
                    save_ckpt_fn()
                    best_score = test_average_scores[-1][self.best_score_metric]

                self._lr_scheduler_step()
        except KeyboardInterrupt:
            pass

        return train_scores, test_scores, train_average_scores, test_average_scores

    def train(
            self, model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01,
            learning_rate_decay=1., finetune=False, save_ckpt_fn=None, no_localization=False
    ):

        self._create_optimizer(model)

        with self.summary_writer.as_default():
            train_scores, test_scores, train_average_scores, test_average_scores = self.iterate_epochs(train_batches,
                                                                                                       test_batches,
                                                                                                       epochs, model,
                                                                                                       scorer,
                                                                                                       save_ckpt_fn)

        return train_scores, test_scores, train_average_scores, test_average_scores

    def train_model(self):

        graph_emb = self._load_grap_embs()
        word_emb = self._load_word_embs()

        train_batcher, test_batcher = self.get_dataloaders(
            word_emb, graph_emb, self.suffix_prefix_buckets, cache_dir=Path(self.data_path).joinpath("__cache__")
        )

        # print(f"\n\n{params}")

        # for trial_ind in range(self.trials):
        #     trial_dir = self.get_trial_dir(trial_ind)
        trial_dir = self.get_training_dir()
        trial_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Running trial: {str(trial_dir)}")
        self._create_summary_writer(trial_dir)

        self.save_params(
            trial_dir, {
                "MODEL_PARAMS": self.model_params,
                "TRAINER_PARAMS": self.trainer_params,
                "model_class": self.model.__class__.__name__,
                "batcher_class": self.batcher.__class__.__name__
            }
        )

        model = self.get_model(
            tok_embedder=word_emb, graph_embedder=graph_emb, train_embeddings=self.finetune,
            suffix_prefix_buckets=self.suffix_prefix_buckets,
            num_classes=train_batcher.num_classes(how=self.classes_for), seq_len=self.seq_len, no_graph=self.no_graph,
            graph_padding_idx=train_batcher.graphpad,
            **self.model_params
        )

        def save_ckpt_fn():
            checkpoint_path = os.path.join(trial_dir, "checkpoint")
            self.save_checkpoint(model, checkpoint_path)

        train_scores, test_scores, train_average_scores, test_average_scores = self.train(
            model=model, train_batches=train_batcher, test_batches=test_batcher, epochs=self.epochs,
            learning_rate=self.learning_rate,
            scorer=lambda pred, true: entity_scorer(pred, true, train_batcher.tagmap,
                                                    no_localization=self.no_localization),
            learning_rate_decay=self.learning_rate_decay, finetune=self.finetune, save_ckpt_fn=save_ckpt_fn,
            no_localization=self.no_localization
        )

        # checkpoint_path = os.path.join(trial_dir, "checkpoint")
        # model.save_weights(checkpoint_path)

        metadata = {
            "train_scores": train_scores,
            "test_scores": test_scores,
            "train_average_scores": train_average_scores,
            "test_average_scores": test_average_scores,
        }

        # write_config(trial_dir, params, extra_params={"suffix_prefix_buckets": suffix_prefix_buckets, "seq_len": seq_len})

        with open(os.path.join(trial_dir, "train_data.json"), "w") as metadata_sink:
            metadata_sink.write(json.dumps(metadata, indent=4))

        pickle.dump(train_batcher.tagmap, open(os.path.join(trial_dir, "tag_types.pkl"), "wb"))


def get_type_prediction_arguments():
    parser = TypePredictorTrainerArguments()
    args = parser.parse()

    if args.finetune is False and args.pretraining_epochs > 0:
        logging.info(
            f"Fine-tuning is disabled, but the the number of pretraining epochs is {args.pretraining_epochs}. Setting pretraining epochs to 0.")
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

    dataset_dir = Path(args.data_path)
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
