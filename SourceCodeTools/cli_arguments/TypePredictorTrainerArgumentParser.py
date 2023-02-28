from SourceCodeTools.cli_arguments import default_config, update_config, load_config
from SourceCodeTools.cli_arguments.AbstractArgumentParser import AbstractArgumentParser


type_pred_config_specification = {
    "DATASET": {
        "data_path": None,
        "min_entity_count": 3,
        "restrict_allowed": False,
    },
    "TRAINING": {
        "mask_unlabeled_declarations": False,
        "no_localization": False,
        "word_emb_path": None,
        "graph_emb_path": None,
        "model_output": None,

        "learning_rate": 1e-3,
        "learning_rate_decay": 1.0,
        "dropout": None,
        "weight_decay": None,

        "batch_size": 32,
        "suffix_prefix_buckets": 3000,
        "max_seq_len": 256,

        "finetune": False,
        "pretraining_epochs": 0,
        "epochs": 500,
        "ckpt_path": None,

        "no_graph": False,

        "gpu": -1,

        "type_ann_edges": None,
        "bert_layer_freeze_start": None,
        "bert_layer_freeze_end": None,

        "model_class": None,
        "batcher_class": None
    },
    "MODEL": {
    },
}


def default_type_pred_config():
    return default_config(type_pred_config_specification)


def get_type_pred_config(**kwargs):
    config = default_type_pred_config()
    return update_config(config, **kwargs)


class TypePredictorTrainerArgumentParser(AbstractArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _dataset_arguments(self):
        self._parser.add_argument('--data_path', dest='data_path', default=None, help='Path to the dataset file')
        self._parser.add_argument('--word_emb_path', dest='word_emb_path', default=None, help='Path to the file with token embeddings')
        self._parser.add_argument('--graph_emb_path', dest='graph_emb_path', default=None, help='Path to the file with graph embeddings')
        self._parser.add_argument('--min_entity_count', dest='min_entity_count', default=3, type=int, help='')
        self._parser.add_argument('--mask_unlabeled_declarations', action='store_true')
        self._parser.add_argument('--no_localization', action='store_true')
        self._parser.add_argument('--restrict_allowed', action='store_true', default=False)

    def _trainer_arguments(self):
        self._parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, type=float, help='')
        self._parser.add_argument('--learning_rate_decay', dest='learning_rate_decay', default=1.0, type=float, help='')
        self._parser.add_argument('--batch_size', dest='batch_size', default=32, type=int, help='')
        self._parser.add_argument('--suffix_prefix_buckets', dest='suffix_prefix_buckets', default=3000, type=int, help='')
        self._parser.add_argument('--max_seq_len', dest='max_seq_len', default=100, type=int, help='')
        self._parser.add_argument('--pretraining_epochs', dest='pretraining_epochs', default=0, type=int, help='')
        self._parser.add_argument('--ckpt_path', dest='ckpt_path', default=None, type=str, help='')
        self._parser.add_argument('--epochs', dest='epochs', default=500, type=int, help='')
        self._parser.add_argument('--gpu', dest='gpu', default=-1, type=int, help='')
        self._parser.add_argument('--finetune', action='store_true')
        self._parser.add_argument('--no_graph', action='store_true', default=False)
        self._parser.add_argument('--dropout', dest='dropout', default=None, type=float, help='')
        self._parser.add_argument('--weight_decay', dest='weight_decay', default=None, type=float, help='')

    def _model_arguments(self):
        pass

    def _additional_arguments(self):
        pass

    def _add_optional_arguments(self):
        self._dataset_arguments()
        self._trainer_arguments()
        self._model_arguments()
        self._additional_arguments()

    def _add_positional_argument(self):
        self._parser.add_argument('model_output', help='')

    def _verify_arguments(self, args):
        import logging
        from os.path import isfile, isdir

        if args.finetune is False and args.pretraining_epochs > 0:
            logging.info(
                f"Fine-tuning is disabled, but the the number of pretraining epochs is {args.pretraining_epochs}. Setting pretraining epochs to 0.")
            args.pretraining_epochs = 0

        if args.graph_emb_path is not None and not (isfile(args.graph_emb_path) or isdir(args.graph_emb_path)):
            logging.warning(f"File with graph embeddings does not exist: {args.graph_emb_path}")
            args.graph_emb_path = None

        return args

    def _make_config(self, args):
        args_ = args.__dict__
        config_path = args_.pop("config", None)

        if config_path is None:
            config = get_type_pred_config(**args_)
        else:
            config = load_config(config_path)

        return config


class CodeBERTTypePredictorTrainerArgumentParser(TypePredictorTrainerArgumentParser):

    def _additional_arguments(self):
        self._parser.add_argument('--type_ann_edges', dest='type_ann_edges', default=None, help='Path to type annotation edges')
        self._parser.add_argument('--bert_layer_freeze_start', dest='bert_layer_freeze_start', default=None, type=int, help='')
        self._parser.add_argument('--bert_layer_freeze_end', dest='bert_layer_freeze_end', default=None, type=int, help='')


class CodeBertEmbeddingExtractorArgumentParser(TypePredictorTrainerArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _additional_arguments(self):
        self._parser.add_argument('--type_ann_edges', dest='type_ann_edges', default=None, help='Path to type annotation edges')
