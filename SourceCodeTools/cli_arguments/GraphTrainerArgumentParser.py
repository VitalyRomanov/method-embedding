from SourceCodeTools.cli_arguments.AbstractArgumentParser import AbstractArgumentParser
from SourceCodeTools.cli_arguments.config import load_config, default_config, update_config


graph_config_specification = {
    "DATASET": {
        "data_path": None,
        "train_frac": 0.9,
        "filter_edges": None,
        "min_count_for_objectives": 5,
        # "packages_file": None,  # partition file
        "self_loops": False,
        "use_node_types": False,
        "use_edge_types": False,
        "no_global_edges": False,
        "remove_reverse": False,
        "custom_reverse": None,
        "restricted_id_pool": None,
        "random_seed": None,
        "subgraph_id_column": "mentioned_in",
        "subgraph_partition": None,
        "partition": None,
        "max_type_ann_level": 3,
        "k_hops": 0
    },
    "TRAINING": {
        "model": "RGCN",
        "model_output_dir": None,
        "pretrained": None,
        "pretraining_phase": 0,

        "sampling_neighbourhood_size": 10,
        "neg_sampling_factor": 3,
        "use_layer_scheduling": False,
        "schedule_layers_every": 10,

        "elem_emb_size": 100,
        "embedding_table_size": 200000,

        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,

        "objectives": None,
        "save_each_epoch": False,
        "save_checkpoints": True,
        "early_stopping": False,
        "early_stopping_tolerance": 20,

        "force_w2v_ns": False,
        "use_ns_groups": False,
        "nn_index": "brute",

        "metric": "inner_prod",

        "measure_scores": False,
        "dilate_scores": 200,  # downsample

        "gpu": -1,

        "external_dataset": None,

        "restore_state": False,
        "skip_final_eval": False,

        "inference_ids_path": None
    },
    "MODEL": {
        "node_emb_size": 100,
        "h_dim": None,
        "n_layers": 5,
        "use_self_loop": False,

        "use_gcn_checkpoint": False,
        "use_att_checkpoint": False,
        "use_gru_checkpoint": False,

        'num_bases': 10,
        'dropout': 0.0,

        'activation': None,  #"leaky_relu",
        # torch.nn.functional.hardswish], #[torch.nn.functional.hardtanh], #torch.nn.functional.leaky_relu
    },
    "TOKENIZER": {
        "tokenizer_path": None,
    }
}


def default_graph_config():
    return default_config(graph_config_specification)


def get_graph_config(**kwargs):
    config = default_graph_config()
    return update_config(config, **kwargs)


class GraphTrainerArgumentParser(AbstractArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _dataset_arguments(self):
        group = self._parser.add_argument_group("DATASET")
        group.add_argument('--data_path', '-d', dest='data_path', default=None, help='Path to folder with dataset')
        group.add_argument('--partition', dest='partition', default=None, help='')
        group.add_argument('--train_frac', dest='train_frac', default=0.9, type=float, help='Fraction of nodes to be used for training')
        group.add_argument('--filter_edges', dest='filter_edges', nargs='+', default=None, help='Space separated list of edges that should be filtered from the graph')
        group.add_argument('--min_count_for_objectives', dest='min_count_for_objectives', default=5, type=int, help='Filter all target examples that occurr less than set numbers of times')
        group.add_argument('--self_loops', action='store_true', help='Add self loops to the graph')
        group.add_argument('--use_node_types', action='store_true', help='Add node types to the graph')
        group.add_argument('--use_edge_types', action='store_true', help='Add edge types to the graph')
        group.add_argument('--restore_state', action='store_true', help='Load from checkpoint')
        group.add_argument('--no_global_edges', action='store_true', help='Remove all global edges from the graph')
        group.add_argument('--remove_reverse', action='store_true', help="Remove reverse edges from the graph")
        group.add_argument('--custom_reverse', dest='custom_reverse', nargs='+', default=None, help='List of edges for which to add reverse types. Should use together with `remove_reverse`')
        group.add_argument('--restricted_id_pool', dest='restricted_id_pool', default=None, help='???')
        group.add_argument('--max_type_ann_level', default=3, type=int, help='')
        group.add_argument('--k_hops', default=0, type=int, help='')

    def _add_pretraining_arguments(self):
        group = self._parser.add_argument_group("PRETRAINING")
        group.add_argument('--pretrained', '-p', dest='pretrained', default=None, help='Path to pretrained subtoken vectors')
        group.add_argument('--tokenizer_path', '-t', dest='tokenizer_path', default=None, help='???')
        group.add_argument('--pretraining_phase', dest='pretraining_phase', default=0, type=int, help='Number of epochs for pretraining')

    def _add_model_arguments(self):
        group = self._parser.add_argument_group("MODEL")
        group.add_argument('--node_emb_size', dest='node_emb_size', default=100, type=int, help='Dimensionality of node embeddings')
        group.add_argument("--h_dim", dest="h_dim", default=None, type=int, help='Should be the same as `node_emb_size`')
        group.add_argument("--n_layers", dest="n_layers", default=5, type=int, help='Number of layers')
        group.add_argument('--use_gcn_checkpoint', action='store_true')
        group.add_argument('--use_att_checkpoint', action='store_true')
        group.add_argument('--use_gru_checkpoint', action='store_true')

    def _add_training_arguments(self):
        group = self._parser.add_argument_group("TRAINING")
        group.add_argument('--model', dest='model', default="RGCN", help='')
        group.add_argument('--embedding_table_size', dest='embedding_table_size', default=200000, type=int, help='Bucket size for the embedding table. Overriden when pretrained vectors provided???')
        group.add_argument('--elem_emb_size', dest='elem_emb_size', default=100, type=int, help='Dimensionality of target embeddings (node names). Should match node embeddings when cosine distance loss is used')
        # group.add_argument('--random_seed', dest='random_seed', default=None, type=int, help='Random seed for generating dataset splits')

        group.add_argument('--sampling_neighbourhood_size', dest='sampling_neighbourhood_size', default=10, type=int, help='Number of dependencies to sample per node')
        group.add_argument('--neg_sampling_factor', dest='neg_sampling_factor', default=3, type=int, help='Number of negative samples for each positive')

        # group.add_argument('--use_layer_scheduling', action='store_true', help='???')
        # group.add_argument('--schedule_layers_every', dest='schedule_layers_every', default=10, type=int, help='???')

        group.add_argument('--epochs', dest='epochs', default=100, type=int, help='Number of epochs')
        group.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size')

        group.add_argument("--objectives", dest="objectives", nargs='+', default=None, type=str, help='???')

        group.add_argument("--skip_final_eval", action="store_true", help='')
        group.add_argument("--save_each_epoch", action="store_true", help='Save checkpoints for each epoch (high disk space utilization)')
        group.add_argument("--early_stopping", action="store_true", help='???')
        group.add_argument("--early_stopping_tolerance", default=20, type=int, help='???')
        group.add_argument("--force_w2v_ns", action="store_true", help='Use w2v negative sampling strategy p_unigram^(3/4)')
        group.add_argument("--use_ns_groups", action="store_true", help='Perform negative sampling only from closest neighbours???')

        group.add_argument("--metric", default="inner_prod", type=str, help='???')
        group.add_argument("--nn_index", default="brute", type=str, help='Index backend for generating negative samples???')

        group.add_argument("--external_dataset", default=None, type=str, help='Path to external graph, use for inference')

        group.add_argument("--learning_rate", default=None, type=float, help='')
        group.add_argument("--weight_decay", default=None, type=float, help='')

        group.add_argument("--inference_ids_path", default=None, type=str, help='')

        group.add_argument('--gpu', dest='gpu', default=-1, type=int, help='')

    def _add_scoring_arguments(self):
        group = self._parser.add_argument_group("SCORING")
        group.add_argument('--measure_scores', action='store_true')
        group.add_argument('--dilate_scores', dest='dilate_scores', default=200, type=int, help='')

    def _add_performance_arguments(self):
        group = self._parser.add_argument_group("PERFORMANCE")
        group.add_argument('--no_checkpoints', dest="save_checkpoints", action='store_false')


    def _trainer_arguments(self):
        self._add_pretraining_arguments()
        self._add_training_arguments()
        self._add_scoring_arguments()
        self._add_performance_arguments()

    def _additional_arguments(self):
        pass

    def _add_optional_arguments(self):
        self._dataset_arguments()
        self._add_model_arguments()
        self._trainer_arguments()
        self._additional_arguments()
        self._parser.add_argument("--config", default=None)

    def _add_positional_argument(self):
        self._parser.add_argument('model_output_dir', help='Location of the final model')

    def _make_config(self, args):
        args_ = args.__dict__
        config_path = args_.pop("config", None)

        if config_path is None:
            config = get_graph_config(**args_)
        else:
            config = load_config(config_path)

        return config


