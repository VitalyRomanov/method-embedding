from SourceCodeTools.cli_arguments.AbstractArgumentParser import AbstractArgumentParser


class GraphTrainerArgumentParser(AbstractArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _dataset_arguments(self):
        self._parser.add_argument('--data_path', '-d', dest='data_path', default=None, help='Path to folder with dataset')
        self._parser.add_argument('--partition', dest='partition', default=None, help='')
        self._parser.add_argument('--train_frac', dest='train_frac', default=0.9, type=float, help='Fraction of nodes to be used for training')
        self._parser.add_argument('--filter_edges', dest='filter_edges', nargs='+', default=None, help='Space separated list of edges that should be filtered from the graph')
        self._parser.add_argument('--min_count_for_objectives', dest='min_count_for_objectives', default=5, type=int, help='Filter all target examples that occurr less than set numbers of times')
        self._parser.add_argument('--self_loops', action='store_true', help='Add self loops to the graph')
        self._parser.add_argument('--use_node_types', action='store_true', help='Add node types to the graph')
        self._parser.add_argument('--use_edge_types', action='store_true', help='Add edge types to the graph')
        self._parser.add_argument('--restore_state', action='store_true', help='Load from checkpoint')
        self._parser.add_argument('--no_global_edges', action='store_true', help='Remove all global edges from the graph')
        self._parser.add_argument('--remove_reverse', action='store_true', help="Remove reverse edges from the graph")
        self._parser.add_argument('--custom_reverse', dest='custom_reverse', nargs='+', default=None, help='List of edges for which to add reverse types. Should use together with `remove_reverse`')
        self._parser.add_argument('--restricted_id_pool', dest='restricted_id_pool', default=None, help='???')

    def _add_pretraining_arguments(self):
        self._parser.add_argument('--pretrained', '-p', dest='pretrained', default=None, help='Path to pretrained subtoken vectors')
        self._parser.add_argument('--tokenizer_path', '-t', dest='tokenizer_path', default=None, help='???')
        self._parser.add_argument('--pretraining_phase', dest='pretraining_phase', default=0, type=int, help='Number of epochs for pretraining')


    def _add_training_arguments(self):
        self._parser.add_argument('--model', dest='model', default="RGCN", help='')
        self._parser.add_argument('--embedding_table_size', dest='embedding_table_size', default=200000, type=int, help='Bucket size for the embedding table. Overriden when pretrained vectors provided???')
        self._parser.add_argument('--random_seed', dest='random_seed', default=None, type=int, help='Random seed for generating dataset splits')

        self._parser.add_argument('--node_emb_size', dest='node_emb_size', default=100, type=int, help='Dimensionality of node embeddings')
        self._parser.add_argument('--elem_emb_size', dest='elem_emb_size', default=100, type=int, help='Dimensionality of target embeddings (node names). Should match node embeddings when cosine distance loss is used')
        self._parser.add_argument('--sampling_neighbourhood_size', dest='sampling_neighbourhood_size', default=10, type=int, help='Number of dependencies to sample per node')
        self._parser.add_argument('--neg_sampling_factor', dest='neg_sampling_factor', default=3, type=int, help='Number of negative samples for each positive')

        self._parser.add_argument('--use_layer_scheduling', action='store_true', help='???')
        self._parser.add_argument('--schedule_layers_every', dest='schedule_layers_every', default=10, type=int, help='???')

        self._parser.add_argument('--epochs', dest='epochs', default=100, type=int, help='Number of epochs')
        self._parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size')

        self._parser.add_argument("--h_dim", dest="h_dim", default=None, type=int, help='Should be the same as `node_emb_size`')
        self._parser.add_argument("--n_layers", dest="n_layers", default=5, type=int, help='Number of layers')
        self._parser.add_argument("--objectives", dest="objectives", nargs='+', default=None, type=str, help='???')

        self._parser.add_argument("--skip_final_eval", action="store_true", help='')
        self._parser.add_argument("--save_each_epoch", action="store_true", help='Save checkpoints for each epoch (high disk space utilization)')
        self._parser.add_argument("--early_stopping", action="store_true", help='???')
        self._parser.add_argument("--early_stopping_tolerance", default=20, type=int, help='???')
        self._parser.add_argument("--force_w2v_ns", action="store_true", help='Use w2v negative sampling strategy p_unigram^(3/4)')
        self._parser.add_argument("--use_ns_groups", action="store_true", help='Perform negative sampling only from closest neighbours???')

        self._parser.add_argument("--metric", default="inner_prod", type=str, help='???')
        self._parser.add_argument("--nn_index", default="brute", type=str, help='Index backend for generating negative samples???')

        self._parser.add_argument("--external_dataset", default=None, type=str, help='Path to external graph, use for inference')

        self._parser.add_argument("--learning_rate", default=None, type=float, help='')

        self._parser.add_argument("--inference_ids_path", default=None, type=str, help='')

        self._parser.add_argument('--gpu', dest='gpu', default=-1, type=int, help='')

    def _add_scoring_arguments(self):
        self._parser.add_argument('--measure_scores', action='store_true')
        self._parser.add_argument('--dilate_scores', dest='dilate_scores', default=200, type=int, help='')

    def _add_performance_arguments(self):
        self._parser.add_argument('--no_checkpoints', dest="save_checkpoints", action='store_false')

        self._parser.add_argument('--use_gcn_checkpoint', action='store_true')
        self._parser.add_argument('--use_att_checkpoint', action='store_true')
        self._parser.add_argument('--use_gru_checkpoint', action='store_true')

    def _trainer_arguments(self):
        self._add_pretraining_arguments()
        self._add_training_arguments()
        self._add_scoring_arguments()
        self._add_performance_arguments()

    def _additional_arguments(self):
        pass

    def _add_optional_arguments(self):
        self._dataset_arguments()
        self._trainer_arguments()
        self._additional_arguments()

    def _add_positional_argument(self):
        self._parser.add_argument('model_output_dir', help='Location of the final model')


