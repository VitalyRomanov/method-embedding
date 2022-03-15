def add_data_arguments(parser):
    parser.add_argument('--data_path', '-d', dest='data_path', default=None, help='Path to folder with dataset')
    parser.add_argument('--train_frac', dest='train_frac', default=0.9, type=float, help='Fraction of nodes to be used for training')
    parser.add_argument('--filter_edges', dest='filter_edges', default=None, help='Comma separated list of edges that should be filtered from the graph')
    parser.add_argument('--min_count_for_objectives', dest='min_count_for_objectives', default=5, type=int, help='Filter all target examples that occurr less than set numbers of times')
    # parser.add_argument('--packages_file', dest='packages_file', default=None, type=str, help='???')
    parser.add_argument('--self_loops', action='store_true', help='Add self loops to the graph')
    parser.add_argument('--use_node_types', action='store_true', help='Add node types to the graph')
    parser.add_argument('--use_edge_types', action='store_true', help='Add edge types to the graph')
    parser.add_argument('--restore_state', action='store_true', help='Load from checkpoint')
    parser.add_argument('--no_global_edges', action='store_true', help='Remove all global edges from the graph')
    parser.add_argument('--remove_reverse', action='store_true', help="Remove reverse edges from the graph")
    parser.add_argument('--custom_reverse', dest='custom_reverse', default=None, help='List of edges for which to add reverse types. Should use together with `remove_reverse`')
    parser.add_argument('--restricted_id_pool', dest='restricted_id_pool', default=None, help='???')
    parser.add_argument('--subgraph_partition', default=None)
    parser.add_argument('--subgraph_id_column', default=None)


def add_pretraining_arguments(parser):
    parser.add_argument('--pretrained', '-p', dest='pretrained', default=None, help='Path to pretrained subtoken vectors')
    parser.add_argument('--tokenizer_path', '-t', dest='tokenizer_path', default=None, help='???')
    parser.add_argument('--pretraining_phase', dest='pretraining_phase', default=0, type=int, help='Number of epochs for pretraining')


def add_training_arguments(parser):
    parser.add_argument('--embedding_table_size', dest='embedding_table_size', default=200000, type=int, help='Bucket size for the embedding table. Overriden when pretrained vectors provided???')
    parser.add_argument('--random_seed', dest='random_seed', default=None, type=int, help='Random seed for generating dataset splits')

    parser.add_argument('--node_emb_size', dest='node_emb_size', default=100, type=int, help='Dimensionality of node embeddings')
    parser.add_argument('--elem_emb_size', dest='elem_emb_size', default=100, type=int, help='Dimensionality of target embeddings (node names). Should match node embeddings when cosine distance loss is used')
    parser.add_argument('--sampling_neighbourhood_size', dest='sampling_neighbourhood_size', default=10, type=int, help='Number of dependencies to sample per node')
    parser.add_argument('--neg_sampling_factor', dest='neg_sampling_factor', default=3, type=int, help='Number of negative samples for each positive')

    parser.add_argument('--use_layer_scheduling', action='store_true', help='???')
    parser.add_argument('--schedule_layers_every', dest='schedule_layers_every', default=10, type=int, help='???')

    parser.add_argument('--epochs', dest='epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', dest='batch_size', default=128, type=int, help='Batch size')

    parser.add_argument("--h_dim", dest="h_dim", default=None, type=int, help='Should be the same as `node_emb_size`')
    parser.add_argument("--n_layers", dest="n_layers", default=5, type=int, help='Number of layers')
    parser.add_argument("--objectives", dest="objectives", default=None, type=str, help='???')

    parser.add_argument("--save_each_epoch", action="store_true", help='Save checkpoints for each epoch (high disk space utilization)')
    parser.add_argument("--early_stopping", action="store_true", help='???')
    parser.add_argument("--early_stopping_tolerance", default=20, type=int, help='???')
    parser.add_argument("--force_w2v_ns", action="store_true", help='Use w2v negative sampling strategy p_unigram^(3/4)')
    parser.add_argument("--use_ns_groups", action="store_true", help='Perform negative sampling only from closest neighbours???')

    parser.add_argument("--metric", default="inner_prod", type=str, help='???')
    parser.add_argument("--nn_index", default="brute", type=str, help='Index backend for generating negative samples???')

    parser.add_argument("--external_dataset", default=None, type=str, help='Path to external graph, use for inference')


def add_scoring_arguments(parser):
    parser.add_argument('--measure_scores', action='store_true')
    parser.add_argument('--dilate_scores', dest='dilate_scores', default=200, type=int, help='')


def add_performance_arguments(parser):
    parser.add_argument('--no_checkpoints', dest="save_checkpoints", action='store_false')

    parser.add_argument('--use_gcn_checkpoint', action='store_true')
    parser.add_argument('--use_att_checkpoint', action='store_true')
    parser.add_argument('--use_gru_checkpoint', action='store_true')


def add_gnn_train_args(parser):
    parser.add_argument("--config", default=None, help="Path to config file")
    add_data_arguments(parser)
    add_pretraining_arguments(parser)
    add_training_arguments(parser)
    add_scoring_arguments(parser)
    add_performance_arguments(parser)

    # parser.add_argument('--note', dest='note', default="", help='Note, added to metadata')
    parser.add_argument('model_output_dir', help='Location of the final model')

    # parser.add_argument('--intermediate_supervision', action='store_true')
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int, help='')


def verify_arguments(args):
    pass