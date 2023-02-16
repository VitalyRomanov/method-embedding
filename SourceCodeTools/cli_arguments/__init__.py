import argparse


class Arguments:
    parser = None

    def __init__(self):
        super().__init__()
        self.parser = self.create_parser()
        self.add_positional_argument()
        self.additional_arguments()

    def create_parser(self):
        pass

    def add_positional_argument(self):
        pass

    def additional_arguments(self):
        pass

    def parse(self):
        return self.parser.parse_args()


class AstDatasetCreatorArguments(Arguments):
    def create_parser(self):
        parser = argparse.ArgumentParser(description='Merge indexed environments into a single graph')
        parser.add_argument('--language', "-l", dest="language", default="python", help='')
        parser.add_argument('--bpe_tokenizer', '-bpe', dest='bpe_tokenizer', type=str, help='')
        parser.add_argument('--create_subword_instances', action='store_true', default=False, help="")
        parser.add_argument('--connect_subwords', action='store_true', default=False,
                            help="Takes effect only when `create_subword_instances` is False")
        parser.add_argument('--only_with_annotations', action='store_true', default=False, help="")
        parser.add_argument('--do_extraction', action='store_true', default=False, help="")
        parser.add_argument('--visualize', action='store_true', default=False, help="")
        parser.add_argument('--track_offsets', action='store_true', default=False, help="")
        parser.add_argument('--recompute_l2g', action='store_true', default=False, help="")
        parser.add_argument('--remove_type_annotations', action='store_true', default=False, help="")
        parser.add_argument('--seed', type=int, default=None, help="")
        return parser

    def add_positional_argument(self):
        self.parser.add_argument(
            'source_code', help='Path to DataFrame csv.'
        )
        self.parser.add_argument('output_directory', help='')

    def additional_arguments(self):
        self.parser.add_argument('--chunksize', default=10000, type=int, help='Chunksize for preparing dataset. Larger chunks are faster to process, but they take more memory.')
        self.parser.add_argument('--keep_frac', default=1.0, type=float, help="Fraction of the dataset to keep")
        self.parser.add_argument('--use_mention_instances', action='store_true', default=False, help="")
        self.parser.add_argument('--graph_format_version', default="v3.5", type=str, help="Possible options: v2.5 | v1.0_control_flow | v3.5 | v3.5_control_flow")

    def parse(self):
        return self.parser.parse_args()


class DatasetCreatorArguments(AstDatasetCreatorArguments):

    def add_positional_argument(self):
        self.parser.add_argument('indexed_environments', help='Path to environments indexed by sourcetrail')
        self.parser.add_argument('output_directory', help='')

    def additional_arguments(self):
        pass


class TypePredictorTrainerArguments(Arguments):
    def create_parser(self):
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
        parser.add_argument('--suffix_prefix_buckets', dest='suffix_prefix_buckets', default=3000, type=int,
                            help='')
        parser.add_argument('--max_seq_len', dest='max_seq_len', default=100, type=int,
                            help='')
        parser.add_argument('--gpu', dest='gpu', default=-1, type=int,
                            help='')
        parser.add_argument('--min_entity_count', dest='min_entity_count', default=3, type=int,
                            help='')
        parser.add_argument('--pretraining_epochs', dest='pretraining_epochs', default=0, type=int,
                            help='')
        parser.add_argument('--ckpt_path', dest='ckpt_path', default=None, type=str,
                            help='')
        parser.add_argument('--epochs', dest='epochs', default=500, type=int,
                            help='')
        # parser.add_argument('--trials', dest='trials', default=1, type=int,
        #                     help='')
        parser.add_argument('--finetune', action='store_true')
        parser.add_argument('--mask_unlabeled_declarations', action='store_true')
        parser.add_argument('--no_localization', action='store_true')
        parser.add_argument('--restrict_allowed', action='store_true', default=False)
        parser.add_argument('--no_graph', action='store_true', default=False)
        return parser

    def add_positional_argument(self):
        self.parser.add_argument('model_output', help='')





