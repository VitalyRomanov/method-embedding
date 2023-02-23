from SourceCodeTools.cli_arguments.AbstractArgumentParser import AbstractArgumentParser


class AstDatasetCreatorArgumentParser(AbstractArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _add_optional_arguments(self):
        self._parser.add_argument('--language', "-l", dest="language", default="python", help='')
        self._parser.add_argument('--bpe_tokenizer', '-bpe', dest='bpe_tokenizer', type=str, help='')
        self._parser.add_argument('--create_subword_instances', action='store_true', default=False, help="")
        self._parser.add_argument('--connect_subwords', action='store_true', default=False,
                                  help="Takes effect only when `create_subword_instances` is False")
        self._parser.add_argument('--only_with_annotations', action='store_true', default=False, help="")
        self._parser.add_argument('--do_extraction', action='store_true', default=False, help="")
        self._parser.add_argument('--visualize', action='store_true', default=False, help="")
        self._parser.add_argument('--track_offsets', action='store_true', default=False, help="")
        self._parser.add_argument('--recompute_l2g', action='store_true', default=False, help="")
        self._parser.add_argument('--remove_type_annotations', action='store_true', default=False, help="")
        self._parser.add_argument('--seed', type=int, default=None, help="")
        self._additional_arguments()

    def _add_positional_argument(self):
        self._parser.add_argument('source_code', help='Path to DataFrame csv.')
        self._parser.add_argument('output_directory', help='')

    def _additional_arguments(self):
        # make it easier to redefine some arguments
        self._parser.add_argument('--chunksize', default=10000, type=int,
                                  help='Chunksize for preparing dataset. Larger chunks are faster to process, but they take more memory.')
        self._parser.add_argument('--keep_frac', default=1.0, type=float, help="Fraction of the dataset to keep")
        self._parser.add_argument('--use_mention_instances', action='store_true', default=False, help="")
        self._parser.add_argument('--graph_format_version', default="v3.5", type=str,
                                  help="Possible options: v2.5 | v1.0_control_flow | v3.5 | v3.5_control_flow")


class DatasetCreatorArgumentParser(AstDatasetCreatorArgumentParser):
    _parser_description = 'Merge indexed environments into a single graph'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_positional_argument(self):
        self._parser.add_argument('indexed_environments', help='Path to environments indexed by sourcetrail')
        self._parser.add_argument('output_directory', help='')

    def additional_arguments(self):
        pass