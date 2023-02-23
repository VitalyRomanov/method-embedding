from argparse import ArgumentParser


class AbstractArgumentParser():
    _parser = None
    _parser_description = ""

    def __init__(self):
        self._create_parser()
        self._add_positional_argument()
        self._add_optional_arguments()

    def _create_parser(self):
        self._parser = ArgumentParser(
            description=self._parser_description
        )

    def _add_positional_argument(self):
        pass

    def _add_optional_arguments(self):
        pass

    def _verify_arguments(self, args):
        return args

    def parse(self):
        return self._verify_arguments(self._parser.parse_args())