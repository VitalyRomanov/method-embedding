import hashlib

import pandas as pd
import ast
import json
from pathlib import Path
from nltk import RegexpTokenizer
from itertools import chain

from tqdm import tqdm

from SourceCodeTools.code.data.cubert_python_benchmarks.SQLTable import SQLTable


class CodeTokenizer:
    tok = RegexpTokenizer("\w+|\s+|\W+")

    @classmethod
    def tokenize(cls, code):
        return cls.tok.tokenize(code)

    @classmethod
    def detokenize(cls, code_tokens):
        return "".join(code_tokens)


def test_CodeTokenizer():
    code = """
data = pd.DataFrame2.from_records(json.loads(line) for line in open(filename).readlines())

success = 0
errors = 0
"""
    assert CodeTokenizer.detokenize(CodeTokenizer.tokenize(code)) == code


test_CodeTokenizer()


class Info:
    """
    Example
    dataset/ETHPy150Open EricssonResearch/calvin-base/calvin/requests/request_handler.py RequestHandler.get_index/VarMisuse@32/36 `async`->`self`
    """
    def __init__(self, info):
        self.info = info
        parts = info.split(" ")
        self.parse_filepath(parts)
        self.parse_fn_path(parts)
        self.parse_state(parts)
        self.parse_package(parts)
        self.set_id()

    def parse_filepath(self, parts):
        self.filepath = " ".join(self.info.split(".py")[0].split(" ")[1:]) + ".py"
        # self.filepath = parts[1].split(".py")[0] + ".py"

    def parse_fn_path(self, parts):
        self.fn_path = self.filepath + "/" + self.info.split(".py")[1].lstrip("/").lstrip(" ").split("/")[0]

    def parse_state(self, parts):
        self.state = "/".join(self.info.split(".py")[1].lstrip("/").lstrip(" ").split("/")[1:])

    def parse_package(self, parts):
        self.package = self.filepath.split("/")[0]

    def set_id(self):
        identifier = "\t".join([self.fn_path, self.state])
        self.id = hashlib.md5(identifier.encode('utf-8')).hexdigest()

        oidentifier = "\t".join([self.fn_path, "original"])

        if self.state.endswith("riginal"):
            assert identifier == oidentifier
        self.original_id = hashlib.md5(oidentifier.encode('utf-8')).hexdigest()

    def __repr__(self):
        return self.info


class DatasetAdapter:
    replacements = {
        "async": "async_",
        "await": "await_"
    }

    fields = {
        "variable_misuse": ["function", "label", "info"],
        "variable_misuse_repair": ["function", "target_mask", "error_location_mask", "candidate_mask", "provenance"],
        "exception": ["function", "label", "info"],
        "function_docstring": ["function", "docstring", "label", "info"],
        "swapped_operands": ["function", "label", "info"],
        "wrong_binary_operator": ["function", "label", "info"]
    }
    supported_partitions = ["train", "dev", "eval"] #

    benchmark_names = {
        "exception": "exception_datasets",
        "function_docstring": "function_docstring_datasets",
        "swapped_operands": "swapped_operands_datasets",
        "variable_misuse": "variable_misuse_dataset",
        "variable_misuse_repair": "variable_misuse_repair_datasets",
        "wrong_binary_operator": "wrong_binary_operator_datasets"
    }

    preferred_column_order = ["id", "package", "function", "info", "label", "partition"]

    import_order = [
        "variable_misuse",
        # "variable_misuse_repair",
        "exception",
        "function_docstring",
        "swapped_operands",
        "wrong_binary_operator"
    ]

    # expected_directory_structure = "dev.jsontxt-00000-of-00004\n"
    #                                 "dev.jsontxt-00001-of-00004\n"
    #                                 "dev.jsontxt-00002-of-00004\n"
    #                                 "dev.jsontxt-00003-of-00004\n"
    #                                 "eval.jsontxt-00000-of-00004\n"
    #                                 "eval.jsontxt-00001-of-00004\n"
    #                                 "eval.jsontxt-00002-of-00004\n"
    #                                 "eval.jsontxt-00003-of-00004\n"
    #                                 "train.jsontxt-00000-of-00004\n"
    #                                 "train.jsontxt-00001-of-00004\n"
    #                                 "train.jsontxt-00002-of-00004\n"
    #                                 "train.jsontxt-00003-of-00004\n"

    def __init__(self, dataset_location):
        self.dataset_location = Path(dataset_location)

        self.replacement_fns = {
            "function": self.fix_code_if_needed,
            "info": self.fix_info_if_needed
        }

        self.preprocess = {
            # "variable_misuse_repair": {
            #     "function": self.cubert_detokenize
            # }
        }

        self.extra_fields = {
            "info": [("package", self.get_package)],
            "provenance": [("info", self.fix_info_if_needed)]
        }

        self.db = SQLTable(self.dataset_location.joinpath("cubert_benchmarks.db"))

    def iterate_dataset(self, dataset_name):
        for partition in self.supported_partitions:
            for record in self.stream_processed_partition(
                    self.dataset_location.joinpath(self.benchmark_names[dataset_name]), partition,
                    preprocess_fns=self.preprocess.get(dataset_name, None)
            ):
                record["partition"] = partition
                yield record

    def load_original_functions(self):
        functions = self.db.query("SELECT DISTINCT original_id, function FROM functions where comment = 'original' AND dataset = 'variable_misuse'")
        self.original_functions = dict(zip(functions["original_id"], functions["function"]))

    def prepare_misuse_repair_record(self, record):
        print(record)

    @staticmethod
    def get_source_from_ast_range(node, function, strip=True):
        lines = function.split("\n")
        start_line = node.lineno
        end_line = node.end_lineno
        start_col = node.col_offset
        end_col = node.end_col_offset

        source = ""
        num_lines = end_line - start_line + 1
        if start_line == end_line:
            section = lines[start_line - 1].encode("utf8")[start_col:end_col].decode(
                "utf8")
            source += section.strip() if strip else section + "\n"
        else:
            for ind, lineno in enumerate(range(start_line - 1, end_line)):
                if ind == 0:
                    section = lines[lineno].encode("utf8")[start_col:].decode(
                        "utf8")
                    source += section.strip() if strip else section + "\n"
                elif ind == num_lines - 1:
                    section = lines[lineno].encode("utf8")[:end_col].decode(
                        "utf8")
                    source += section.strip() if strip else section + "\n"
                else:
                    section = lines[lineno]
                    source += section.strip() if strip else section + "\n"

        return source.rstrip()

    def get_dispatch(self, function):
        root = ast.parse(function)
        return self.get_source_from_ast_range(root.body[0].decorator_list[0], function)

    def import_data(self):

        functions = []

        # added_original = set()

        for dataset_name in self.import_order:

            parsed_successfully = 0
            parsed_with_errors = 0

            dataset_file = open(self.dataset_location.joinpath(f"{dataset_name}.jsonl"), "w")

            for record in tqdm(self.iterate_dataset(dataset_name), desc=f"Processing {dataset_name}"):
                info = Info(record["info"])

                # if info.id != "12101e677d30de9462596e7894d2bbd1":
                #     continue

                # if dataset_name == "variable_misuse_repair":
                #     record = self.prepare_misuse_repair_record(record)
                # if record["function"].startswith("@dispatch"):
                #     dispatch = self.get_dispatch(record["function"])
                #     info.fn_path += f"@{dispatch}"
                #     info.set_id()

                record_for_writing = {
                    # "id": info.id,
                    # "original_id": info.original_id,
                    # "filepath": info.filepath,
                    "fn_path": info.fn_path,
                    "function": record["function"],
                    "comment": info.state,
                    "package": info.package,
                    "label": record["label"],
                    # "dataset": dataset_name,
                    "partition": record["partition"],
                    "parsing_error": record["parsing_error"]
                }

                if record["parsing_error"] is None:
                    parsed_successfully += 1
                    dataset_file.write(f"{json.dumps(record_for_writing)}\n")
                else:
                    parsed_with_errors += 1

            dataset_file.close()
            print(f"{dataset_name}: success {parsed_successfully} error {parsed_with_errors}")
            #     functions.append(record_for_writing)
            #
            #     if len(functions) > 100000:
            #         self.db.add_records(pd.DataFrame.from_records(functions), "functions")
            #         functions.clear()
            #
            # if len(functions) > 0:
            #     self.db.add_records(pd.DataFrame.from_records(functions), "functions")
            #     functions.clear()

    @staticmethod
    def remove_indent(code):
        lines = code.strip("\n").split("\n")
        first_line_indent = lines[0][:len(lines[0]) - len(lines[0].lstrip())]
        start_char = len(first_line_indent)
        if start_char != 0:
            clean = "\n".join(line[start_char:] if line.startswith(first_line_indent) else line for line in lines)
        else:
            if lines[0].lstrip().startswith("@"):
                for ind, line in enumerate(lines):
                    stripped = line.lstrip()
                    if stripped.startswith("def "):
                        lines[ind] = stripped
                        break
            clean = "\n".join(lines)
        return clean

    @classmethod
    def fix_code_if_needed(cls, code):
        # f = code.lstrip()
        f = cls.remove_indent(code)
        try:
            ast.parse(f)
        except Exception as e:
            tokens = CodeTokenizer.tokenize(f)
            recovered_tokens = [cls.replacements[token] if token in cls.replacements else token for token in tokens]
            f = CodeTokenizer.detokenize(recovered_tokens)
            ast.parse(f)
        return f

    @classmethod
    def fix_info_if_needed(cls, info):
        if not info.endswith("original"):
            parts = info.split(" ")
            variable_replacement = parts[-1]

            variable_replacement = CodeTokenizer.detokenize(
                cls.replacements[token] if token in cls.replacements else token
                for token in CodeTokenizer.tokenize(variable_replacement)
            )

            parts[-1] = variable_replacement

            info = " ".join(parts)
        return info

    @staticmethod
    def get_package(info):
        return info.split(" ")[1].split("/")[0]

    @classmethod
    def sort_columns(cls, columns):
        return sorted(columns, key=cls.preferred_column_order.index)

    def process_record(self, record, preprocess_fns):
        new_record = {}
        for field, data in record.items():
            if preprocess_fns is not None and field in preprocess_fns:
                record[field] = preprocess_fns[field](data)

            new_record[field] = data if field not in self.replacement_fns else self.replacement_fns[field](data)

            for new_field, new_field_fn in self.extra_fields.get(field, []):
                new_record[new_field] = new_field_fn(data)

        return new_record

    @classmethod
    def stream_original_partition(cls, directory, partition, *args, **kwargs):
        directory = Path(directory)
        assert partition in cls.supported_partitions, f"Only the partitions should be one of: {cls.supported_partitions}, but {partition} given"
        for file in directory.iterdir():
            if file.name.startswith(partition):
                with open(file) as p:
                    for line in p:
                        yield json.loads(line)

    def stream_processed_partition(self, directory, partition, add_partition=False, preprocess_fns=None):
        for record in self.stream_original_partition(directory, partition):
            try:
                r = self.process_record(record, preprocess_fns=preprocess_fns)
                r["parsing_error"] = None
            except Exception as e:
                r = record
                r["parsing_error"] = e.msg if hasattr(e, "msg") else e.__class__.__name__
                r["package"] = self.get_package(r["info"])
            # except MemoryError:  # there are two functions that cause this error
            #     continue
            # except SyntaxError as e:
            #     continue
            if add_partition:
                r["partition"] = partition
            yield r

    @classmethod
    def process_dataset(cls, original_data_location, output_location):
        partitions = ["train", "dev", "eval"]

        last_id = 0
        column_order = None

        data = pd.DataFrame.from_records(
            chain(
                *(cls.stream_processed_partition(original_data_location, partition, add_partition=True) for partition in
                  partitions)),
            # columns = column_order
        )
        data["id"] = range(len(data))
        # data.to_pickle(output_location, index=False, columns=cls.sort_columns(data.columns))
        data.to_pickle(output_location)


# class DatasetAdapter:
#     replacements = {
#         "async": "async_",
#         "await": "await_"
#     }
#
#     fields = {"function", "info", "label"}
#     supported_partitions = {"dev", "eval", "train"}
#
#     # expected_directory_structure = "dev.jsontxt-00000-of-00004\n"
#     #                                 "dev.jsontxt-00001-of-00004\n"
#     #                                 "dev.jsontxt-00002-of-00004\n"
#     #                                 "dev.jsontxt-00003-of-00004\n"
#     #                                 "eval.jsontxt-00000-of-00004\n"
#     #                                 "eval.jsontxt-00001-of-00004\n"
#     #                                 "eval.jsontxt-00002-of-00004\n"
#     #                                 "eval.jsontxt-00003-of-00004\n"
#     #                                 "train.jsontxt-00000-of-00004\n"
#     #                                 "train.jsontxt-00001-of-00004\n"
#     #                                 "train.jsontxt-00002-of-00004\n"
#     #                                 "train.jsontxt-00003-of-00004\n"
#
#     @classmethod
#     def fix_code_if_needed(cls, code):
#         f = code.lstrip()
#         try:
#             ast.parse(f)
#         except Exception as e:
#             tokens = CodeTokenizer.tokenize(f)
#             recovered_tokens = [cls.replacements[token] if token in cls.replacements else token for token in tokens]
#             f = CodeTokenizer.detokenize(recovered_tokens)
#             ast.parse(f)
#         return f
#
#     @classmethod
#     def fix_info_if_needed(cls, info):
#         if not info.endswith("original"):
#             parts = info.split(" ")
#             variable_replacement = parts[-1]
#
#             tokens = []
#
#             variable_replacement = CodeTokenizer.detokenize(
#                 cls.replacements[token] if token in cls.replacements else token
#                 for token in CodeTokenizer.tokenize(variable_replacement)
#             )
#
#             parts[-1] = variable_replacement
#
#             info = " ".join(parts)
#         return info
#
#     @staticmethod
#     def get_package(info):
#         return info.split(" ")[1].split("/")[0]
#
#     replacement_fns = {
#         "function": fix_code_if_needed.__func__,
#         "info": fix_info_if_needed.__func__
#     }
#
#     extra_fields = {
#         "info": [("package", get_package.__func__)]
#     }
#
#     preferred_column_order = ["id", "package", "function", "info", "label", "partition"]
#
#     @classmethod
#     def sort_columns(cls, columns):
#         return sorted(columns, key=cls.preferred_column_order.index)
#
#     @classmethod
#     def process_record(cls, record):
#         new_record = {}
#         for field, data in record.items():
#             new_record[field] = data if field not in cls.replacement_fns else cls.replacement_fns[field](data)
#
#             for new_field, new_field_fn in cls.extra_fields.get(field, []):
#                 new_record[new_field] = new_field_fn(data)
#
#         return new_record
#
#     @classmethod
#     def stream_original_partition(cls, directory, partition):
#         directory = Path(directory)
#         assert partition in cls.supported_partitions, f"Only the partitions should be one of: {cls.supported_partitions}, but {partition} given"
#         for file in directory.iterdir():
#             if file.name.startswith(partition):
#                 with open(file) as p:
#                     for line in p:
#                         yield json.loads(line)
#
#     @classmethod
#     def stream_processed_partition(cls, directory, partition, add_partition=False):
#         for record in cls.stream_original_partition(directory, partition):
#             try:
#                 r = cls.process_record(record)
#             except MemoryError:  # there are two functions that cause this error
#                 continue
#             if add_partition:
#                 r["partition"] = partition
#             yield r
#
#     @classmethod
#     def process_dataset(cls, original_data_location, output_location):
#         partitions = ["train", "dev", "eval"]
#
#         last_id = 0
#         column_order = None
#
#         data = pd.DataFrame.from_records(
#             chain(
#                 *(cls.stream_processed_partition(original_data_location, partition, add_partition=True) for partition in
#                   partitions)),
#             # columns = column_order
#         )
#         data["id"] = range(len(data))
#         # data.to_pickle(output_location, index=False, columns=cls.sort_columns(data.columns))
#         data.to_pickle(output_location)


def test_fix_info_if_needed():
    example_info = "dataset/ETHPy150Open EricssonResearch/calvin-base/calvin/requests/request_handler.py RequestHandler.get_index/VarMisuse@32/36 `async`->`self`"
    assert DatasetAdapter.fix_info_if_needed(
        example_info) == "dataset/ETHPy150Open EricssonResearch/calvin-base/calvin/requests/request_handler.py RequestHandler.get_index/VarMisuse@32/36 `async_`->`self`"

    example_info = "dataset/ETHPy150Open EricssonResearch/calvin-base/calvin/requests/request_handler.py RequestHandler.get_index/VarMisuse@32/36 `async_call`->`self`"
    assert DatasetAdapter.fix_info_if_needed(
        example_info) == "dataset/ETHPy150Open EricssonResearch/calvin-base/calvin/requests/request_handler.py RequestHandler.get_index/VarMisuse@32/36 `async_call`->`self`"


test_fix_info_if_needed()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Convert CuBERT's variable misuse detection dataset for further processing")
    parser.add_argument("dataset_path", help="Path to dataset folder")
    parser.add_argument("output_path", help="Path to output file")
    args = parser.parse_args()

    dataset = DatasetAdapter(args.dataset_path)
    dataset.import_data()
    # DatasetAdapter.process_dataset(args.dataset_path, args.output_path)