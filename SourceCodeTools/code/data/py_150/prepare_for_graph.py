import pandas as pd
import ast
import json
from pathlib import Path
from nltk import RegexpTokenizer
from itertools import chain


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


class DatasetAdapter:
    replacements = {
        "async": "async_",
        "await": "await_"
    }

    fields = {"function", "info", "label"}
    supported_partitions = {"dev", "eval", "train"}

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

    @classmethod
    def fix_code_if_needed(cls, code):
        f = code.lstrip()
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

            tokens = []

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

    replacement_fns = {
        "function": fix_code_if_needed.__func__,
        "info": fix_info_if_needed.__func__
    }

    extra_fields = {
        "info": [("package", get_package.__func__)]
    }

    preferred_column_order = ["id", "package", "function", "info", "label", "partition"]

    @classmethod
    def sort_columns(cls, columns):
        return sorted(columns, key=cls.preferred_column_order.index)

    @classmethod
    def process_record(cls, record):
        new_record = {}
        for field, data in record.items():
            new_record[field] = data if field not in cls.replacement_fns else cls.replacement_fns[field](data)

            for new_field, new_field_fn in cls.extra_fields.get(field, []):
                new_record[new_field] = new_field_fn(data)

        return new_record

    @classmethod
    def stream_original_partition(cls, directory, partition):
        directory = Path(directory)
        assert partition in cls.supported_partitions, f"Only the partitions should be one of: {cls.supported_partitions}, but {partition} given"
        for file in directory.iterdir():
            if file.name.startswith(partition):
                with open(file) as p:
                    for line in p:
                        yield json.loads(line)

    @classmethod
    def stream_processed_partition(cls, directory, partition, add_partition=False):
        for record in cls.stream_original_partition(directory, partition):
            try:
                r = cls.process_record(record)
            except MemoryError:  # there are two functions that cause this error
                continue
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

    DatasetAdapter.process_dataset(args.dataset_path, args.output_path)