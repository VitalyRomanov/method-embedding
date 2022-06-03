import argparse
import json
from ast import literal_eval
from collections import defaultdict
from functools import partial
from os.path import join

from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.code.annotator_utils import resolve_self_collisions2


def prepare_offsets(offsets_path):
    offsets = unpersist(offsets_path)

    offsets_dict = defaultdict(list)
    for file_id, start, end, node_id in offsets[["file_id", "start", "end", "node_id"]].values:
        offsets_dict[file_id].append((int(start), int(end), int(node_id)))

    return offsets_dict


def create_text_dataset(dataset_path):
    get_path = partial(join, dataset_path)

    filecontent = unpersist(get_path("common_filecontent.json.bz2"))
    # partition = unpersist(get_path("common_filecontentpartition.json.bz2"))

    offsets = prepare_offsets(get_path("common_offsets.json.bz2"))

    train_set = open(get_path("var_misuse_seq_train.json"), "w")
    test_set = open(get_path("var_misuse_seq_test.json"), "w")
    val_set = open(get_path("var_misuse_seq_val.json"), "w")

    for file_id, filecontent, original_function, partition, misuse_span, label in filecontent[["id", "filecontent", "original_function", "partition", "misuse_span", "label"]].values:
        if partition == "train":
            dst = train_set
        elif partition == "eval":
            dst = test_set
        elif partition == "dev":
            dst = val_set
        else:
            raise ValueError()

        # if label == "Variable misuse":
        #     misuse_spans = [misuse_span]
        # else:
        #     misuse_spans = []
        assert label == "Variable misuse"
        misuse_span = literal_eval(misuse_span)
        misuse_spans = [(misuse_span[0], misuse_span[1], "misuse")]
        resolved_offsets = resolve_self_collisions2(offsets[file_id])

        entry = {
            "text": filecontent,
            "replacements": [],  # resolved_offsets,
            "entities": misuse_spans,
        }
        dst.write(f"{json.dumps(entry)}\n")

        entry["text"] = original_function
        entry["entities"] = []
        dst.write(f"{json.dumps(entry)}\n")

    train_set.close()
    test_set.close()
    val_set.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    args = parser.parse_args()
    create_text_dataset(args.dataset_path)
