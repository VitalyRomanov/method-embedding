import argparse
import json
import os
from collections import Counter

from SourceCodeTools.nlp.entity.utils.data import read_data


def get_all_annotations(dataset):
    ann = []
    for _, annotations in dataset:
        for _, _, e in annotations["entities"]:
            ann.append(e)
    return ann


def split_dataset(
        data_path, min_entity_count, random_seed, name_suffix
):
    train_data, test_data = read_data(
        open(data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True,
        include_only="entities",
        min_entity_count=min_entity_count, random_seed=random_seed
    )

    directory = os.path.dirname(data_path)

    def write_to_file(data, directory, suffix, partition):
        with open(os.path.join(directory, f"type_prediction_dataset_{suffix}_{partition}.json"), "w") as sink:
            for entry in data:
                sink.write(f"{json.dumps(entry)}\n")

    write_to_file(train_data, directory, name_suffix, "train")
    write_to_file(test_data, directory, name_suffix, "test")

    ent_counts = Counter(get_all_annotations(train_data)) | Counter(get_all_annotations(test_data))

    with open(os.path.join(directory, f"type_prediction_dataset_{name_suffix}_annotations_counts.txt"),
              "w") as sink:
        for ent, count in ent_counts.most_common():
            sink.write(f"{ent}\t{count}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default=None,
                        help='Path to the dataset file')
    parser.add_argument('--min_entity_count', dest='min_entity_count', default=3, type=int,
                        help='')
    parser.add_argument('--random_seed', dest='random_seed', default=None, type=int,
                        help='')
    parser.add_argument('--name_suffix', default="", type=str,
                        help='')

    args = parser.parse_args()

    split_dataset(args.data_path, args.min_entity_count, args.random_seed, args.name_suffix)
