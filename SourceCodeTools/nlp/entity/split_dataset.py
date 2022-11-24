import pickle
from collections import Counter

from scripts.training.type_prediction.cnn_type_prediction import get_type_prediction_arguments
from SourceCodeTools.nlp.entity.utils.data import read_data


def get_all_annotations(dataset):
    ann = []
    for _, annotations in dataset:
        for _, _, e in annotations["entities"]:
            ann.append(e)
    return ann


if __name__ == "__main__":

    args = get_type_prediction_arguments()

    train_data, test_data = read_data(
            open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True, include_only="entities",
            min_entity_count=args.min_entity_count, random_seed=args.random_seed
        )

    pickle.dump(train_data, open("type_prediction_dataset_no_defaults_train.pkl", "wb"))
    pickle.dump(test_data, open("type_prediction_dataset_no_defaults_test.pkl", "wb"))

    ent_counts = Counter(get_all_annotations(train_data)) | Counter(get_all_annotations(test_data))

    with open("type_prediction_dataset_argument_annotations_counts.txt", "w") as sink:
        for ent, count in ent_counts.most_common():
            sink.write(f"{ent}\t{count}\n")

    print()