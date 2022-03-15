import os.path

from SourceCodeTools.code.data.file_utils import unpersist, persist


def extract_partitions(path):
    filecontent = unpersist(path)

    items = filecontent[["id"]]
    masks = filecontent["partition"].values

    # create masks
    items["train_mask"] = masks == "train"
    items["val_mask"] = masks == "dev"
    items["test_mask"] = masks == "eval"

    dirname = os.path.dirname(path)
    persist(items, os.path.join(dirname, "partition.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filecontent")

    args = parser.parse_args()

    extract_partitions(args.filecontent)