import json


def get_node_ids_from_dataset(dataset_path):
    node_ids = []
    with open(dataset_path, "r") as dataset:
        for line in dataset:
            entry = json.loads(line)
            for _, _, id_ in entry["replacements"]:
                node_ids.append(int(id_))
    return node_ids


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("output")
    args = parser.parse_args()

    node_ids = get_node_ids_from_dataset(args.dataset)
    with open(args.output, "w") as sink:
        sink.write("node_id\n")
        for id_ in node_ids:
            sink.write(f"{id_}\n")


if __name__ == "__main__":
    main()