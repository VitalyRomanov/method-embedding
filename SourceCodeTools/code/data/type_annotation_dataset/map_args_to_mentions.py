import argparse
import json
from os.path import join

import pandas as pd

from SourceCodeTools.code.common import read_nodes
from SourceCodeTools.code.data.dataset.Dataset import load_data
from SourceCodeTools.code.data.file_utils import unpersist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("working_directory")
    parser.add_argument("output")
    parser.add_argument("--dataset_file", default=None)
    args = parser.parse_args()

    if args.dataset_file is None:
        args.dataset_file = join(args.working_directory, "function_annotations.jsonl")

    # nodes, edges = load_data(join(args.working_directory, "common_nodes.json.bz2"), join(args.working_directory, "common_edges.json.bz2"))

    arguments = set()
    mentions = set()
    for nodes in read_nodes(join(args.working_directory, "common_nodes.json.bz2"), as_chunks=True):
        arguments.update(nodes.query("type == 'arg'")["id"])
        mentions.update(nodes.query("type == 'mention'")["id"])

    # type_annotated = set(unpersist(join(args.working_directory, "type_annotations.json.bz2"))["src"].tolist())

    scrutinize_edges = []

    for edges in read_nodes(join(args.working_directory, "common_edges.json.bz2"), as_chunks=True):
        edges = edges.query("(source_node_id in @mentions) and (target_node_id in @arguments)", local_dict={"arguments": arguments, "mentions": mentions})
        scrutinize_edges.append(edges)

    edges = pd.concat(scrutinize_edges)
    mapping = {}
    for src, dst in edges[["source_node_id", "target_node_id"]].values:
        if dst in mapping:
            print()
        mapping[dst] = src

    with open(args.output, "w") as sink:
        with open(args.dataset_file) as fa:
            for line in fa:
                entry = json.loads(line)
                new_repl = [[s, e, int(mapping.get(r, r))] for s, e, r in entry["replacements"]]
                entry["replacements"] = new_repl

                sink.write(f"{json.dumps(entry)}\n")


    print()

    # pickle.dump(mapping, open(args.output, "wb"))


if __name__ == "__main__":
    main()
