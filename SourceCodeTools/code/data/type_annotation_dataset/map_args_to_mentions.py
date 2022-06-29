import argparse
import json
from os.path import join

import pandas as pd
from tqdm import tqdm

from SourceCodeTools.code.common import read_nodes, read_edges


def map_args_to_mention(working_directory, output, dataset_file=None):
    if dataset_file is None:
        dataset_file = join(working_directory, "function_annotations.json")

    arguments = set()
    mentions = set()
    for nodes in tqdm(read_nodes(join(working_directory, "common_nodes.json"), as_chunks=True), desc="Collecting mentions"):
        arguments.update(nodes.query("type == 'arg'")["id"])
        mentions.update(nodes.query("type == 'mention'")["id"])

    scrutinize_edges = []

    for edges in tqdm(read_edges(join(working_directory, "common_edges.json"), as_chunks=True), desc="Collecting mentions"):
        edges = edges.query("(source_node_id in @mentions) and (target_node_id in @arguments)", local_dict={"arguments": arguments, "mentions": mentions})
        scrutinize_edges.append(edges)

    edges = pd.concat(scrutinize_edges)
    mapping = {}
    for src, dst in edges[["source_node_id", "target_node_id"]].values:
        mapping[dst] = src

    with open(output, "w") as sink:
        with open(dataset_file) as fa:
            for line in fa:
                entry = json.loads(line)
                new_repl = [[s, e, int(mapping.get(r, r))] for s, e, r in entry["replacements"]]
                entry["replacements"] = new_repl

                sink.write(f"{json.dumps(entry)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("working_directory")
    parser.add_argument("output")
    parser.add_argument("--dataset_file", default=None)
    args = parser.parse_args()

    map_args_to_mention(args.working_directory, args.output, args.dataset_file)
