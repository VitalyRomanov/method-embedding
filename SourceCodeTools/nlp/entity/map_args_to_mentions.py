import argparse
import json
import pickle
from os.path import join

from SourceCodeTools.code.data.sourcetrail.Dataset import load_data
from SourceCodeTools.code.data.sourcetrail.file_utils import unpersist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("working_directory")
    parser.add_argument("output")
    args = parser.parse_args()

    nodes, edges = load_data(join(args.working_directory, "nodes.bz2"), join(args.working_directory, "edges.bz2"))
    type_annotated = set(unpersist(join(args.working_directory, "type_annotations.bz2"))["src"].tolist())
    arguments = set(nodes.query("type == 'arg'")["id"].tolist())
    mentions = set(nodes.query("type == 'mention'")["id"].tolist())

    edges["in_mentions"] = edges["src"].apply(lambda src: src in mentions)

    edges["in_args"] = edges["dst"].apply(lambda dst: dst in arguments)

    edges = edges.query("in_mentions == True and in_args == True")

    mapping = {}
    for src, dst in edges[["src", "dst"]].values:
        if dst in mapping:
            print()
        mapping[dst] = src

    with open(args.output, "w") as sink:
        with open(join(args.working_directory, "function_annotations.jsonl")) as fa:
            for line in fa:
                entry = json.loads(line)
                new_repl = [[s, e, int(mapping.get(r, r))] for s, e, r in entry["replacements"]]
                entry["replacements"] = new_repl

                sink.write(f"{json.dumps(entry)}\n")


    print()

    # pickle.dump(mapping, open(args.output, "wb"))


if __name__ == "__main__":
    main()
