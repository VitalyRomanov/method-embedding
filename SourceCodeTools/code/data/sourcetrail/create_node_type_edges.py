from SourceCodeTools.code.common import read_nodes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes")
    parser.add_argument("output")

    args = parser.parse_args()

    for nodes in read_nodes(args.nodes, as_chunks=True):
        nodes["edge_type"] = "node_type"
        nodes[["id", "type", "edge_type"]] \
            .astype({"id": "string", "type": "string", "edge_type": "string"}) \
            .to_csv(args.output, index=False, header=False, mode="a", sep="\t")


if __name__ == "__main__":
    main()