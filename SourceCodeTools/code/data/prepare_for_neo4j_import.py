import argparse
import os.path
from os.path import join

from tqdm import tqdm

from SourceCodeTools.code.common import read_nodes, read_edges
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import special_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes")
    parser.add_argument("edges")
    parser.add_argument("--remove_global_edges", action="store_true", default=False)

    args = parser.parse_args()

    nodes_dir = os.path.dirname(args.nodes)

    for ind, nodes in enumerate(tqdm(read_nodes(args.nodes, as_chunks=True))):
        nodes = nodes[["id", "type", "serialized_name", "mentioned_in"]]
        nodes.rename({
            "id": "id:ID",
            "serialized_name": "name",
            "type": ":LABEL",
            "mentioned_in": "mentioned_in:int"
        }, axis=1, inplace=True)

        if ind == 0:
            nodes.to_csv(join(nodes_dir, "nodes_n4j.csv"), index=False)
        else:
            nodes.to_csv(join(nodes_dir, "nodes_n4j.csv"), index=False, header=False, mode="a")

    edges_dir = os.path.dirname(args.edges)

    restricted_edge_types = set()
    if args.remove_global_edges:
        restricted_edge_types.update(special_mapping.keys())
        restricted_edge_types.update(special_mapping.values())

    for ind, edges in enumerate(tqdm(read_edges(args.edges, as_chunks=True))):
        edges = edges[["type", "source_node_id", "target_node_id", "mentioned_in", "file_id"]]
        edges.query("type not in @restricted", local_dict={"restricted": restricted_edge_types}, inplace=True)
        edges.rename({
            "source_node_id": ":START_ID",
            "target_node_id": ":END_ID",
            "type": ":TYPE",
            "mentioned_in": "mentioned_in:int",
            "file_id": "file_id:int"
        }, axis=1, inplace=True)

        if ind == 0:
            edges.to_csv(join(edges_dir, "edges_n4j.csv"), index=False)
        else:
            edges.to_csv(join(edges_dir, "edges_n4j.csv"), index=False, header=False, mode="a")


if __name__ == "__main__":
    main()