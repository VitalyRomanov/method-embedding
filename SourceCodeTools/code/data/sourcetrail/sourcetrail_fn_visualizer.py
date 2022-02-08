from os.path import join

from SourceCodeTools.code.data.dataset.Dataset import load_data
from SourceCodeTools.code.data.sourcetrail.sourcetrail_draw_graph import visualize


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("working_directory")
    parser.add_argument("function_id", type=int)
    parser.add_argument("output_path")

    args = parser.parse_args()

    nodes_path = join(args.working_directory, "common_nodes.bz2")
    edges_path = join(args.working_directory, "common_edges.bz2")

    nodes, edges = load_data(nodes_path, edges_path, rename_columns=False)

    fn_edges = edges.query(f"mentioned_in == {args.function_id}")
    fn_node_ids = set(fn_edges["source_node_id"] + fn_edges["target_node_id"])
    registered_edges = set(fn_edges["id"])
    extra_edges = edges[
        (edges["source_node_id"].apply(lambda id_: id_ in fn_node_ids) |
        edges["target_node_id"].apply(lambda id_: id_ in fn_node_ids) ) #&
        # edges["id"].apply(lambda id_: id_ not in registered_edges)
    ]
    fn_nodes = nodes[
        nodes["id"].apply(lambda id_: id_ in fn_node_ids)
    ]
    visualize(nodes, fn_edges, args.output_path)

if __name__ == "__main__":
    main()