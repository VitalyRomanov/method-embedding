from SourceCodeTools.code.data.ast_graph.draw_graph import visualize as visualize_base
from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types, special_mapping


def get_global_nodes():
    return set(node_types.values())


def get_global_edges():
    """
    :return: Set of global edges and their reverses
    """
    types = set()

    for key, value in special_mapping.items():
        types.add(key)
        types.add(value)

    for _, value in node_types.items():
        types.add(value + "_name")

    return types


def ensure_connectedness(nodes, edges):
    unique_nodes = set(edges['src']) | set(edges['dst'])
    num_unique_existing_nodes = nodes["id"].nunique()

    if len(unique_nodes) == num_unique_existing_nodes:
        return

    nodes = nodes.query("id in @unique_nodes", local_dict={"unique_nodes": unique_nodes})
    return nodes, edges


def remove_ast_edges_(nodes, edges):
    global_edges = get_global_edges()
    global_edges.add("subword")
    is_global = lambda type: type in global_edges
    edges = edges.query("type.map(@is_global)", local_dict={"is_global": is_global})
    return ensure_connectedness(nodes, edges)


def remove_global_edges_(nodes, edges):
    global_edges = get_global_edges()
    global_edges.add("global_mention")
    is_ast = lambda type: type not in global_edges
    edges = edges.query("type.map(@is_ast)", local_dict={"is_ast": is_ast})
    return ensure_connectedness(nodes, edges)


def visualize(
        nodes, edges, output_path, show_reverse=False,
        remove_ast_edges=False, remove_global_edges=False
):
    global_node_types = get_global_nodes()
    global_edge_types = get_global_edges()

    if show_reverse is False:
        edges = edges[edges["type"].apply(lambda x: not x.endswith("_rev"))]

    assert not (remove_global_edges is True and remove_ast_edges is False), "Cannot remove all edges"

    if remove_global_edges:
        nodes, edges = remove_global_edges_(nodes, edges)

    if remove_ast_edges:
        nodes, edges = remove_ast_edges_(nodes, edges)

    visualize_base(
        nodes, edges, output_path, show_reverse=show_reverse,
        special_node_types=global_node_types,
        special_edge_types=global_edge_types
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes")
    parser.add_argument("edges")
    parser.add_argument("output")
    args = parser.parse_args()

    nodes = unpersist(args.nodes)
    edges = unpersist(args.edges)
    visualize(nodes, edges, args.output)
