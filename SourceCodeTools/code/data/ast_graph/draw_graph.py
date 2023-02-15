from SourceCodeTools.code.data.file_utils import unpersist


def visualize(
        nodes, edges, output_path, show_reverse=False,
        special_node_types=None, special_edge_types=None
):
    # do not import unless necessary, not installed by default
    try:
        import pygraphviz as pgv
    except ModuleNotFoundError:
        raise Exception("Install 'pygraphviz'")

    nodes = nodes.rename({"serialized_name": "name"}, axis=1)
    edges = edges.rename({"source_node_id": "src", "target_node_id": "dst"}, axis=1)

    if show_reverse is False:
        edges = edges[edges["type"].apply(lambda x: not x.endswith("_rev"))]

    id2name = dict(zip(nodes['id'], nodes['name']))
    id2type = dict(zip(nodes['id'], nodes['type']))

    g = pgv.AGraph(strict=False, directed=True)

    from SourceCodeTools.code.ast.python_ast2 import PythonNodeEdgeDefinitions
    auxiliaty_edge_types = PythonNodeEdgeDefinitions.auxiliary_edges()
    if special_edge_types is not None:
        auxiliaty_edge_types.update(special_edge_types)

    auxiliaty_node_types = set()
    if special_node_types is not None:
        auxiliaty_node_types.update(special_node_types)

    for ind, edge in edges.iterrows():
        src = edge['src']
        dst = edge['dst']
        src_name = id2name[src]
        dst_name = id2name[dst]
        if src_name.startswith("%"):
            src_name = "▁" + src_name
        if dst_name.startswith("%"):
            dst_name = "▁" + dst_name
        g.add_node(src_name, color="blue" if id2type[src] in auxiliaty_node_types else "black")
        g.add_node(dst_name, color="blue" if id2type[dst] in auxiliaty_node_types else "black")
        g.add_edge(src_name, dst_name, color="blue" if edge['type'] in auxiliaty_edge_types else "black")
        g_edge = g.get_edge(src_name, dst_name)
        g_edge.attr['label'] = edge['type']

    g.layout("dot")
    g.draw(output_path)


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