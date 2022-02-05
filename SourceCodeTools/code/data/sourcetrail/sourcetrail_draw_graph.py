import pandas as pd
import pygraphviz as pgv


def visualize(nodes, edges, output_path):
    from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types

    global_types = list(node_types.values())

    edges = edges[edges["type"].apply(lambda x: not x.endswith("_rev"))]

    from SourceCodeTools.code.data.sourcetrail.Dataset import get_global_edges, ensure_connectedness

    # def remove_ast_edges(nodes, edges):
    #     global_edges = get_global_edges()
    #     global_edges.add("subword")
    #     is_global = lambda type: type in global_edges
    #     edges = edges.query("type.map(@is_global)", local_dict={"is_global": is_global})
    #     return nodes, edges # ensure_connectedness(nodes, edges)

    # def remove_global_edges(nodes, edges):
    #     global_edges = get_global_edges()
    #     global_edges.add("global_mention")
    #     is_ast = lambda type: type not in global_edges
    #     edges = edges.query("type.map(@is_ast)", local_dict={"is_ast": is_ast})
    #     return nodes, edges # ensure_connectedness(nodes, edges)
    #
    # nodes, edges = remove_global_edges(nodes, edges)

    id2name = dict(zip(nodes['id'], nodes['serialized_name']))
    id2type = dict(zip(nodes['id'], nodes['type']))

    g = pgv.AGraph(strict=False, directed=True)

    from SourceCodeTools.code.python_ast2 import PythonNodeEdgeDefinitions
    auxiliaty_edge_types = PythonNodeEdgeDefinitions.auxiliary_edges()

    for ind, edge in edges.iterrows():
        src = edge['source_node_id']
        dst = edge['target_node_id']
        src_name = id2name[src]
        dst_name = id2name[dst]
        g.add_node(src_name, color="blue" if id2type[src] in global_types else "black")
        g.add_node(dst_name, color="blue" if id2type[dst] in global_types else "black")
        g.add_edge(src_name, dst_name, color="blue" if edge['type'] in auxiliaty_edge_types else "black")
        g_edge = g.get_edge(src_name, dst_name)
        g_edge.attr['label'] = edge['type']

    g.layout("dot")
    g.draw(output_path)


if __name__ == "__main__":
    nodes = pd.read_pickle("common_nodes.bz2")
    edges = pd.read_pickle("common_edges.bz2")
    visualize(nodes, edges, "test.pdf")
