import pandas as pd



def visualize(nodes, edges, output_path):
    import pygraphviz as pgv

    edges = edges[edges["type"].apply(lambda x: not x.endswith("_rev"))]

    id2name = dict(zip(nodes['id'], nodes['serialized_name']))

    g = pgv.AGraph(strict=False, directed=True)

    from SourceCodeTools.code.ast.python_ast2 import PythonNodeEdgeDefinitions
    auxiliaty_edge_types = PythonNodeEdgeDefinitions.auxiliary_edges()

    for ind, edge in edges.iterrows():
        src = edge['source_node_id']
        dst = edge['target_node_id']
        src_name = id2name[src]
        dst_name = id2name[dst]
        g.add_node(src_name, color="black")
        g.add_node(dst_name, color="black")
        g.add_edge(src_name, dst_name, color="blue" if edge['type'] in auxiliaty_edge_types else "black")
        g_edge = g.get_edge(src_name, dst_name)
        g_edge.attr['label'] = edge['type']

    g.layout("dot")
    g.draw(output_path)


if __name__ == "__main__":
    nodes = pd.read_pickle("common_nodes.bz2")
    edges = pd.read_pickle("common_edges.bz2")
    visualize(nodes, edges, "test.pdf")