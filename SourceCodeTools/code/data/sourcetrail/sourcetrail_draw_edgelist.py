import sys

from SourceCodeTools.code.data.file_utils import unpersist


def visualize(edges, output_path):
    import pygraphviz as pgv

    edges = edges[edges["type"].apply(lambda x: not x.endswith("_rev"))]

    g = pgv.AGraph(strict=False, directed=True)

    from SourceCodeTools.code.ast.python_ast2 import PythonNodeEdgeDefinitions
    auxiliaty_edge_types = PythonNodeEdgeDefinitions.auxiliary_edges()

    def strip_name(name):
        parts = name.split("_0x")
        if len(parts) == 1:
            return name
        else:
            return parts[0] + "_" + parts[1][-3:]

    edges["src"] = edges["src"].apply(strip_name)
    edges["dst"] = edges["dst"].apply(strip_name)

    for ind, edge in edges.iterrows():
        src_name = edge['src']
        dst_name = edge['dst']
        g.add_edge(src_name, dst_name, color="blue" if edge['type'] in auxiliaty_edge_types else "black")
        # g_edge = g.get_edge(src_name, dst_name)
        # g_edge.attr['label'] = edge['type']

    g.layout("dot")
    g.draw(output_path)


if __name__ == "__main__":
    edges = unpersist(sys.argv[1])
    visualize(edges, sys.argv[2])
