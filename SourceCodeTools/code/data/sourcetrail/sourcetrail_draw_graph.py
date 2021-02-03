import pandas as pd
import pygraphviz as pgv


def visualize(nodes, edges, output_path):
    from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types

    global_types = list(node_types.values())

    edges = edges[edges["type"].apply(lambda x: not x.endswith("_rev"))]

    id2name = dict(zip(nodes['id'], nodes['serialized_name']))
    id2type = dict(zip(nodes['id'], nodes['type']))

    g = pgv.AGraph(strict=False, directed=True)

    for ind, edge in edges.iterrows():
        src = edge['source_node_id']
        dst = edge['target_node_id']
        src_name = id2name[src]
        dst_name = id2name[dst]
        g.add_node(src_name, color="blue" if id2type[src] in global_types else "black")
        g.add_node(dst_name, color="blue" if id2type[dst] in global_types else "black")
        g.add_edge(src_name, dst_name)
        g_edge = g.get_edge(src_name, dst_name)
        g_edge.attr['label'] = edge['type']

    g.layout("dot")
    g.draw(output_path)


if __name__ == "__main__":
    nodes = pd.read_pickle("common_nodes.bz2")
    edges = pd.read_pickle("common_edges.bz2")
