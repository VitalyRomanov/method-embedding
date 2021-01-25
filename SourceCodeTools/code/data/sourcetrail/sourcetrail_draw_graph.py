import pandas as pd
import pygraphviz as pgv

from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types

global_types = list(node_types.values())

nodes = pd.read_pickle("common_nodes.bz2")
edges = pd.read_pickle("common_edges.bz2")

id2name = dict(zip(nodes['id'], nodes['serialized_name']))
id2type = dict(zip(nodes['id'], nodes['type']))

G=pgv.AGraph(strict=False,directed=True)

for ind, edge in edges.iterrows():
    src = edge['source_node_id']
    dst = edge['target_node_id']
    src_name = id2name[src]
    dst_name = id2name[dst]
    G.add_node(src_name, color="blue" if id2type[src] in global_types else "black")
    G.add_node(dst_name, color="blue" if id2type[dst] in global_types else "black")
    G.add_edge(src_name, dst_name)
    g_edge = G.get_edge(src_name, dst_name)
    g_edge.attr['label'] = edge['type']

G.layout("dot")
G.draw('file.pdf')
