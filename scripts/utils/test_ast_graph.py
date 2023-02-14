from SourceCodeTools.code.ast.python_examples import PythonCodeExamplesForNodes
from SourceCodeTools.code.data.ast_graph import source_code_to_graph
from SourceCodeTools.code.data.ast_graph.draw_graph import visualize


node = "visualize"
code = """def print_sum(iterable, add_text=False):
    acc = 0
    for a in iterable:
        acc += a
    while True:
        print("Total sum:" if add_text else "", acc.a.c.v.v.b, this_call(a,b,c))"""
# code = """def visualize(nodes, edges, output_path, show_reverse=False):
#     import pygraphviz as pgv
#
#     if show_reverse is False:
#         edges = edges[edges["type"].apply(lambda x: not x.endswith("_rev"))]
#
#     id2name = dict(zip(nodes['id'], nodes['serialized_name']))
#
#     g = pgv.AGraph(strict=False, directed=True)
#
#     from SourceCodeTools.code.ast.python_ast2 import PythonNodeEdgeDefinitions
#     auxiliaty_edge_types = PythonNodeEdgeDefinitions.auxiliary_edges()
#
#     for ind, edge in edges.iterrows():
#         src = edge['source_node_id']
#         dst = edge['target_node_id']
#         src_name = id2name[src]
#         dst_name = id2name[dst]
#         g.add_node(src_name, color="black")
#         g.add_node(dst_name, color="black")
#         g.add_edge(src_name, dst_name, color="blue" if edge['type'] in auxiliaty_edge_types else "black")
#         g_edge = g.get_edge(src_name, dst_name)
#         g_edge.attr['label'] = edge['type']
#
#     g.layout("dot")
#     g.draw(output_path)"""


# for node, code in PythonCodeExamplesForNodes.examples.items():
# print(node)
# variety = "with_mention"
# nodes, edges = source_code_to_graph(code, variety=variety, reverse_edges=True)
# print("\t", variety, len(nodes), len(edges))
# visualize(nodes, edges, f"{node}_{variety}.png", show_reverse=True)
#
# variety = "cf"
# nodes, edges = source_code_to_graph(code, variety=variety)
# print("\t", variety, len(nodes), len(edges))
# visualize(nodes, edges, f"{node}_{variety}.png", show_reverse=True)
#
variety = "new_with_mention"
nodes, edges = source_code_to_graph(code, variety=variety, reverse_edges=True, mention_instances=True)
print("\t", variety, len(nodes), len(edges))
visualize(
    nodes.rename({"name": "serialized_name"}, axis=1),
    edges.rename({"src": "source_node_id", "dst": "target_node_id"}, axis=1),
    f"{node}_{variety}_instances.png", show_reverse=True
)

variety = "new_cf"
nodes, edges = source_code_to_graph(code, variety=variety, reverse_edges=False, mention_instances=True)
print("\t", variety, len(nodes), len(edges))
visualize(
    nodes.rename({"name": "serialized_name"}, axis=1),
    edges.rename({"src": "source_node_id", "dst": "target_node_id"}, axis=1),
    f"{node}_{variety}.png", show_reverse=True
)