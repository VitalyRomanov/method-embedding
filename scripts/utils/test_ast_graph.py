import json
import os
from collections import defaultdict
from pathlib import Path

from SourceCodeTools.code.ast.python_examples import PythonCodeExamplesForNodes
from SourceCodeTools.code.data.ast_graph.build_ast_graph import source_code_to_graph
from SourceCodeTools.code.data.ast_graph.draw_graph import visualize


# node = "visualize"
# code = """def print_sum(iterable, add_text=False):
#     acc = 0
#     for a in iterable:
#         acc += a
#     while True:
#         print("Total sum:" if add_text else "", acc.a.c.v.v.b, this_call(a,b,c))"""
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


def test_graph_builder():

    correct_answers = {
        "v2.5": {
            "Assign": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "AugAssign1": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign2": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign3": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign4": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign5": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign6": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign7": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign8": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign9": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign10": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign11": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign12": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "AugAssign13": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "Delete": {
                "num_nodes": 4,
                "num_edges": 5
            },
            "Global": {
                "num_nodes": 4,
                "num_edges": 5
            },
            "Nonlocal": {
                "num_nodes": 4,
                "num_edges": 5
            },
            "Slice": {
                "num_nodes": 6,
                "num_edges": 9
            },
            "ExtSlice": {
                "num_nodes": 8,
                "num_edges": 14
            },
            "Index": {
                "num_nodes": 6,
                "num_edges": 8
            },
            "Starred": {
                "num_nodes": 4,
                "num_edges": 5
            },
            "Yield": {
                "num_nodes": 4,
                "num_edges": 5
            },
            "YieldFrom": {
                "num_nodes": 4,
                "num_edges": 5
            },
            "Compare1": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare2": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare3": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare4": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare5": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare6": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare7": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare8": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare9": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Compare10": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp1": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp2": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp3": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp4": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp5": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp6": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp7": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp8": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp9": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp10": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp11": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BinOp12": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BoolOp1": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "BoolOp2": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "UnaryOp1": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "UnaryOp2": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "UnaryOp3": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Assert": {
                "num_nodes": 8,
                "num_edges": 11
            },
            "FunctionDef": {
                "num_nodes": 9,
                "num_edges": 16
            },
            "AsyncFunctionDef": {
                "num_nodes": 9,
                "num_edges": 16
            },
            "ClassDef": {
                "num_nodes": 9,
                "num_edges": 13
            },
            "AnnAssign": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "With": {
                "num_nodes": 15,
                "num_edges": 24
            },
            "AsyncWith": {
                "num_nodes": 15,
                "num_edges": 24
            },
            "arg": {
                "num_nodes": 12,
                "num_edges": 19
            },
            "Await": {
                "num_nodes": 5,
                "num_edges": 7
            },
            "Raise": {
                "num_nodes": 5,
                "num_edges": 7
            },
            "Lambda": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "IfExp": {
                "num_nodes": 7,
                "num_edges": 10
            },
            "keyword": {
                "num_nodes": 11,
                "num_edges": 15
            },
            "Attribute": {
                "num_nodes": 9,
                "num_edges": 11
            },
            "If": {
                "num_nodes": 17,
                "num_edges": 28
            },
            "For": {
                "num_nodes": 22,
                "num_edges": 48
            },
            "AsyncFor": {
                "num_nodes": 22,
                "num_edges": 48
            },
            "Try": {
                "num_nodes": 19,
                "num_edges": 36
            },
            "While": {
                "num_nodes": 13,
                "num_edges": 20
            },
            "Break": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Continue": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Pass": {
                "num_nodes": 3,
                "num_edges": 3
            },
            "Dict": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Set": {
                "num_nodes": 6,
                "num_edges": 8
            },
            "ListComp": {
                "num_nodes": 7,
                "num_edges": 12
            },
            "DictComp": {
                "num_nodes": 10,
                "num_edges": 19
            },
            "SetComp": {
                "num_nodes": 7,
                "num_edges": 12
            },
            "GeneratorExp": {
                "num_nodes": 10,
                "num_edges": 18
            },
            "BinOp": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "ImportFrom": {
                "num_nodes": 7,
                "num_edges": 10
            },
            "alias": {
                "num_nodes": 7,
                "num_edges": 10
            },
            "List": {
                "num_nodes": 6,
                "num_edges": 8
            },
            "Tuple": {
                "num_nodes": 6,
                "num_edges": 8
            },
            "JoinedStr": {
                "num_nodes": 2,
                "num_edges": 1
            },
            "FormattedValue": {
                "num_nodes": 2,
                "num_edges": 1
            },
            "Bytes": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Num": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Str": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "FunctionDef2": {
                "num_nodes": 46,
                "num_edges": 75
            },
            "FunctionDef3": {
                "num_nodes": 17,
                "num_edges": 25
            }
        },
        "v1.0_control_flow": {
            "Assign": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "AugAssign1": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign2": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign3": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign4": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign5": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign6": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign7": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign8": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign9": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign10": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign11": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign12": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "AugAssign13": {
                "num_nodes": 8,
                "num_edges": 8
            },
            "Delete": {
                "num_nodes": 5,
                "num_edges": 5
            },
            "Global": {
                "num_nodes": 5,
                "num_edges": 5
            },
            "Nonlocal": {
                "num_nodes": 5,
                "num_edges": 5
            },
            "Slice": {
                "num_nodes": 8,
                "num_edges": 8
            },
            "ExtSlice": {
                "num_nodes": 10,
                "num_edges": 10
            },
            "Index": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Starred": {
                "num_nodes": 4,
                "num_edges": 4
            },
            "Yield": {
                "num_nodes": 5,
                "num_edges": 5
            },
            "YieldFrom": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Compare1": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Compare2": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "Compare3": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Compare4": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Compare5": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "Compare6": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "Compare7": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Compare8": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "Compare9": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Compare10": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "BinOp1": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp2": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp3": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp4": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp5": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp6": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp7": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "BinOp8": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp9": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp10": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp11": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BinOp12": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BoolOp1": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "BoolOp2": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "UnaryOp1": {
                "num_nodes": 4,
                "num_edges": 4
            },
            "UnaryOp2": {
                "num_nodes": 4,
                "num_edges": 4
            },
            "UnaryOp3": {
                "num_nodes": 4,
                "num_edges": 4
            },
            "Assert": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "FunctionDef": {
                "num_nodes": 11,
                "num_edges": 17
            },
            "AsyncFunctionDef": {
                "num_nodes": 11,
                "num_edges": 17
            },
            "ClassDef": {
                "num_nodes": 9,
                "num_edges": 13
            },
            "AnnAssign": {
                "num_nodes": 8,
                "num_edges": 8
            },
            "With": {
                "num_nodes": 12,
                "num_edges": 19
            },
            "AsyncWith": {
                "num_nodes": 12,
                "num_edges": 19
            },
            "arg": {
                "num_nodes": 12,
                "num_edges": 18
            },
            "Await": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "Raise": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "Lambda": {
                "num_nodes": 8,
                "num_edges": 8
            },
            "IfExp": {
                "num_nodes": 10,
                "num_edges": 10
            },
            "keyword": {
                "num_nodes": 12,
                "num_edges": 12
            },
            "Attribute": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "If": {
                "num_nodes": 14,
                "num_edges": 21
            },
            "For": {
                "num_nodes": 22,
                "num_edges": 44
            },
            "AsyncFor": {
                "num_nodes": 22,
                "num_edges": 44
            },
            "Try": {
                "num_nodes": 18,
                "num_edges": 30
            },
            "While": {
                "num_nodes": 11,
                "num_edges": 14
            },
            "Break": {
                "num_nodes": 6,
                "num_edges": 8
            },
            "Continue": {
                "num_nodes": 6,
                "num_edges": 8
            },
            "Pass": {
                "num_nodes": 3,
                "num_edges": 3
            },
            "Dict": {
                "num_nodes": 11,
                "num_edges": 11
            },
            "Set": {
                "num_nodes": 8,
                "num_edges": 8
            },
            "ListComp": {
                "num_nodes": 9,
                "num_edges": 9
            },
            "DictComp": {
                "num_nodes": 12,
                "num_edges": 12
            },
            "SetComp": {
                "num_nodes": 9,
                "num_edges": 9
            },
            "GeneratorExp": {
                "num_nodes": 13,
                "num_edges": 13
            },
            "BinOp": {
                "num_nodes": 8,
                "num_edges": 8
            },
            "ImportFrom": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "alias": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "List": {
                "num_nodes": 12,
                "num_edges": 12
            },
            "Tuple": {
                "num_nodes": 12,
                "num_edges": 12
            },
            "JoinedStr": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "FormattedValue": {
                "num_nodes": 10,
                "num_edges": 10
            },
            "Bytes": {
                "num_nodes": 8,
                "num_edges": 8
            },
            "Num": {
                "num_nodes": 6,
                "num_edges": 6
            },
            "Str": {
                "num_nodes": 7,
                "num_edges": 7
            },
            "FunctionDef2": {
                "num_nodes": 24,
                "num_edges": 36
            },
            "FunctionDef3": {
                "num_nodes": 12,
                "num_edges": 18
            }
        },
        "v3.5": {
            "Assign": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign1": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign2": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign3": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign4": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign5": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign6": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign7": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign8": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign9": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign10": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign11": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign12": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "AugAssign13": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "Delete": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Global": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "Nonlocal": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "Slice": {
                "num_nodes": 9,
                "num_edges": 13
            },
            "ExtSlice": {
                "num_nodes": 11,
                "num_edges": 18
            },
            "Index": {
                "num_nodes": 9,
                "num_edges": 12
            },
            "Starred": {
                "num_nodes": 7,
                "num_edges": 9
            },
            "Yield": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "YieldFrom": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Compare1": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare2": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare3": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare4": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare5": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare6": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare7": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare8": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare9": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "Compare10": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp1": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp2": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp3": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp4": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp5": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp6": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp7": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp8": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp9": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp10": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp11": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BinOp12": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BoolOp1": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "BoolOp2": {
                "num_nodes": 12,
                "num_edges": 15
            },
            "UnaryOp1": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "UnaryOp2": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "UnaryOp3": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "Assert": {
                "num_nodes": 13,
                "num_edges": 17
            },
            "FunctionDef": {
                "num_nodes": 15,
                "num_edges": 23
            },
            "AsyncFunctionDef": {
                "num_nodes": 15,
                "num_edges": 23
            },
            "ClassDef": {
                "num_nodes": 13,
                "num_edges": 17
            },
            "AnnAssign": {
                "num_nodes": 10,
                "num_edges": 11
            },
            "With": {
                "num_nodes": 26,
                "num_edges": 39
            },
            "AsyncWith": {
                "num_nodes": 26,
                "num_edges": 39
            },
            "arg": {
                "num_nodes": 18,
                "num_edges": 26
            },
            "Await": {
                "num_nodes": 8,
                "num_edges": 10
            },
            "Raise": {
                "num_nodes": 8,
                "num_edges": 10
            },
            "Lambda": {
                "num_nodes": 10,
                "num_edges": 12
            },
            "IfExp": {
                "num_nodes": 10,
                "num_edges": 13
            },
            "keyword": {
                "num_nodes": 14,
                "num_edges": 18
            },
            "Attribute": {
                "num_nodes": 12,
                "num_edges": 16
            },
            "If": {
                "num_nodes": 30,
                "num_edges": 55
            },
            "For": {
                "num_nodes": 38,
                "num_edges": 75
            },
            "AsyncFor": {
                "num_nodes": 38,
                "num_edges": 75
            },
            "Try": {
                "num_nodes": 36,
                "num_edges": 63
            },
            "While": {
                "num_nodes": 21,
                "num_edges": 32
            },
            "Break": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Continue": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Pass": {
                "num_nodes": 3,
                "num_edges": 3
            },
            "Dict": {
                "num_nodes": 19,
                "num_edges": 26
            },
            "Set": {
                "num_nodes": 11,
                "num_edges": 14
            },
            "ListComp": {
                "num_nodes": 14,
                "num_edges": 21
            },
            "DictComp": {
                "num_nodes": 20,
                "num_edges": 35
            },
            "SetComp": {
                "num_nodes": 14,
                "num_edges": 21
            },
            "GeneratorExp": {
                "num_nodes": 18,
                "num_edges": 30
            },
            "BinOp": {
                "num_nodes": 18,
                "num_edges": 23
            },
            "ImportFrom": {
                "num_nodes": 11,
                "num_edges": 14
            },
            "alias": {
                "num_nodes": 11,
                "num_edges": 14
            },
            "List": {
                "num_nodes": 10,
                "num_edges": 12
            },
            "Tuple": {
                "num_nodes": 10,
                "num_edges": 12
            },
            "JoinedStr": {
                "num_nodes": 2,
                "num_edges": 1
            },
            "FormattedValue": {
                "num_nodes": 2,
                "num_edges": 1
            },
            "Bytes": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "Num": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "Str": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "FunctionDef2": {
                "num_nodes": 69,
                "num_edges": 99
            },
            "FunctionDef3": {
                "num_nodes": 26,
                "num_edges": 34
            }
        },
        "v3.5_control_flow": {
            "Assign": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "AugAssign1": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign2": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign3": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign4": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign5": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign6": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign7": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign8": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign9": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign10": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign11": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign12": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "AugAssign13": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "Delete": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "Global": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Nonlocal": {
                "num_nodes": 5,
                "num_edges": 6
            },
            "Slice": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "ExtSlice": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Index": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Starred": {
                "num_nodes": 6,
                "num_edges": 8
            },
            "Yield": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "YieldFrom": {
                "num_nodes": 6,
                "num_edges": 7
            },
            "Compare1": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare2": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare3": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare4": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare5": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare6": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare7": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare8": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare9": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "Compare10": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp1": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp2": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp3": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp4": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp5": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp6": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp7": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp8": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp9": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp10": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp11": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BinOp12": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BoolOp1": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "BoolOp2": {
                "num_nodes": 10,
                "num_edges": 14
            },
            "UnaryOp1": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "UnaryOp2": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "UnaryOp3": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Assert": {
                "num_nodes": 11,
                "num_edges": 15
            },
            "FunctionDef": {
                "num_nodes": 11,
                "num_edges": 17
            },
            "AsyncFunctionDef": {
                "num_nodes": 11,
                "num_edges": 17
            },
            "ClassDef": {
                "num_nodes": 11,
                "num_edges": 13
            },
            "AnnAssign": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "With": {
                "num_nodes": 21,
                "num_edges": 34
            },
            "AsyncWith": {
                "num_nodes": 21,
                "num_edges": 34
            },
            "arg": {
                "num_nodes": 14,
                "num_edges": 20
            },
            "Await": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Raise": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Lambda": {
                "num_nodes": 8,
                "num_edges": 9
            },
            "IfExp": {
                "num_nodes": 9,
                "num_edges": 11
            },
            "keyword": {
                "num_nodes": 11,
                "num_edges": 12
            },
            "Attribute": {
                "num_nodes": 10,
                "num_edges": 11
            },
            "If": {
                "num_nodes": 26,
                "num_edges": 49
            },
            "For": {
                "num_nodes": 33,
                "num_edges": 65
            },
            "AsyncFor": {
                "num_nodes": 33,
                "num_edges": 65
            },
            "Try": {
                "num_nodes": 30,
                "num_edges": 57
            },
            "While": {
                "num_nodes": 18,
                "num_edges": 29
            },
            "Break": {
                "num_nodes": 5,
                "num_edges": 4
            },
            "Continue": {
                "num_nodes": 5,
                "num_edges": 4
            },
            "Pass": {
                "num_nodes": 3,
                "num_edges": 2
            },
            "Dict": {
                "num_nodes": 15,
                "num_edges": 25
            },
            "Set": {
                "num_nodes": 9,
                "num_edges": 13
            },
            "ListComp": {
                "num_nodes": 11,
                "num_edges": 18
            },
            "DictComp": {
                "num_nodes": 16,
                "num_edges": 30
            },
            "SetComp": {
                "num_nodes": 11,
                "num_edges": 18
            },
            "GeneratorExp": {
                "num_nodes": 14,
                "num_edges": 25
            },
            "BinOp": {
                "num_nodes": 15,
                "num_edges": 21
            },
            "ImportFrom": {
                "num_nodes": 8,
                "num_edges": 11
            },
            "alias": {
                "num_nodes": 8,
                "num_edges": 11
            },
            "List": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "Tuple": {
                "num_nodes": 9,
                "num_edges": 10
            },
            "JoinedStr": {
                "num_nodes": 2,
                "num_edges": 1
            },
            "FormattedValue": {
                "num_nodes": 2,
                "num_edges": 1
            },
            "Bytes": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Num": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "Str": {
                "num_nodes": 7,
                "num_edges": 8
            },
            "FunctionDef2": {
                "num_nodes": 49,
                "num_edges": 79
            },
            "FunctionDef3": {
                "num_nodes": 19,
                "num_edges": 26
            }
        }
    }

    bpe_tokenizer_path = Path(os.getcwd())\
        .joinpath(__file__)\
        .parent.parent.parent\
        .joinpath("examples", "sentencepiece_bpe.model")

    assert bpe_tokenizer_path.is_file()

    answers = {
        "v2.5": defaultdict(lambda: {}),
        "v1.0_control_flow": defaultdict(lambda: {}),
        "v3.5": defaultdict(lambda: {}),
        "v3.5_control_flow": defaultdict(lambda: {}),
    }

    for node, code in PythonCodeExamplesForNodes.examples.items():
        print(node)
        variety = "v2.5"
        graph = source_code_to_graph(
            code, variety=variety, reverse_edges=True, bpe_tokenizer_path=bpe_tokenizer_path
        )
        print("\t", variety, len(graph["nodes"]), len(graph["edges"]))
        assert (
                correct_answers[variety][node]["num_nodes"] == len(graph["nodes"]) and
                correct_answers[variety][node]["num_edges"] == len(graph["edges"])
        )
        # answers[variety][node]["num_nodes"] = len(graph["nodes"])
        # answers[variety][node]["num_edges"] = len(graph["edges"])
        visualize(graph["nodes"], graph["edges"], f"{node}_{variety}.png", show_reverse=True)

        variety = "v1.0_control_flow"
        graph = source_code_to_graph(code, variety=variety)
        print("\t", variety, len(graph["nodes"]), len(graph["edges"]))
        assert (
                correct_answers[variety][node]["num_nodes"] == len(graph["nodes"]) and
                correct_answers[variety][node]["num_edges"] == len(graph["edges"])
        )
        visualize(graph["nodes"], graph["edges"], f"{node}_{variety}.png", show_reverse=True)

        variety = "v3.5"
        graph = source_code_to_graph(
            code, variety=variety, reverse_edges=True, mention_instances=True, bpe_tokenizer_path=bpe_tokenizer_path
        )
        print("\t", variety, len(graph["nodes"]), len(graph["edges"]))
        assert (
                correct_answers[variety][node]["num_nodes"] == len(graph["nodes"]) and
                correct_answers[variety][node]["num_edges"] == len(graph["edges"])
        )
        visualize(
            graph["nodes"].rename({"name": "serialized_name"}, axis=1),
            graph["edges"].rename({"src": "source_node_id", "dst": "target_node_id"}, axis=1),
            f"{node}_{variety}_instances.png", show_reverse=True
        )

        variety = "v3.5_control_flow"
        graph = source_code_to_graph(
            code, variety=variety, reverse_edges=False, mention_instances=True, bpe_tokenizer_path=bpe_tokenizer_path
        )
        assert (
                correct_answers[variety][node]["num_nodes"] == len(graph["nodes"]) and
                correct_answers[variety][node]["num_edges"] == len(graph["edges"])
        )
        print("\t", variety, len(graph["nodes"]), len(graph["edges"]))
        visualize(
            graph["nodes"].rename({"name": "serialized_name"}, axis=1),
            graph["edges"].rename({"src": "source_node_id", "dst": "target_node_id"}, axis=1),
            f"{node}_{variety}.png", show_reverse=True
        )

    print(json.dumps(answers, indent=4))


if __name__ == "__main__":
    test_graph_builder()