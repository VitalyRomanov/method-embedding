import json
import os
from collections import defaultdict, Counter
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
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "Assign": 1,
                        "Module": 1,
                        "subword": 1,
                        "mention": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "AugAssign1": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign2": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign3": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign4": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign5": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            7
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign6": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            7
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign7": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            9
                        ],
                        [
                            5,
                            9
                        ]
                    ]
                }
            },
            "AugAssign8": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            9
                        ],
                        [
                            5,
                            9
                        ]
                    ]
                }
            },
            "AugAssign9": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            7
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign10": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            7
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign11": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign12": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign13": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "AugAssign": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Delete": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Delete": 1
                    },
                    "edges": {
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            3
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Global": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Global": 1
                    },
                    "edges": {
                        "subword": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            6
                        ]
                    ]
                }
            },
            "Nonlocal": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Nonlocal": 1
                    },
                    "edges": {
                        "subword": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            8
                        ]
                    ]
                }
            },
            "Slice": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Subscript": 1,
                        "Constant": 1,
                        "Slice": 1
                    },
                    "edges": {
                        "subword": 1,
                        "value": 1,
                        "value_rev": 1,
                        "lower": 1,
                        "upper": 1,
                        "slice": 1,
                        "slice_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            2,
                            3
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "ExtSlice": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Subscript": 1,
                        "Constant": 1,
                        "Index": 1,
                        "ExtSlice": 1,
                        "Slice": 1
                    },
                    "edges": {
                        "subword": 1,
                        "value": 2,
                        "value_rev": 1,
                        "dims": 2,
                        "dims_rev": 2,
                        "lower": 1,
                        "upper": 1,
                        "slice": 1,
                        "slice_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            8
                        ],
                        [
                            2,
                            3
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "Index": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Subscript": 1,
                        "Constant": 1,
                        "Index": 1
                    },
                    "edges": {
                        "subword": 1,
                        "value": 2,
                        "value_rev": 1,
                        "slice": 1,
                        "slice_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            4
                        ],
                        [
                            2,
                            3
                        ]
                    ]
                }
            },
            "Starred": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Starred": 1
                    },
                    "edges": {
                        "subword": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "Yield": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Yield": 1
                    },
                    "edges": {
                        "subword": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "YieldFrom": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "YieldFrom": 1
                    },
                    "edges": {
                        "subword": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            10
                        ],
                        [
                            11,
                            12
                        ]
                    ]
                }
            },
            "Compare1": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare2": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare3": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Compare4": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Compare5": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare6": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare7": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare8": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            10
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "Compare9": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare10": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            10
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "BinOp1": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp2": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp3": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp4": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp5": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp6": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp7": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp8": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp9": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp10": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp11": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp12": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BoolOp1": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BoolOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "values": 2,
                        "values_rev": 2,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            7
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "BoolOp2": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "BoolOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 2,
                        "values": 2,
                        "values_rev": 2,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            6
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BoolOp3": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "UnaryOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "UnaryOp1": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "UnaryOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            2
                        ],
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "UnaryOp2": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "UnaryOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            2
                        ],
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "UnaryOp3": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "UnaryOp": 1,
                        "Op": 1
                    },
                    "edges": {
                        "subword": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            2
                        ],
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "Assert": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Assert": 1
                    },
                    "edges": {
                        "subword": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "test": 1,
                        "test_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            6
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            7,
                            13
                        ],
                        [
                            12,
                            13
                        ]
                    ]
                }
            },
            "FunctionDef": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 1,
                        "arguments": 1,
                        "Return": 1
                    },
                    "edges": {
                        "subword": 2,
                        "arg": 2,
                        "arg_rev": 2,
                        "args": 1,
                        "args_rev": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            3
                        ],
                        [
                            6,
                            7
                        ],
                        [
                            13,
                            19
                        ],
                        [
                            20,
                            21
                        ]
                    ]
                }
            },
            "AsyncFunctionDef": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "AsyncFunctionDef": 1,
                        "Module": 1,
                        "arg": 1,
                        "arguments": 1,
                        "Return": 1
                    },
                    "edges": {
                        "subword": 2,
                        "arg": 2,
                        "arg_rev": 2,
                        "args": 1,
                        "args_rev": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            9
                        ],
                        [
                            12,
                            13
                        ],
                        [
                            19,
                            25
                        ],
                        [
                            26,
                            27
                        ]
                    ]
                }
            },
            "ClassDef": {
                "description": {
                    "nodes": {
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1,
                        "FunctionDef": 1,
                        "ClassDef": 1,
                        "Module": 1,
                        "subword": 2,
                        "mention": 2
                    },
                    "edges": {
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "subword": 2,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_class": 1,
                        "defined_in_class_rev": 1,
                        "class_name": 1,
                        "class_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            12,
                            15
                        ],
                        [
                            28,
                            32
                        ]
                    ]
                }
            },
            "AnnAssign": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 1,
                        "Module": 1,
                        "AnnAssign": 1,
                        "Constant": 1,
                        "type_annotation": 1
                    },
                    "edges": {
                        "subword": 2,
                        "target": 1,
                        "target_rev": 1,
                        "value": 1,
                        "annotation_for": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            10
                        ],
                        [
                            3,
                            6
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "With": {
                "description": {
                    "nodes": {
                        "subword": 6,
                        "mention": 4,
                        "Module": 1,
                        "Call": 2,
                        "withitem": 1,
                        "With": 1
                    },
                    "edges": {
                        "subword": 6,
                        "func": 2,
                        "func_rev": 2,
                        "args": 2,
                        "args_rev": 2,
                        "context_expr": 1,
                        "context_expr_rev": 1,
                        "optional_vars": 1,
                        "optional_vars_rev": 1,
                        "items": 1,
                        "items_rev": 1,
                        "executed_inside_with": 1,
                        "executed_inside_with_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            4
                        ],
                        [
                            5,
                            9
                        ],
                        [
                            5,
                            12
                        ],
                        [
                            10,
                            11
                        ],
                        [
                            16,
                            17
                        ],
                        [
                            22,
                            30
                        ],
                        [
                            22,
                            33
                        ],
                        [
                            31,
                            32
                        ]
                    ]
                }
            },
            "AsyncWith": {
                "description": {
                    "nodes": {
                        "subword": 6,
                        "mention": 4,
                        "Module": 1,
                        "Call": 2,
                        "withitem": 1,
                        "AsyncWith": 1
                    },
                    "edges": {
                        "subword": 6,
                        "func": 2,
                        "func_rev": 2,
                        "args": 2,
                        "args_rev": 2,
                        "context_expr": 1,
                        "context_expr_rev": 1,
                        "optional_vars": 1,
                        "optional_vars_rev": 1,
                        "items": 1,
                        "items_rev": 1,
                        "executed_inside_with": 1,
                        "executed_inside_with_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            10
                        ],
                        [
                            11,
                            15
                        ],
                        [
                            11,
                            18
                        ],
                        [
                            16,
                            17
                        ],
                        [
                            22,
                            23
                        ],
                        [
                            28,
                            36
                        ],
                        [
                            28,
                            39
                        ],
                        [
                            37,
                            38
                        ]
                    ]
                }
            },
            "arg": {
                "description": {
                    "nodes": {
                        "subword": 3,
                        "mention": 2,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 1,
                        "type_annotation": 1,
                        "Constant": 1,
                        "arguments": 1,
                        "Return": 1
                    },
                    "edges": {
                        "subword": 3,
                        "arg": 2,
                        "arg_rev": 2,
                        "annotation_for": 1,
                        "default": 1,
                        "args": 1,
                        "args_rev": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            3
                        ],
                        [
                            6,
                            12
                        ],
                        [
                            9,
                            12
                        ],
                        [
                            15,
                            16
                        ],
                        [
                            22,
                            28
                        ],
                        [
                            29,
                            30
                        ]
                    ]
                }
            },
            "Await": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Call": 1,
                        "Await": 1
                    },
                    "edges": {
                        "subword": 1,
                        "func": 1,
                        "func_rev": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            6,
                            10
                        ],
                        [
                            6,
                            12
                        ]
                    ]
                }
            },
            "Raise": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "Call": 1,
                        "Raise": 1
                    },
                    "edges": {
                        "subword": 1,
                        "func": 1,
                        "func_rev": 1,
                        "exc": 1,
                        "exc_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            6,
                            15
                        ],
                        [
                            6,
                            17
                        ]
                    ]
                }
            },
            "Lambda": {
                "description": {
                    "nodes": {
                        "subword": 1,
                        "mention": 1,
                        "Module": 1,
                        "BinOp": 1,
                        "Constant": 1,
                        "Op": 1,
                        "Lambda": 1
                    },
                    "edges": {
                        "subword": 1,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "op": 1,
                        "lambda": 1,
                        "lambda_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            6
                        ],
                        [
                            10,
                            11
                        ],
                        [
                            10,
                            15
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "IfExp": {
                "description": {
                    "nodes": {
                        "Constant": 2,
                        "IfExp": 1,
                        "Module": 1,
                        "Assign": 1,
                        "subword": 1,
                        "mention": 1
                    },
                    "edges": {
                        "test": 1,
                        "if_true": 1,
                        "if_false": 1,
                        "value": 1,
                        "value_rev": 1,
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            20
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            4,
                            20
                        ],
                        [
                            9,
                            13
                        ],
                        [
                            19,
                            20
                        ]
                    ]
                }
            },
            "keyword": {
                "description": {
                    "nodes": {
                        "subword": 3,
                        "mention": 1,
                        "Module": 1,
                        "Call": 1,
                        "#keyword#": 2,
                        "keyword": 2,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 3,
                        "func": 1,
                        "func_rev": 1,
                        "arg": 2,
                        "value": 2,
                        "keywords": 2,
                        "keywords_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            2
                        ],
                        [
                            0,
                            12
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            10,
                            11
                        ]
                    ]
                }
            },
            "Attribute": {
                "description": {
                    "nodes": {
                        "subword": 3,
                        "mention": 1,
                        "Module": 1,
                        "Attribute": 2,
                        "#attr#": 2
                    },
                    "edges": {
                        "subword": 3,
                        "value": 2,
                        "value_rev": 2,
                        "attr": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            3
                        ],
                        [
                            0,
                            5
                        ]
                    ]
                }
            },
            "If": {
                "description": {
                    "nodes": {
                        "subword": 4,
                        "mention": 4,
                        "Module": 1,
                        "Compare": 2,
                        "Op": 1,
                        "Constant": 1,
                        "If": 2,
                        "Assign": 3,
                        "Tuple": 2
                    },
                    "edges": {
                        "subword": 4,
                        "left": 2,
                        "left_rev": 2,
                        "ops": 2,
                        "comparators": 2,
                        "test": 2,
                        "test_rev": 2,
                        "value": 3,
                        "value_rev": 3,
                        "targets": 3,
                        "targets_rev": 3,
                        "executed_if_true": 2,
                        "executed_if_true_rev": 2,
                        "elts": 2,
                        "elts_rev": 2,
                        "executed_if_false": 2,
                        "executed_if_false_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            2
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            3,
                            12
                        ],
                        [
                            8,
                            12
                        ],
                        [
                            17,
                            18
                        ],
                        [
                            17,
                            22
                        ],
                        [
                            21,
                            22
                        ],
                        [
                            23,
                            27
                        ],
                        [
                            28,
                            29
                        ],
                        [
                            28,
                            38
                        ],
                        [
                            33,
                            38
                        ],
                        [
                            42,
                            43
                        ],
                        [
                            42,
                            47
                        ],
                        [
                            46,
                            47
                        ],
                        [
                            57,
                            58
                        ],
                        [
                            57,
                            61
                        ],
                        [
                            57,
                            68
                        ],
                        [
                            60,
                            61
                        ],
                        [
                            64,
                            65
                        ],
                        [
                            64,
                            68
                        ],
                        [
                            67,
                            68
                        ]
                    ]
                }
            },
            "For": {
                "description": {
                    "nodes": {
                        "subword": 5,
                        "mention": 5,
                        "Module": 1,
                        "For": 1,
                        "Call": 3,
                        "Assign": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1,
                        "If": 1,
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1
                    },
                    "edges": {
                        "subword": 6,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "func": 3,
                        "func_rev": 3,
                        "args": 3,
                        "args_rev": 2,
                        "value": 1,
                        "value_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "executed_in_for": 2,
                        "executed_in_for_rev": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "test": 1,
                        "test_rev": 1,
                        "executed_if_true": 2,
                        "executed_if_true_rev": 2,
                        "control_flow": 1,
                        "next": 2,
                        "prev": 2,
                        "executed_in_for_orelse": 1,
                        "executed_in_for_orelse_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            3
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            9,
                            13
                        ],
                        [
                            18,
                            19
                        ],
                        [
                            18,
                            27
                        ],
                        [
                            22,
                            24
                        ],
                        [
                            22,
                            27
                        ],
                        [
                            25,
                            26
                        ],
                        [
                            31,
                            33
                        ],
                        [
                            34,
                            35
                        ],
                        [
                            34,
                            40
                        ],
                        [
                            39,
                            40
                        ],
                        [
                            49,
                            52
                        ],
                        [
                            49,
                            55
                        ],
                        [
                            53,
                            54
                        ],
                        [
                            63,
                            68
                        ],
                        [
                            78,
                            81
                        ],
                        [
                            78,
                            84
                        ],
                        [
                            82,
                            83
                        ]
                    ]
                }
            },
            "AsyncFor": {
                "description": {
                    "nodes": {
                        "subword": 5,
                        "mention": 5,
                        "Module": 1,
                        "AsyncFor": 1,
                        "Call": 3,
                        "Assign": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1,
                        "If": 1,
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1
                    },
                    "edges": {
                        "subword": 6,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "func": 3,
                        "func_rev": 3,
                        "args": 3,
                        "args_rev": 2,
                        "value": 1,
                        "value_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "executed_in_for": 2,
                        "executed_in_for_rev": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "test": 1,
                        "test_rev": 1,
                        "executed_if_true": 2,
                        "executed_if_true_rev": 2,
                        "control_flow": 1,
                        "next": 2,
                        "prev": 2,
                        "executed_in_for_orelse": 1,
                        "executed_in_for_orelse_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            9
                        ],
                        [
                            10,
                            11
                        ],
                        [
                            15,
                            19
                        ],
                        [
                            24,
                            25
                        ],
                        [
                            24,
                            33
                        ],
                        [
                            28,
                            30
                        ],
                        [
                            28,
                            33
                        ],
                        [
                            31,
                            32
                        ],
                        [
                            37,
                            39
                        ],
                        [
                            40,
                            41
                        ],
                        [
                            40,
                            46
                        ],
                        [
                            45,
                            46
                        ],
                        [
                            55,
                            58
                        ],
                        [
                            55,
                            61
                        ],
                        [
                            59,
                            60
                        ],
                        [
                            69,
                            74
                        ],
                        [
                            84,
                            87
                        ],
                        [
                            84,
                            90
                        ],
                        [
                            88,
                            89
                        ]
                    ]
                }
            },
            "Try": {
                "description": {
                    "nodes": {
                        "subword": 6,
                        "mention": 6,
                        "Module": 1,
                        "Assign": 3,
                        "Try": 1,
                        "ExceptHandler": 1,
                        "Call": 1
                    },
                    "edges": {
                        "subword": 6,
                        "value": 3,
                        "value_rev": 3,
                        "targets": 3,
                        "targets_rev": 3,
                        "executed_in_try": 1,
                        "executed_in_try_rev": 1,
                        "type": 1,
                        "type_rev": 1,
                        "executed_with_try_handler": 1,
                        "executed_with_try_handler_rev": 1,
                        "executed_in_try_except": 1,
                        "executed_in_try_except_rev": 1,
                        "func": 1,
                        "func_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "executed_in_try_final": 1,
                        "executed_in_try_final_rev": 1,
                        "executed_in_try_else": 1,
                        "executed_in_try_else_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            8,
                            9
                        ],
                        [
                            8,
                            13
                        ],
                        [
                            12,
                            13
                        ],
                        [
                            21,
                            30
                        ],
                        [
                            40,
                            41
                        ],
                        [
                            40,
                            45
                        ],
                        [
                            44,
                            45
                        ],
                        [
                            55,
                            56
                        ],
                        [
                            55,
                            60
                        ],
                        [
                            59,
                            60
                        ],
                        [
                            73,
                            78
                        ],
                        [
                            73,
                            81
                        ],
                        [
                            79,
                            80
                        ]
                    ]
                }
            },
            "While": {
                "description": {
                    "nodes": {
                        "subword": 5,
                        "mention": 3,
                        "Module": 1,
                        "Compare": 1,
                        "Op": 1,
                        "While": 1,
                        "Call": 1
                    },
                    "edges": {
                        "subword": 5,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "test": 1,
                        "test_rev": 1,
                        "func": 1,
                        "func_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "executed_in_while": 1,
                        "executed_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            6,
                            7
                        ],
                        [
                            6,
                            12
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            17,
                            24
                        ],
                        [
                            17,
                            27
                        ],
                        [
                            25,
                            26
                        ]
                    ]
                }
            },
            "Break": {
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "While": 1,
                        "Module": 1,
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1
                    },
                    "edges": {
                        "test": 1,
                        "control_flow": 1,
                        "executed_in_while": 1,
                        "executed_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            6,
                            10
                        ],
                        [
                            15,
                            20
                        ]
                    ]
                }
            },
            "Continue": {
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "While": 1,
                        "Module": 1,
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1
                    },
                    "edges": {
                        "test": 1,
                        "control_flow": 1,
                        "executed_in_while": 1,
                        "executed_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            5
                        ],
                        [
                            6,
                            10
                        ],
                        [
                            15,
                            23
                        ]
                    ]
                }
            },
            "Pass": {
                "description": {
                    "nodes": {
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1,
                        "Module": 1
                    },
                    "edges": {
                        "control_flow": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            4
                        ]
                    ]
                }
            },
            "Dict": {
                "description": {
                    "nodes": {
                        "subword": 4,
                        "mention": 4,
                        "Module": 1,
                        "Dict": 1
                    },
                    "edges": {
                        "subword": 4,
                        "keys": 2,
                        "keys_rev": 2,
                        "values": 2,
                        "values_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            1,
                            2
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            6,
                            7
                        ],
                        [
                            8,
                            9
                        ]
                    ]
                }
            },
            "Set": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "Set": 1
                    },
                    "edges": {
                        "subword": 2,
                        "elts": 2,
                        "elts_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            1,
                            2
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "ListComp": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "ListComp": 1,
                        "comprehension": 1
                    },
                    "edges": {
                        "subword": 2,
                        "elt": 1,
                        "elt_rev": 1,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ]
                    ]
                }
            },
            "DictComp": {
                "description": {
                    "nodes": {
                        "subword": 3,
                        "mention": 3,
                        "Module": 1,
                        "DictComp": 1,
                        "Tuple": 1,
                        "comprehension": 1
                    },
                    "edges": {
                        "subword": 3,
                        "key": 1,
                        "key_rev": 1,
                        "value": 1,
                        "value_rev": 1,
                        "elts": 2,
                        "elts_rev": 2,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            1,
                            2
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            9,
                            10
                        ],
                        [
                            9,
                            12
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            16,
                            20
                        ]
                    ]
                }
            },
            "SetComp": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "SetComp": 1,
                        "comprehension": 1
                    },
                    "edges": {
                        "subword": 2,
                        "elt": 1,
                        "elt_rev": 1,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ]
                    ]
                }
            },
            "GeneratorExp": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "GeneratorExp": 1,
                        "comprehension": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1
                    },
                    "edges": {
                        "subword": 2,
                        "elt": 1,
                        "elt_rev": 1,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "ifs": 1,
                        "ifs_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            27
                        ],
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ],
                        [
                            20,
                            21
                        ],
                        [
                            25,
                            26
                        ]
                    ]
                }
            },
            "BinOp": {
                "description": {
                    "nodes": {
                        "subword": 3,
                        "mention": 3,
                        "Module": 1,
                        "BinOp": 1,
                        "Op": 1,
                        "Assign": 1
                    },
                    "edges": {
                        "subword": 3,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "value": 1,
                        "value_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            9
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            4,
                            9
                        ],
                        [
                            8,
                            9
                        ]
                    ]
                }
            },
            "ImportFrom": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "ImportFrom": 1,
                        "alias": 1
                    },
                    "edges": {
                        "subword": 2,
                        "module": 1,
                        "module_rev": 1,
                        "name": 1,
                        "name_rev": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            4
                        ]
                    ]
                }
            },
            "alias": {
                "description": {
                    "nodes": {
                        "subword": 2,
                        "mention": 2,
                        "Module": 1,
                        "alias": 1,
                        "Import": 1
                    },
                    "edges": {
                        "subword": 2,
                        "name": 1,
                        "name_rev": 1,
                        "asname": 1,
                        "asname_rev": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            6
                        ]
                    ]
                }
            },
            "List": {
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "List": 1,
                        "Module": 1,
                        "Assign": 1,
                        "subword": 1,
                        "mention": 1
                    },
                    "edges": {
                        "elts": 1,
                        "value": 1,
                        "value_rev": 1,
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            16
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            8,
                            9
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "Tuple": {
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "Tuple": 1,
                        "Module": 1,
                        "Assign": 1,
                        "subword": 1,
                        "mention": 1
                    },
                    "edges": {
                        "elts": 1,
                        "value": 1,
                        "value_rev": 1,
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            16
                        ],
                        [
                            4,
                            16
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            8,
                            9
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "JoinedStr": {
                "description": {
                    "nodes": {
                        "JoinedStr": 1,
                        "Module": 1
                    },
                    "edges": {
                        "defined_in_module": 1
                    },
                    "offsets": [
                        [
                            0,
                            6
                        ]
                    ]
                }
            },
            "FormattedValue": {
                "description": {
                    "nodes": {
                        "JoinedStr": 1,
                        "Module": 1
                    },
                    "edges": {
                        "defined_in_module": 1
                    },
                    "offsets": [
                        [
                            0,
                            11
                        ]
                    ]
                }
            },
            "Bytes": {
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "Assign": 1,
                        "Module": 1,
                        "subword": 1,
                        "mention": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            10
                        ],
                        [
                            4,
                            10
                        ]
                    ]
                }
            },
            "Num": {
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "Assign": 1,
                        "Module": 1,
                        "subword": 1,
                        "mention": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Str": {
                "description": {
                    "nodes": {
                        "Constant": 1,
                        "Assign": 1,
                        "Module": 1,
                        "subword": 1,
                        "mention": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            9
                        ],
                        [
                            4,
                            9
                        ]
                    ]
                }
            },
            "FunctionDef2": {
                "description": {
                    "nodes": {
                        "subword": 20,
                        "mention": 10,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 8,
                        "arguments": 1,
                        "type_annotation": 3,
                        "Constant": 3,
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1
                    },
                    "edges": {
                        "subword": 28,
                        "arg": 12,
                        "arg_rev": 12,
                        "posonlyarg": 1,
                        "posonlyarg_rev": 1,
                        "annotation_for": 2,
                        "default": 4,
                        "kwonlyarg": 2,
                        "kwonlyarg_rev": 2,
                        "kwarg": 1,
                        "kwarg_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "decorator_list": 1,
                        "decorator_list_rev": 1,
                        "returned_by": 1,
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            11,
                            14
                        ],
                        [
                            37,
                            41
                        ],
                        [
                            53,
                            57
                        ],
                        [
                            66,
                            74
                        ],
                        [
                            71,
                            74
                        ],
                        [
                            91,
                            123
                        ],
                        [
                            109,
                            123
                        ],
                        [
                            125,
                            133
                        ],
                        [
                            142,
                            146
                        ],
                        [
                            147,
                            148
                        ],
                        [
                            159,
                            163
                        ],
                        [
                            164,
                            168
                        ],
                        [
                            177,
                            181
                        ],
                        [
                            182,
                            183
                        ],
                        [
                            194,
                            200
                        ],
                        [
                            206,
                            218
                        ],
                        [
                            223,
                            227
                        ]
                    ]
                }
            },
            "FunctionDef3": {
                "description": {
                    "nodes": {
                        "subword": 6,
                        "mention": 4,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 2,
                        "arguments": 1,
                        "CtlFlow": 1,
                        "CtlFlowInstance": 1
                    },
                    "edges": {
                        "subword": 6,
                        "arg": 2,
                        "arg_rev": 2,
                        "vararg": 1,
                        "vararg_rev": 1,
                        "kwarg": 1,
                        "kwarg_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "decorator_list": 1,
                        "decorator_list_rev": 1,
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            11,
                            14
                        ],
                        [
                            30,
                            34
                        ],
                        [
                            38,
                            44
                        ],
                        [
                            50,
                            54
                        ]
                    ]
                }
            }
        },
        "v1.0_control_flow": {
            "Assign": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign1": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign2": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign3": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign4": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign5": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign6": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign7": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign8": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign9": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign10": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign11": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign12": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AugAssign13": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 6,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Delete": {
                "description": {
                    "nodes": {
                        "Name": 3,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 3,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Global": {
                "description": {
                    "nodes": {
                        "Name": 3,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 3,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Nonlocal": {
                "description": {
                    "nodes": {
                        "Name": 3,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 3,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Slice": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 6,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "ExtSlice": {
                "description": {
                    "nodes": {
                        "Name": 8,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 8,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Index": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Starred": {
                "description": {
                    "nodes": {
                        "Name": 2,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Yield": {
                "description": {
                    "nodes": {
                        "Name": 3,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 3,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "YieldFrom": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare1": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare2": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare3": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare4": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare5": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare6": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare7": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare8": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare9": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Compare10": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp1": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp2": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp3": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp4": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp5": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp6": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp7": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp8": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp9": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp10": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp11": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp12": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BoolOp1": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BoolOp2": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BoolOp3": {
                "description": {
                    "nodes": {
                        "Name": 3,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 3,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "UnaryOp1": {
                "description": {
                    "nodes": {
                        "Name": 2,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "UnaryOp2": {
                "description": {
                    "nodes": {
                        "Name": 2,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "UnaryOp3": {
                "description": {
                    "nodes": {
                        "Name": 2,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Assert": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "FunctionDef": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 3,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 1,
                        "arguments": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "arg": 1,
                        "arg_rev": 1,
                        "args": 2,
                        "args_rev": 2,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AsyncFunctionDef": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 3,
                        "AsyncFunctionDef": 1,
                        "Module": 1,
                        "arg": 1,
                        "arguments": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "arg": 1,
                        "arg_rev": 1,
                        "args": 2,
                        "args_rev": 2,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            12,
                            13
                        ]
                    ]
                }
            },
            "ClassDef": {
                "description": {
                    "nodes": {
                        "Name": 3,
                        "mention": 3,
                        "FunctionDef": 1,
                        "ClassDef": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 3,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_class": 1,
                        "defined_in_class_rev": 1,
                        "class_name": 1,
                        "class_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "AnnAssign": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 6,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "With": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 3,
                        "Module": 1,
                        "withitem": 1,
                        "With": 1
                    },
                    "edges": {
                        "local_mention": 9,
                        "context_expr": 1,
                        "context_expr_rev": 1,
                        "optional_vars": 1,
                        "optional_vars_rev": 1,
                        "items": 1,
                        "items_rev": 1,
                        "defined_in_with": 1,
                        "defined_in_with_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            5,
                            12
                        ],
                        [
                            16,
                            17
                        ]
                    ]
                }
            },
            "AsyncWith": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 3,
                        "Module": 1,
                        "withitem": 1,
                        "AsyncWith": 1
                    },
                    "edges": {
                        "local_mention": 9,
                        "context_expr": 1,
                        "context_expr_rev": 1,
                        "optional_vars": 1,
                        "optional_vars_rev": 1,
                        "items": 1,
                        "items_rev": 1,
                        "defined_in_with": 1,
                        "defined_in_with_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            11,
                            18
                        ],
                        [
                            22,
                            23
                        ]
                    ]
                }
            },
            "arg": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 3,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 1,
                        "type_annotation": 1,
                        "arguments": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "arg": 1,
                        "arg_rev": 1,
                        "annotation_for": 1,
                        "args": 2,
                        "args_rev": 2,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            12
                        ],
                        [
                            9,
                            12
                        ]
                    ]
                }
            },
            "Await": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Raise": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Lambda": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 6,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "IfExp": {
                "description": {
                    "nodes": {
                        "Name": 8,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 8,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "keyword": {
                "description": {
                    "nodes": {
                        "Name": 10,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 10,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Attribute": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "If": {
                "description": {
                    "nodes": {
                        "Name": 10,
                        "mention": 5,
                        "Module": 1,
                        "If": 2
                    },
                    "edges": {
                        "local_mention": 21,
                        "test": 2,
                        "test_rev": 2,
                        "defined_in_if_true": 2,
                        "defined_in_if_true_rev": 2,
                        "defined_in_if_false": 2,
                        "defined_in_if_false_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            3,
                            12
                        ],
                        [
                            28,
                            38
                        ]
                    ]
                }
            },
            "For": {
                "description": {
                    "nodes": {
                        "Name": 12,
                        "mention": 7,
                        "Module": 1,
                        "For": 1,
                        "If": 1
                    },
                    "edges": {
                        "local_mention": 22,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "defined_in_for": 2,
                        "defined_in_for_rev": 2,
                        "test": 1,
                        "test_rev": 1,
                        "defined_in_if_true": 2,
                        "defined_in_if_true_rev": 2,
                        "next": 2,
                        "prev": 2,
                        "defined_in_for_orelse": 1,
                        "defined_in_for_orelse_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            4,
                            5
                        ],
                        [
                            9,
                            13
                        ],
                        [
                            34,
                            40
                        ]
                    ]
                }
            },
            "AsyncFor": {
                "description": {
                    "nodes": {
                        "Name": 12,
                        "mention": 7,
                        "Module": 1,
                        "AsyncFor": 1,
                        "If": 1
                    },
                    "edges": {
                        "local_mention": 22,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "defined_in_for": 2,
                        "defined_in_for_rev": 2,
                        "test": 1,
                        "test_rev": 1,
                        "defined_in_if_true": 2,
                        "defined_in_if_true_rev": 2,
                        "next": 2,
                        "prev": 2,
                        "defined_in_for_orelse": 1,
                        "defined_in_for_orelse_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            10,
                            11
                        ],
                        [
                            15,
                            19
                        ],
                        [
                            40,
                            46
                        ]
                    ]
                }
            },
            "Try": {
                "description": {
                    "nodes": {
                        "Name": 10,
                        "mention": 5,
                        "Module": 1,
                        "Try": 1,
                        "ExceptHandler": 1
                    },
                    "edges": {
                        "local_mention": 17,
                        "defined_in_try": 1,
                        "defined_in_try_rev": 1,
                        "type": 1,
                        "type_rev": 1,
                        "defined_in_try_handler": 1,
                        "defined_in_try_handler_rev": 1,
                        "try_except": 1,
                        "defined_in_try_final": 1,
                        "defined_in_try_final_rev": 1,
                        "defined_in_try_else": 1,
                        "defined_in_try_else_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            21,
                            30
                        ]
                    ]
                }
            },
            "While": {
                "description": {
                    "nodes": {
                        "Name": 7,
                        "mention": 2,
                        "Module": 1,
                        "While": 1
                    },
                    "edges": {
                        "local_mention": 8,
                        "test": 1,
                        "test_rev": 1,
                        "defined_in_while": 1,
                        "defined_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            12
                        ]
                    ]
                }
            },
            "Break": {
                "description": {
                    "nodes": {
                        "Name": 2,
                        "mention": 2,
                        "Module": 1,
                        "While": 1
                    },
                    "edges": {
                        "local_mention": 2,
                        "test": 1,
                        "test_rev": 1,
                        "defined_in_while": 1,
                        "defined_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ]
                    ]
                }
            },
            "Continue": {
                "description": {
                    "nodes": {
                        "Name": 2,
                        "mention": 2,
                        "Module": 1,
                        "While": 1
                    },
                    "edges": {
                        "local_mention": 2,
                        "test": 1,
                        "test_rev": 1,
                        "defined_in_while": 1,
                        "defined_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ]
                    ]
                }
            },
            "Pass": {
                "description": {
                    "nodes": {
                        "Name": 1,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Dict": {
                "description": {
                    "nodes": {
                        "Name": 9,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 9,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Set": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 6,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "ListComp": {
                "description": {
                    "nodes": {
                        "Name": 7,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 7,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "DictComp": {
                "description": {
                    "nodes": {
                        "Name": 10,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 10,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "SetComp": {
                "description": {
                    "nodes": {
                        "Name": 7,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 7,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "GeneratorExp": {
                "description": {
                    "nodes": {
                        "Name": 11,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 11,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "BinOp": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 6,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "ImportFrom": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "alias": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "List": {
                "description": {
                    "nodes": {
                        "Name": 10,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 10,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Tuple": {
                "description": {
                    "nodes": {
                        "Name": 10,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 10,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "JoinedStr": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "FormattedValue": {
                "description": {
                    "nodes": {
                        "Name": 8,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 8,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Bytes": {
                "description": {
                    "nodes": {
                        "Name": 6,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 6,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Num": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Str": {
                "description": {
                    "nodes": {
                        "Name": 5,
                        "mention": 1,
                        "Module": 1
                    },
                    "edges": {
                        "local_mention": 5,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "FunctionDef2": {
                "description": {
                    "nodes": {
                        "Name": 7,
                        "mention": 7,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 4,
                        "arguments": 1,
                        "type_annotation": 3
                    },
                    "edges": {
                        "local_mention": 7,
                        "arg": 4,
                        "arg_rev": 4,
                        "args": 5,
                        "args_rev": 5,
                        "annotation_for": 2,
                        "decorator_list": 1,
                        "decorator_list_rev": 1,
                        "returned_by": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            53,
                            57
                        ],
                        [
                            66,
                            74
                        ],
                        [
                            71,
                            74
                        ],
                        [
                            91,
                            123
                        ],
                        [
                            109,
                            123
                        ],
                        [
                            142,
                            146
                        ],
                        [
                            206,
                            218
                        ]
                    ]
                }
            },
            "FunctionDef3": {
                "description": {
                    "nodes": {
                        "Name": 4,
                        "mention": 4,
                        "FunctionDef": 1,
                        "Module": 1,
                        "arg": 1,
                        "arguments": 1
                    },
                    "edges": {
                        "local_mention": 4,
                        "arg": 1,
                        "arg_rev": 1,
                        "vararg": 1,
                        "vararg_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "decorator_list": 1,
                        "decorator_list_rev": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            30,
                            34
                        ]
                    ]
                }
            }
        },
        "v3.5": {
            "Assign": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "AugAssign1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign4": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign5": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign6": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign7": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            9
                        ]
                    ]
                }
            },
            "AugAssign8": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            9
                        ]
                    ]
                }
            },
            "AugAssign9": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign10": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign11": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign12": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign13": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Delete": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Delete": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Global": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Global": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Nonlocal": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Nonlocal": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "Slice": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Subscript": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Slice": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 2,
                        "value": 1,
                        "value_rev": 1,
                        "lower": 1,
                        "upper": 1,
                        "slice": 1,
                        "slice_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            2,
                            3
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "ExtSlice": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Subscript": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "ExtSlice": 1,
                        "Index": 1,
                        "Constant": 1,
                        "Slice": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 2,
                        "value": 2,
                        "value_rev": 1,
                        "dims": 2,
                        "dims_rev": 2,
                        "lower": 1,
                        "upper": 1,
                        "slice": 1,
                        "slice_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            2,
                            3
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "Index": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Subscript": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Index": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 2,
                        "value": 2,
                        "value_rev": 1,
                        "slice": 1,
                        "slice_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            2,
                            3
                        ]
                    ]
                }
            },
            "Starred": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Starred": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 2,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "Yield": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Yield": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "YieldFrom": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "YieldFrom": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            11,
                            12
                        ]
                    ]
                }
            },
            "Compare1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Compare4": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Compare5": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare6": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare7": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare8": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "Compare9": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare10": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "BinOp1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp4": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp5": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp6": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp7": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp8": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp9": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp10": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp11": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp12": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BoolOp1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BoolOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "values": 2,
                        "values_rev": 2,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "BoolOp2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BoolOp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "values": 2,
                        "values_rev": 2,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BoolOp3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "UnaryOp1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "UnaryOp2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "UnaryOp3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "operand": 1,
                        "operand_rev": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "Assert": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assert": 1,
                        "Compare": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "test": 1,
                        "test_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            7,
                            8
                        ],
                        [
                            7,
                            13
                        ],
                        [
                            12,
                            13
                        ]
                    ]
                }
            },
            "FunctionDef": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "arguments": 1,
                        "arg": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 3,
                        "Return": 1,
                        "ctx": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "instance_rev": 3,
                        "arg": 2,
                        "arg_rev": 2,
                        "args": 1,
                        "args_rev": 1,
                        "ctx": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            7
                        ],
                        [
                            20,
                            21
                        ]
                    ]
                }
            },
            "AsyncFunctionDef": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AsyncFunctionDef": 1,
                        "arguments": 1,
                        "arg": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 3,
                        "Return": 1,
                        "ctx": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "instance_rev": 3,
                        "arg": 2,
                        "arg_rev": 2,
                        "args": 1,
                        "args_rev": 1,
                        "ctx": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            12,
                            13
                        ],
                        [
                            26,
                            27
                        ]
                    ]
                }
            },
            "ClassDef": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "ClassDef": 1,
                        "FunctionDef": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "subword": 2
                    },
                    "edges": {
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_class": 1,
                        "defined_in_class_rev": 1,
                        "class_name": 1,
                        "class_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            28,
                            32
                        ]
                    ]
                }
            },
            "AnnAssign": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "type_annotation": 1,
                        "AnnAssign": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Constant": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "target": 1,
                        "target_rev": 1,
                        "value": 1,
                        "annotation_for": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            3,
                            6
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "With": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "With": 1,
                        "withitem": 1,
                        "Call": 2,
                        "mention": 4,
                        "Name": 4,
                        "instance": 5,
                        "ctx": 2,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 5,
                        "instance_rev": 5,
                        "ctx": 5,
                        "func": 2,
                        "func_rev": 2,
                        "args": 2,
                        "args_rev": 2,
                        "context_expr": 1,
                        "context_expr_rev": 1,
                        "optional_vars": 1,
                        "optional_vars_rev": 1,
                        "items": 1,
                        "items_rev": 1,
                        "executed_inside_with": 1,
                        "executed_inside_with_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            5,
                            9
                        ],
                        [
                            5,
                            12
                        ],
                        [
                            10,
                            11
                        ],
                        [
                            16,
                            17
                        ],
                        [
                            22,
                            30
                        ],
                        [
                            31,
                            32
                        ]
                    ]
                }
            },
            "AsyncWith": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AsyncWith": 1,
                        "withitem": 1,
                        "Call": 2,
                        "mention": 4,
                        "Name": 4,
                        "instance": 5,
                        "ctx": 2,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 5,
                        "instance_rev": 5,
                        "ctx": 5,
                        "func": 2,
                        "func_rev": 2,
                        "args": 2,
                        "args_rev": 2,
                        "context_expr": 1,
                        "context_expr_rev": 1,
                        "optional_vars": 1,
                        "optional_vars_rev": 1,
                        "items": 1,
                        "items_rev": 1,
                        "executed_inside_with": 1,
                        "executed_inside_with_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            11,
                            15
                        ],
                        [
                            11,
                            18
                        ],
                        [
                            16,
                            17
                        ],
                        [
                            22,
                            23
                        ],
                        [
                            28,
                            36
                        ],
                        [
                            37,
                            38
                        ]
                    ]
                }
            },
            "arg": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "arguments": 1,
                        "arg": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 3,
                        "type_annotation": 1,
                        "Constant": 1,
                        "Return": 1,
                        "ctx": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 3,
                        "instance_rev": 3,
                        "arg": 2,
                        "arg_rev": 2,
                        "annotation_for": 1,
                        "default": 1,
                        "args": 1,
                        "args_rev": 1,
                        "ctx": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            12
                        ],
                        [
                            9,
                            12
                        ],
                        [
                            15,
                            16
                        ],
                        [
                            29,
                            30
                        ]
                    ]
                }
            },
            "Await": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Await": 1,
                        "Call": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "func": 1,
                        "func_rev": 1,
                        "value": 1,
                        "value_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ],
                        [
                            6,
                            12
                        ]
                    ]
                }
            },
            "Raise": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Raise": 1,
                        "Call": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "func": 1,
                        "func_rev": 1,
                        "exc": 1,
                        "exc_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            15
                        ],
                        [
                            6,
                            17
                        ]
                    ]
                }
            },
            "Lambda": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Lambda": 1,
                        "BinOp": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Constant": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "op": 1,
                        "lambda": 1,
                        "lambda_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            10,
                            11
                        ],
                        [
                            10,
                            15
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "IfExp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "IfExp": 1,
                        "Constant": 2,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "test": 1,
                        "if_true": 1,
                        "if_false": 1,
                        "value": 1,
                        "value_rev": 1,
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            4,
                            20
                        ],
                        [
                            9,
                            13
                        ],
                        [
                            19,
                            20
                        ]
                    ]
                }
            },
            "keyword": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Call": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "keyword": 2,
                        "#keyword#": 2,
                        "Constant": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "func": 1,
                        "func_rev": 1,
                        "arg": 2,
                        "value": 2,
                        "keywords": 2,
                        "keywords_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            2
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            10,
                            11
                        ]
                    ]
                }
            },
            "Attribute": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Attribute": 2,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "#attr#": 2,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 3,
                        "value": 2,
                        "value_rev": 2,
                        "attr": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            0,
                            3
                        ]
                    ]
                }
            },
            "If": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "If": 2,
                        "Compare": 2,
                        "mention": 4,
                        "Name": 4,
                        "instance": 10,
                        "ctx": 2,
                        "Op": 1,
                        "Constant": 1,
                        "Assign": 3,
                        "Tuple": 2,
                        "subword": 4
                    },
                    "edges": {
                        "subword": 4,
                        "instance": 10,
                        "instance_rev": 10,
                        "ctx": 12,
                        "left": 2,
                        "left_rev": 2,
                        "ops": 2,
                        "comparators": 2,
                        "test": 2,
                        "test_rev": 2,
                        "value": 3,
                        "value_rev": 3,
                        "targets": 3,
                        "targets_rev": 3,
                        "executed_if_true": 2,
                        "executed_if_true_rev": 2,
                        "elts": 4,
                        "elts_rev": 4,
                        "executed_if_false": 2,
                        "executed_if_false_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            3,
                            4
                        ],
                        [
                            3,
                            12
                        ],
                        [
                            8,
                            12
                        ],
                        [
                            17,
                            18
                        ],
                        [
                            21,
                            22
                        ],
                        [
                            28,
                            29
                        ],
                        [
                            28,
                            38
                        ],
                        [
                            33,
                            38
                        ],
                        [
                            42,
                            43
                        ],
                        [
                            46,
                            47
                        ],
                        [
                            57,
                            58
                        ],
                        [
                            57,
                            61
                        ],
                        [
                            60,
                            61
                        ],
                        [
                            64,
                            65
                        ],
                        [
                            64,
                            68
                        ],
                        [
                            67,
                            68
                        ]
                    ]
                }
            },
            "For": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "For": 1,
                        "mention": 5,
                        "Name": 5,
                        "instance": 9,
                        "ctx": 2,
                        "Assign": 1,
                        "Call": 3,
                        "If": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 5
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 9,
                        "instance_rev": 9,
                        "ctx": 9,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "func": 3,
                        "func_rev": 3,
                        "args": 3,
                        "args_rev": 2,
                        "value": 1,
                        "value_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "executed_in_for": 2,
                        "executed_in_for_rev": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "test": 1,
                        "test_rev": 1,
                        "executed_if_true": 2,
                        "executed_if_true_rev": 2,
                        "control_flow": 1,
                        "next": 2,
                        "prev": 2,
                        "executed_in_for_orelse": 1,
                        "executed_in_for_orelse_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            4,
                            5
                        ],
                        [
                            9,
                            13
                        ],
                        [
                            18,
                            19
                        ],
                        [
                            22,
                            24
                        ],
                        [
                            22,
                            27
                        ],
                        [
                            25,
                            26
                        ],
                        [
                            34,
                            35
                        ],
                        [
                            34,
                            40
                        ],
                        [
                            39,
                            40
                        ],
                        [
                            49,
                            52
                        ],
                        [
                            53,
                            54
                        ],
                        [
                            63,
                            68
                        ],
                        [
                            78,
                            81
                        ],
                        [
                            82,
                            83
                        ]
                    ]
                }
            },
            "AsyncFor": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AsyncFor": 1,
                        "mention": 5,
                        "Name": 5,
                        "instance": 9,
                        "ctx": 2,
                        "Assign": 1,
                        "Call": 3,
                        "If": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 5
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 9,
                        "instance_rev": 9,
                        "ctx": 9,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "func": 3,
                        "func_rev": 3,
                        "args": 3,
                        "args_rev": 2,
                        "value": 1,
                        "value_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "executed_in_for": 2,
                        "executed_in_for_rev": 2,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "test": 1,
                        "test_rev": 1,
                        "executed_if_true": 2,
                        "executed_if_true_rev": 2,
                        "control_flow": 1,
                        "next": 2,
                        "prev": 2,
                        "executed_in_for_orelse": 1,
                        "executed_in_for_orelse_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            10,
                            11
                        ],
                        [
                            15,
                            19
                        ],
                        [
                            24,
                            25
                        ],
                        [
                            28,
                            30
                        ],
                        [
                            28,
                            33
                        ],
                        [
                            31,
                            32
                        ],
                        [
                            40,
                            41
                        ],
                        [
                            40,
                            46
                        ],
                        [
                            45,
                            46
                        ],
                        [
                            55,
                            58
                        ],
                        [
                            59,
                            60
                        ],
                        [
                            69,
                            74
                        ],
                        [
                            84,
                            87
                        ],
                        [
                            88,
                            89
                        ]
                    ]
                }
            },
            "Try": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Try": 1,
                        "Assign": 3,
                        "mention": 6,
                        "Name": 6,
                        "instance": 9,
                        "ctx": 2,
                        "ExceptHandler": 1,
                        "Call": 1,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 9,
                        "instance_rev": 9,
                        "ctx": 9,
                        "value": 3,
                        "value_rev": 3,
                        "targets": 3,
                        "targets_rev": 3,
                        "executed_in_try": 1,
                        "executed_in_try_rev": 1,
                        "type": 1,
                        "type_rev": 1,
                        "executed_with_try_handler": 1,
                        "executed_with_try_handler_rev": 1,
                        "executed_in_try_except": 1,
                        "executed_in_try_except_rev": 1,
                        "func": 1,
                        "func_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "executed_in_try_final": 1,
                        "executed_in_try_final_rev": 1,
                        "executed_in_try_else": 1,
                        "executed_in_try_else_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            8,
                            9
                        ],
                        [
                            12,
                            13
                        ],
                        [
                            21,
                            30
                        ],
                        [
                            40,
                            41
                        ],
                        [
                            44,
                            45
                        ],
                        [
                            55,
                            56
                        ],
                        [
                            59,
                            60
                        ],
                        [
                            73,
                            78
                        ],
                        [
                            79,
                            80
                        ]
                    ]
                }
            },
            "While": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "While": 1,
                        "Compare": 1,
                        "mention": 3,
                        "Name": 3,
                        "instance": 4,
                        "ctx": 1,
                        "Op": 1,
                        "Call": 1,
                        "subword": 5
                    },
                    "edges": {
                        "subword": 5,
                        "instance": 4,
                        "instance_rev": 4,
                        "ctx": 4,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "comparators_rev": 1,
                        "test": 1,
                        "test_rev": 1,
                        "func": 1,
                        "func_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "executed_in_while": 1,
                        "executed_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            7
                        ],
                        [
                            6,
                            12
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            17,
                            24
                        ],
                        [
                            25,
                            26
                        ]
                    ]
                }
            },
            "Break": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "While": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1
                    },
                    "edges": {
                        "test": 1,
                        "control_flow": 1,
                        "executed_in_while": 1,
                        "executed_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ],
                        [
                            15,
                            20
                        ]
                    ]
                }
            },
            "Continue": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "While": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1
                    },
                    "edges": {
                        "test": 1,
                        "control_flow": 1,
                        "executed_in_while": 1,
                        "executed_in_while_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ],
                        [
                            15,
                            23
                        ]
                    ]
                }
            },
            "Pass": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1
                    },
                    "edges": {
                        "control_flow": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            4
                        ]
                    ]
                }
            },
            "Dict": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Dict": 1,
                        "mention": 4,
                        "Name": 4,
                        "instance": 4,
                        "ctx": 1,
                        "subword": 4
                    },
                    "edges": {
                        "subword": 4,
                        "instance": 4,
                        "instance_rev": 4,
                        "ctx": 4,
                        "keys": 2,
                        "keys_rev": 2,
                        "values": 2,
                        "values_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            6,
                            7
                        ],
                        [
                            8,
                            9
                        ]
                    ]
                }
            },
            "Set": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Set": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "ctx": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "ctx": 2,
                        "elts": 2,
                        "elts_rev": 2,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "ListComp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "ListComp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 3,
                        "ctx": 2,
                        "comprehension": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "instance_rev": 3,
                        "ctx": 3,
                        "elt": 1,
                        "elt_rev": 1,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ]
                    ]
                }
            },
            "DictComp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "DictComp": 1,
                        "mention": 3,
                        "Name": 3,
                        "instance": 5,
                        "ctx": 2,
                        "comprehension": 1,
                        "Tuple": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 5,
                        "instance_rev": 5,
                        "ctx": 6,
                        "key": 1,
                        "key_rev": 1,
                        "value": 1,
                        "value_rev": 1,
                        "elts": 2,
                        "elts_rev": 2,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            9,
                            10
                        ],
                        [
                            9,
                            12
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            16,
                            20
                        ]
                    ]
                }
            },
            "SetComp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "SetComp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 3,
                        "ctx": 2,
                        "comprehension": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "instance_rev": 3,
                        "ctx": 3,
                        "elt": 1,
                        "elt_rev": 1,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ]
                    ]
                }
            },
            "GeneratorExp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "GeneratorExp": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 4,
                        "ctx": 2,
                        "comprehension": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 4,
                        "instance_rev": 4,
                        "ctx": 4,
                        "elt": 1,
                        "elt_rev": 1,
                        "target": 1,
                        "target_rev": 1,
                        "iter": 1,
                        "iter_rev": 1,
                        "left": 1,
                        "left_rev": 1,
                        "ops": 1,
                        "comparators": 1,
                        "ifs": 1,
                        "ifs_rev": 1,
                        "generators": 1,
                        "generators_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ],
                        [
                            20,
                            21
                        ],
                        [
                            20,
                            26
                        ],
                        [
                            25,
                            26
                        ]
                    ]
                }
            },
            "BinOp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "BinOp": 1,
                        "mention": 3,
                        "Name": 3,
                        "instance": 3,
                        "ctx": 2,
                        "Op": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 3,
                        "instance_rev": 3,
                        "ctx": 3,
                        "left": 1,
                        "left_rev": 1,
                        "right": 1,
                        "right_rev": 1,
                        "op": 1,
                        "value": 1,
                        "value_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            4,
                            9
                        ],
                        [
                            8,
                            9
                        ]
                    ]
                }
            },
            "ImportFrom": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "ImportFrom": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "alias": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "module": 1,
                        "module_rev": 1,
                        "name": 1,
                        "name_rev": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "alias": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Import": 1,
                        "alias": 1,
                        "mention": 2,
                        "Name": 2,
                        "instance": 2,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "instance_rev": 2,
                        "name": 1,
                        "name_rev": 1,
                        "asname": 1,
                        "asname_rev": 1,
                        "names": 1,
                        "names_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    }
                }
            },
            "List": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "List": 1,
                        "Constant": 1,
                        "ctx": 2,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "elts": 1,
                        "ctx": 2,
                        "value": 1,
                        "value_rev": 1,
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            8,
                            9
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "Tuple": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Tuple": 1,
                        "Constant": 1,
                        "ctx": 2,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "elts": 1,
                        "ctx": 2,
                        "value": 1,
                        "value_rev": 1,
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            16
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            8,
                            9
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "JoinedStr": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "JoinedStr": 1
                    },
                    "edges": {
                        "defined_in_module": 1
                    }
                }
            },
            "FormattedValue": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "JoinedStr": 1
                    },
                    "edges": {
                        "defined_in_module": 1
                    }
                }
            },
            "Bytes": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            10
                        ]
                    ]
                }
            },
            "Num": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Str": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "Name": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "instance_rev": 1,
                        "ctx": 1,
                        "targets": 1,
                        "targets_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            9
                        ]
                    ]
                }
            },
            "FunctionDef2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "arguments": 1,
                        "arg": 8,
                        "mention": 10,
                        "Name": 10,
                        "instance": 10,
                        "type_annotation": 3,
                        "Constant": 3,
                        "ctx": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 20
                    },
                    "edges": {
                        "subword": 28,
                        "instance": 10,
                        "instance_rev": 10,
                        "arg": 12,
                        "arg_rev": 12,
                        "posonlyarg": 1,
                        "posonlyarg_rev": 1,
                        "annotation_for": 2,
                        "default": 4,
                        "kwonlyarg": 2,
                        "kwonlyarg_rev": 2,
                        "kwarg": 1,
                        "kwarg_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "ctx": 1,
                        "decorator_list": 1,
                        "decorator_list_rev": 1,
                        "returned_by": 1,
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            37,
                            41
                        ],
                        [
                            53,
                            57
                        ],
                        [
                            66,
                            74
                        ],
                        [
                            71,
                            74
                        ],
                        [
                            91,
                            123
                        ],
                        [
                            109,
                            123
                        ],
                        [
                            125,
                            133
                        ],
                        [
                            142,
                            146
                        ],
                        [
                            147,
                            148
                        ],
                        [
                            159,
                            163
                        ],
                        [
                            164,
                            168
                        ],
                        [
                            177,
                            181
                        ],
                        [
                            182,
                            183
                        ],
                        [
                            194,
                            200
                        ],
                        [
                            206,
                            218
                        ],
                        [
                            223,
                            227
                        ]
                    ]
                }
            },
            "FunctionDef3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "arguments": 1,
                        "arg": 2,
                        "mention": 4,
                        "Name": 4,
                        "instance": 4,
                        "ctx": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 4,
                        "instance_rev": 4,
                        "arg": 2,
                        "arg_rev": 2,
                        "vararg": 1,
                        "vararg_rev": 1,
                        "kwarg": 1,
                        "kwarg_rev": 1,
                        "args": 1,
                        "args_rev": 1,
                        "ctx": 1,
                        "decorator_list": 1,
                        "decorator_list_rev": 1,
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "defined_in_function_rev": 1,
                        "function_name": 1,
                        "function_name_rev": 1,
                        "defined_in_module": 1,
                        "defined_in_module_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            30,
                            34
                        ],
                        [
                            38,
                            44
                        ],
                        [
                            50,
                            54
                        ]
                    ]
                }
            }
        },
        "v3.5_control_flow": {
            "Assign": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "AugAssign1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign4": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign5": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign6": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign7": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            9
                        ]
                    ]
                }
            },
            "AugAssign8": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            9
                        ]
                    ]
                }
            },
            "AugAssign9": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign10": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "AugAssign11": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign12": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "AugAssign13": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AugAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "op": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Delete": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Delete": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Global": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Global": 1,
                        "mention": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "names": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "names_rev": 1
                    }
                }
            },
            "Nonlocal": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Nonlocal": 1,
                        "mention": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "names": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "names_rev": 1
                    }
                }
            },
            "Slice": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Subscript": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 2,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "ExtSlice": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Subscript": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 2,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "Index": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Subscript": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Constant": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 2,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            2,
                            3
                        ]
                    ]
                }
            },
            "Starred": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Starred": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 2,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "value_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "Yield": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Yield": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "value_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "YieldFrom": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "YieldFrom": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "value_rev": 1
                    },
                    "offsets": [
                        [
                            11,
                            12
                        ]
                    ]
                }
            },
            "Compare1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Compare4": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Compare5": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare6": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare7": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare8": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "Compare9": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "Compare10": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "BinOp1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp4": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp5": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp6": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp7": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp8": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp9": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "BinOp10": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp11": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BinOp12": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BinOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "left_rev": 1,
                        "right_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BoolOp1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BoolOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "values": 2,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "values_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            6,
                            7
                        ]
                    ]
                }
            },
            "BoolOp2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "BoolOp": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "values": 2,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "values_rev": 2
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            5,
                            6
                        ]
                    ]
                }
            },
            "BoolOp3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "operand": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "operand_rev": 1
                    },
                    "offsets": [
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "UnaryOp1": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "operand": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "operand_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "UnaryOp2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "operand": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "operand_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "UnaryOp3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "UnaryOp": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "operand": 1,
                        "op": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "operand_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ]
                    ]
                }
            },
            "Assert": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assert": 1,
                        "Compare": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "Op": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "inside": 3,
                        "test": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            7,
                            8
                        ],
                        [
                            7,
                            13
                        ],
                        [
                            12,
                            13
                        ]
                    ]
                }
            },
            "FunctionDef": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "mention": 2,
                        "instance": 3,
                        "Return": 1,
                        "ctx": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "inside": 1,
                        "ctx": 1,
                        "value": 1,
                        "defined_in_function": 1,
                        "function_name": 1,
                        "defined_in_module": 1,
                        "instance_rev": 3,
                        "inside_rev": 1,
                        "value_rev": 1,
                        "function_name_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            7
                        ],
                        [
                            20,
                            21
                        ]
                    ]
                }
            },
            "AsyncFunctionDef": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AsyncFunctionDef": 1,
                        "mention": 2,
                        "instance": 3,
                        "Return": 1,
                        "ctx": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "inside": 1,
                        "ctx": 1,
                        "value": 1,
                        "defined_in_function": 1,
                        "function_name": 1,
                        "defined_in_module": 1,
                        "instance_rev": 3,
                        "inside_rev": 1,
                        "value_rev": 1,
                        "function_name_rev": 1
                    },
                    "offsets": [
                        [
                            12,
                            13
                        ],
                        [
                            26,
                            27
                        ]
                    ]
                }
            },
            "ClassDef": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "ClassDef": 1,
                        "FunctionDef": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "mention": 2,
                        "instance": 2,
                        "subword": 2
                    },
                    "edges": {
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "subword": 2,
                        "instance": 2,
                        "function_name": 1,
                        "defined_in_class": 1,
                        "class_name": 1,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "function_name_rev": 1
                    },
                    "offsets": [
                        [
                            28,
                            32
                        ]
                    ]
                }
            },
            "AnnAssign": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "type_annotation": 1,
                        "AnnAssign": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Constant": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 1,
                        "ctx": 1,
                        "target": 1,
                        "value": 1,
                        "annotation_for": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "target_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            3,
                            6
                        ],
                        [
                            9,
                            10
                        ]
                    ]
                }
            },
            "With": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "With": 1,
                        "withitem": 1,
                        "mention": 4,
                        "instance": 5,
                        "ctx": 2,
                        "Call": 1,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 5,
                        "ctx": 5,
                        "inside": 5,
                        "items": 1,
                        "executed_inside_with": 1,
                        "defined_in_module": 1,
                        "instance_rev": 5,
                        "inside_rev": 5
                    },
                    "offsets": [
                        [
                            5,
                            9
                        ],
                        [
                            10,
                            11
                        ],
                        [
                            16,
                            17
                        ],
                        [
                            22,
                            30
                        ],
                        [
                            31,
                            32
                        ]
                    ]
                }
            },
            "AsyncWith": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AsyncWith": 1,
                        "withitem": 1,
                        "mention": 4,
                        "instance": 5,
                        "ctx": 2,
                        "Call": 1,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 5,
                        "ctx": 5,
                        "inside": 5,
                        "items": 1,
                        "executed_inside_with": 1,
                        "defined_in_module": 1,
                        "instance_rev": 5,
                        "inside_rev": 5
                    },
                    "offsets": [
                        [
                            11,
                            15
                        ],
                        [
                            16,
                            17
                        ],
                        [
                            22,
                            23
                        ],
                        [
                            28,
                            36
                        ],
                        [
                            37,
                            38
                        ]
                    ]
                }
            },
            "arg": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "mention": 2,
                        "instance": 3,
                        "type_annotation": 1,
                        "Constant": 1,
                        "Return": 1,
                        "ctx": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 3,
                        "annotation_for": 1,
                        "inside": 2,
                        "ctx": 1,
                        "value": 1,
                        "defined_in_function": 1,
                        "function_name": 1,
                        "defined_in_module": 1,
                        "instance_rev": 3,
                        "inside_rev": 1,
                        "value_rev": 1,
                        "function_name_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            12
                        ],
                        [
                            9,
                            12
                        ],
                        [
                            15,
                            16
                        ],
                        [
                            29,
                            30
                        ]
                    ]
                }
            },
            "Await": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Await": 1,
                        "Call": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 1,
                        "value": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ],
                        [
                            6,
                            12
                        ]
                    ]
                }
            },
            "Raise": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Raise": 1,
                        "Call": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 1,
                        "exc": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            6,
                            15
                        ],
                        [
                            6,
                            17
                        ]
                    ]
                }
            },
            "Lambda": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Lambda": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "Constant": 1,
                        "Op": 1,
                        "subword": 1
                    },
                    "edges": {
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            10,
                            11
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "IfExp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "IfExp": 1,
                        "Constant": 2,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "test": 1,
                        "if_true": 1,
                        "if_false": 1,
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            4,
                            20
                        ],
                        [
                            9,
                            13
                        ],
                        [
                            19,
                            20
                        ]
                    ]
                }
            },
            "keyword": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Call": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "#keyword#": 2,
                        "Constant": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 4,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            2
                        ],
                        [
                            10,
                            11
                        ]
                    ]
                }
            },
            "Attribute": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Attribute": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "#attr#": 2,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 1,
                        "ctx": 1,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "inside_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ]
                    ]
                }
            },
            "If": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "If": 2,
                        "Compare": 2,
                        "mention": 4,
                        "instance": 10,
                        "ctx": 2,
                        "Op": 1,
                        "Constant": 1,
                        "Assign": 3,
                        "Tuple": 2,
                        "subword": 4
                    },
                    "edges": {
                        "subword": 4,
                        "instance": 10,
                        "ctx": 12,
                        "inside": 6,
                        "test": 2,
                        "value": 3,
                        "targets": 3,
                        "executed_if_true": 2,
                        "elts": 4,
                        "executed_if_false": 2,
                        "defined_in_module": 1,
                        "instance_rev": 10,
                        "inside_rev": 2,
                        "value_rev": 2,
                        "targets_rev": 2,
                        "elts_rev": 4
                    },
                    "offsets": [
                        [
                            3,
                            4
                        ],
                        [
                            3,
                            12
                        ],
                        [
                            8,
                            12
                        ],
                        [
                            17,
                            18
                        ],
                        [
                            21,
                            22
                        ],
                        [
                            28,
                            29
                        ],
                        [
                            28,
                            38
                        ],
                        [
                            33,
                            38
                        ],
                        [
                            42,
                            43
                        ],
                        [
                            46,
                            47
                        ],
                        [
                            57,
                            58
                        ],
                        [
                            57,
                            61
                        ],
                        [
                            60,
                            61
                        ],
                        [
                            64,
                            65
                        ],
                        [
                            64,
                            68
                        ],
                        [
                            67,
                            68
                        ]
                    ]
                }
            },
            "For": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "For": 1,
                        "mention": 5,
                        "instance": 9,
                        "ctx": 2,
                        "Assign": 1,
                        "Call": 3,
                        "If": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 5
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 9,
                        "ctx": 9,
                        "target": 1,
                        "iter": 1,
                        "inside": 9,
                        "value": 1,
                        "targets": 1,
                        "executed_in_for": 2,
                        "test": 1,
                        "executed_if_true": 2,
                        "control_flow": 1,
                        "next": 2,
                        "executed_in_for_orelse": 1,
                        "defined_in_module": 1,
                        "instance_rev": 9,
                        "target_rev": 1,
                        "iter_rev": 1,
                        "inside_rev": 6,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            4,
                            5
                        ],
                        [
                            9,
                            13
                        ],
                        [
                            18,
                            19
                        ],
                        [
                            22,
                            24
                        ],
                        [
                            22,
                            27
                        ],
                        [
                            25,
                            26
                        ],
                        [
                            34,
                            35
                        ],
                        [
                            34,
                            40
                        ],
                        [
                            39,
                            40
                        ],
                        [
                            49,
                            52
                        ],
                        [
                            53,
                            54
                        ],
                        [
                            63,
                            68
                        ],
                        [
                            78,
                            81
                        ],
                        [
                            82,
                            83
                        ]
                    ]
                }
            },
            "AsyncFor": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "AsyncFor": 1,
                        "mention": 5,
                        "instance": 9,
                        "ctx": 2,
                        "Assign": 1,
                        "Call": 3,
                        "If": 1,
                        "Compare": 1,
                        "Op": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 5
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 9,
                        "ctx": 9,
                        "target": 1,
                        "iter": 1,
                        "inside": 9,
                        "value": 1,
                        "targets": 1,
                        "executed_in_for": 2,
                        "test": 1,
                        "executed_if_true": 2,
                        "control_flow": 1,
                        "next": 2,
                        "executed_in_for_orelse": 1,
                        "defined_in_module": 1,
                        "instance_rev": 9,
                        "target_rev": 1,
                        "iter_rev": 1,
                        "inside_rev": 6,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            10,
                            11
                        ],
                        [
                            15,
                            19
                        ],
                        [
                            24,
                            25
                        ],
                        [
                            28,
                            30
                        ],
                        [
                            28,
                            33
                        ],
                        [
                            31,
                            32
                        ],
                        [
                            40,
                            41
                        ],
                        [
                            40,
                            46
                        ],
                        [
                            45,
                            46
                        ],
                        [
                            55,
                            58
                        ],
                        [
                            59,
                            60
                        ],
                        [
                            69,
                            74
                        ],
                        [
                            84,
                            87
                        ],
                        [
                            88,
                            89
                        ]
                    ]
                }
            },
            "Try": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Try": 1,
                        "Assign": 3,
                        "mention": 6,
                        "instance": 9,
                        "ctx": 2,
                        "ExceptHandler": 1,
                        "Call": 1,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 9,
                        "ctx": 9,
                        "value": 3,
                        "targets": 3,
                        "executed_in_try": 1,
                        "type": 1,
                        "executed_with_try_handler": 1,
                        "executed_in_try_except": 1,
                        "inside": 2,
                        "executed_in_try_final": 1,
                        "executed_in_try_else": 1,
                        "defined_in_module": 1,
                        "instance_rev": 9,
                        "value_rev": 3,
                        "targets_rev": 3,
                        "type_rev": 1,
                        "inside_rev": 2
                    },
                    "offsets": [
                        [
                            8,
                            9
                        ],
                        [
                            12,
                            13
                        ],
                        [
                            21,
                            30
                        ],
                        [
                            40,
                            41
                        ],
                        [
                            44,
                            45
                        ],
                        [
                            55,
                            56
                        ],
                        [
                            59,
                            60
                        ],
                        [
                            73,
                            78
                        ],
                        [
                            79,
                            80
                        ]
                    ]
                }
            },
            "While": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "While": 1,
                        "Compare": 1,
                        "mention": 3,
                        "instance": 4,
                        "ctx": 1,
                        "Op": 1,
                        "Call": 1,
                        "subword": 5
                    },
                    "edges": {
                        "subword": 5,
                        "instance": 4,
                        "ctx": 4,
                        "inside": 5,
                        "test": 1,
                        "executed_in_while": 1,
                        "defined_in_module": 1,
                        "instance_rev": 4,
                        "inside_rev": 4
                    },
                    "offsets": [
                        [
                            6,
                            7
                        ],
                        [
                            6,
                            12
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            17,
                            24
                        ],
                        [
                            25,
                            26
                        ]
                    ]
                }
            },
            "Break": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "While": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1
                    },
                    "edges": {
                        "test": 1,
                        "control_flow": 1,
                        "executed_in_while": 1,
                        "defined_in_module": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ],
                        [
                            15,
                            20
                        ]
                    ]
                }
            },
            "Continue": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "While": 1,
                        "Constant": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1
                    },
                    "edges": {
                        "test": 1,
                        "control_flow": 1,
                        "executed_in_while": 1,
                        "defined_in_module": 1
                    },
                    "offsets": [
                        [
                            6,
                            10
                        ],
                        [
                            15,
                            23
                        ]
                    ]
                }
            },
            "Pass": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1
                    },
                    "edges": {
                        "control_flow": 1,
                        "defined_in_module": 1
                    },
                    "offsets": [
                        [
                            0,
                            4
                        ]
                    ]
                }
            },
            "Dict": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Dict": 1,
                        "mention": 4,
                        "instance": 4,
                        "ctx": 1,
                        "subword": 4
                    },
                    "edges": {
                        "subword": 4,
                        "instance": 4,
                        "ctx": 4,
                        "keys": 2,
                        "values": 2,
                        "defined_in_module": 1,
                        "instance_rev": 4,
                        "keys_rev": 2,
                        "values_rev": 2
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            6,
                            7
                        ],
                        [
                            8,
                            9
                        ]
                    ]
                }
            },
            "Set": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Set": 1,
                        "mention": 2,
                        "instance": 2,
                        "ctx": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "ctx": 2,
                        "elts": 2,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "elts_rev": 2
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "ListComp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "ListComp": 1,
                        "mention": 2,
                        "instance": 3,
                        "ctx": 2,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "ctx": 3,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 3,
                        "inside_rev": 3
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ]
                    ]
                }
            },
            "DictComp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "DictComp": 1,
                        "mention": 3,
                        "instance": 5,
                        "ctx": 2,
                        "Tuple": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 5,
                        "ctx": 6,
                        "inside": 5,
                        "defined_in_module": 1,
                        "instance_rev": 5,
                        "inside_rev": 5
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            9,
                            10
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            16,
                            20
                        ]
                    ]
                }
            },
            "SetComp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "SetComp": 1,
                        "mention": 2,
                        "instance": 3,
                        "ctx": 2,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 3,
                        "ctx": 3,
                        "inside": 3,
                        "defined_in_module": 1,
                        "instance_rev": 3,
                        "inside_rev": 3
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ]
                    ]
                }
            },
            "GeneratorExp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "GeneratorExp": 1,
                        "mention": 2,
                        "instance": 4,
                        "ctx": 2,
                        "Op": 1,
                        "Constant": 1,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 4,
                        "ctx": 4,
                        "inside": 6,
                        "defined_in_module": 1,
                        "instance_rev": 4,
                        "inside_rev": 4
                    },
                    "offsets": [
                        [
                            1,
                            2
                        ],
                        [
                            7,
                            8
                        ],
                        [
                            12,
                            16
                        ],
                        [
                            20,
                            21
                        ],
                        [
                            25,
                            26
                        ]
                    ]
                }
            },
            "BinOp": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "BinOp": 1,
                        "mention": 3,
                        "instance": 3,
                        "ctx": 2,
                        "Op": 1,
                        "subword": 3
                    },
                    "edges": {
                        "subword": 3,
                        "instance": 3,
                        "ctx": 3,
                        "left": 1,
                        "right": 1,
                        "op": 1,
                        "value": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 3,
                        "left_rev": 1,
                        "right_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            4,
                            9
                        ],
                        [
                            8,
                            9
                        ]
                    ]
                }
            },
            "ImportFrom": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "ImportFrom": 1,
                        "mention": 2,
                        "instance": 2,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "inside": 2,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    }
                }
            },
            "alias": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Import": 1,
                        "mention": 2,
                        "instance": 2,
                        "subword": 2
                    },
                    "edges": {
                        "subword": 2,
                        "instance": 2,
                        "inside": 2,
                        "defined_in_module": 1,
                        "instance_rev": 2,
                        "inside_rev": 2
                    }
                }
            },
            "List": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "List": 1,
                        "Constant": 1,
                        "ctx": 2,
                        "mention": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "elts": 1,
                        "ctx": 2,
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            8,
                            9
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "Tuple": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Tuple": 1,
                        "Constant": 1,
                        "ctx": 2,
                        "mention": 1,
                        "instance": 1,
                        "subword": 1
                    },
                    "edges": {
                        "elts": 1,
                        "ctx": 2,
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            16
                        ],
                        [
                            5,
                            6
                        ],
                        [
                            8,
                            9
                        ],
                        [
                            11,
                            12
                        ],
                        [
                            14,
                            15
                        ]
                    ]
                }
            },
            "JoinedStr": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "JoinedStr": 1
                    },
                    "edges": {
                        "defined_in_module": 1
                    }
                }
            },
            "FormattedValue": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "JoinedStr": 1
                    },
                    "edges": {
                        "defined_in_module": 1
                    }
                }
            },
            "Bytes": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            10
                        ]
                    ]
                }
            },
            "Num": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            5
                        ]
                    ]
                }
            },
            "Str": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "Assign": 1,
                        "Constant": 1,
                        "mention": 1,
                        "instance": 1,
                        "ctx": 1,
                        "subword": 1
                    },
                    "edges": {
                        "value": 1,
                        "subword": 1,
                        "instance": 1,
                        "ctx": 1,
                        "targets": 1,
                        "defined_in_module": 1,
                        "instance_rev": 1,
                        "targets_rev": 1
                    },
                    "offsets": [
                        [
                            0,
                            1
                        ],
                        [
                            4,
                            9
                        ]
                    ]
                }
            },
            "FunctionDef2": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "mention": 10,
                        "instance": 10,
                        "type_annotation": 3,
                        "Constant": 3,
                        "ctx": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 20
                    },
                    "edges": {
                        "subword": 28,
                        "instance": 10,
                        "annotation_for": 2,
                        "ctx": 1,
                        "inside": 12,
                        "returned_by": 1,
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "function_name": 1,
                        "defined_in_module": 1,
                        "instance_rev": 10,
                        "inside_rev": 9,
                        "function_name_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            37,
                            41
                        ],
                        [
                            53,
                            57
                        ],
                        [
                            66,
                            74
                        ],
                        [
                            71,
                            74
                        ],
                        [
                            91,
                            123
                        ],
                        [
                            109,
                            123
                        ],
                        [
                            125,
                            133
                        ],
                        [
                            142,
                            146
                        ],
                        [
                            159,
                            163
                        ],
                        [
                            164,
                            168
                        ],
                        [
                            177,
                            181
                        ],
                        [
                            182,
                            183
                        ],
                        [
                            194,
                            200
                        ],
                        [
                            206,
                            218
                        ],
                        [
                            223,
                            227
                        ]
                    ]
                }
            },
            "FunctionDef3": {
                "description": {
                    "nodes": {
                        "Module": 1,
                        "FunctionDef": 1,
                        "mention": 4,
                        "instance": 4,
                        "ctx": 1,
                        "CtlFlowInstance": 1,
                        "CtlFlow": 1,
                        "subword": 6
                    },
                    "edges": {
                        "subword": 6,
                        "instance": 4,
                        "ctx": 1,
                        "inside": 3,
                        "control_flow": 1,
                        "defined_in_function": 1,
                        "function_name": 1,
                        "defined_in_module": 1,
                        "instance_rev": 4,
                        "inside_rev": 3,
                        "function_name_rev": 1
                    },
                    "offsets": [
                        [
                            1,
                            10
                        ],
                        [
                            30,
                            34
                        ],
                        [
                            38,
                            44
                        ],
                        [
                            50,
                            54
                        ]
                    ]
                }
            }
        }
    }

    def make_graph_description(graph):
        desc = {
            "nodes": Counter(graph["nodes"]["type"]),
            "edges": Counter(graph["edges"]["type"]),
        }
        if graph["offsets"] is not None:
            desc["offsets"] = sorted(sorted(list(zip(
                graph["offsets"]["start" if "start" in graph["offsets"].columns else "offset_start"].map(int),
                graph["offsets"]["end" if "end" in graph["offsets"].columns else "offset_end"].map(int),
            )), key=lambda x: x[1]), key=lambda x: x[0])
        return desc

    bpe_tokenizer_path = Path(os.getcwd())\
        .joinpath(__file__)\
        .parent.parent\
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
        desc = make_graph_description(graph)
        answers[variety][node]["description"] = desc
        assert (
                correct_answers[variety][node]["description"] == json.loads(json.dumps(desc))
        )
        visualize(graph["nodes"], graph["edges"], f"{node}_{variety}.png", show_reverse=True)

        variety = "v1.0_control_flow"
        graph = source_code_to_graph(code, variety=variety)
        print("\t", variety, len(graph["nodes"]), len(graph["edges"]))
        desc = make_graph_description(graph)
        answers[variety][node]["description"] = desc
        assert (
                correct_answers[variety][node]["description"] == json.loads(json.dumps(desc))
        )
        visualize(graph["nodes"], graph["edges"], f"{node}_{variety}.png", show_reverse=True)

        variety = "v3.5"
        graph = source_code_to_graph(
            code, variety=variety, reverse_edges=True, mention_instances=True, bpe_tokenizer_path=bpe_tokenizer_path
        )
        print("\t", variety, len(graph["nodes"]), len(graph["edges"]))
        desc = make_graph_description(graph)
        answers[variety][node]["description"] = desc
        assert (
                correct_answers[variety][node]["description"] == json.loads(json.dumps(desc))
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
        print("\t", variety, len(graph["nodes"]), len(graph["edges"]))
        desc = make_graph_description(graph)
        answers[variety][node]["description"] = desc
        assert (
                correct_answers[variety][node]["description"] == json.loads(json.dumps(desc))
        )
        visualize(
            graph["nodes"].rename({"name": "serialized_name"}, axis=1),
            graph["edges"].rename({"src": "source_node_id", "dst": "target_node_id"}, axis=1),
            f"{node}_{variety}.png", show_reverse=True
        )

    print(json.dumps(answers, indent=4))


if __name__ == "__main__":
    test_graph_builder()