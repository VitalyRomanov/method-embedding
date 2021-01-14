import argparse
import pandas as pd
from SourceCodeTools.data.sourcetrail.Dataset import load_data
from SourceCodeTools.data.sourcetrail.sourcetrail_types import node_types, edge_types
from SourceCodeTools.data.sourcetrail.file_utils import *
import networkx as nx

from SourceCodeTools.data.sourcetrail.common import custom_tqdm

leaf_types = [
'subword', 'Op', 'CtlFlow', 'Constant'
]

def filter_sourcetrail_edges(edges):
    sourcetrail_types = set(edge_types.values())
    not_sourcetrail_edges = edges[
        edges['type'].apply(lambda type: type not in sourcetrail_types)
    ]

    return not_sourcetrail_edges


def filter_reverse_edges(edges):
    not_reverse_edges = edges[
        edges['type'].apply(lambda type: not type.endswith("_rev"))
    ]

    return not_reverse_edges


def filter_service_edges(edges):
    to_filter = {'prev', 'next'}

    edges = edges[
        edges['type'].apply(lambda type: type not in to_filter)
    ]

    edges = edges[
        edges['type'].apply(lambda type: not type.startswith("depends_on_") and not type.startswith("execute_when_"))
    ]

    return edges

def get_leaf_nodes(nodes, edges):
    leafs = set(edges['src'].values) - set(edges['dst'].values)

    leaf_nodes = nodes[
        nodes['id'].apply(lambda id_: id_ in leafs)
    ]
    return leaf_nodes


def assign_initial_hierarchy_level(nodes, only_calls):

    only_functions = nodes.query(f"type == '{node_types[4096]}' or type == '{node_types[8192]}'")
    initial_level = set(only_functions['id'].tolist()) - set(only_calls['src'].tolist())

    function_levels = {}

    for f_id in initial_level:
        function_levels[f_id] = 0

    return function_levels


def assign_hierarcy_levels(nodes, edges):
    only_calls = edges.query(f"type == '{edge_types[8]}'")

    function_levels = assign_initial_hierarchy_level(nodes, only_calls)

    call_groups = dict(list(only_calls.groupby("src")))
    it = 0
    while len(call_groups) > 0:
        to_pop = []
        for func_id, func_edges in call_groups.items():
            dsts = func_edges['dst']
            if all(dst in function_levels for dst in dsts):
                function_levels[func_id] = max(function_levels[dst] for dst in dsts) + 1
                to_pop.append(func_id)
        for func_id in to_pop:
            call_groups.pop(func_id)
        print(
            f"Iteration: {it}, unassigned functions: {len(call_groups)}, maximum level: {max(function_levels.values())}",
            end="\r")
    print()

    hierarchy_levels = []
    for func_id, level in function_levels.items():
        hierarchy_levels.append({
            'id': func_id,
            'hierarchy_level': level
        })

    return pd.DataFrame(hierarchy_levels)


def main(args):
    nodes, edges = load_data(args.nodes, args.edges)

    hierarchy_levels = assign_hierarcy_levels(nodes, edges)

    persist(hierarchy_levels, os.path.join(os.path.dirname(args.nodes), "hierarchies.csv"))

    print(
        hierarchy_levels.groupby("hierarchy_level")\
            .count().rename({'id': 'count'}, axis=1)\
            .sort_values(by='count', ascending=False).to_string()
    )



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('nodes', help='')
    parser.add_argument('edges', help='')
    args = parser.parse_args()

    main(args)