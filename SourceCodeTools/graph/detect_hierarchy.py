import argparse
from SourceCodeTools.data.sourcetrail.Dataset import load_data
from SourceCodeTools.data.sourcetrail.sourcetrail_types import node_types, edge_types
from SourceCodeTools.data.sourcetrail.file_utils import *
import networkx as nx
from itertools import chain

from SourceCodeTools.data.sourcetrail.common import custom_tqdm

# leaf_types = [
# 'subword', 'Op', 'CtlFlow', 'Constant'
# ]
#
# def filter_sourcetrail_edges(edges):
#     sourcetrail_types = set(edge_types.values())
#     not_sourcetrail_edges = edges[
#         edges['type'].apply(lambda type: type not in sourcetrail_types)
#     ]
#
#     return not_sourcetrail_edges
#
#
# def filter_reverse_edges(edges):
#     not_reverse_edges = edges[
#         edges['type'].apply(lambda type: not type.endswith("_rev"))
#     ]
#
#     return not_reverse_edges
#
#
# def filter_service_edges(edges):
#     to_filter = {'prev', 'next'}
#
#     edges = edges[
#         edges['type'].apply(lambda type: type not in to_filter)
#     ]
#
#     edges = edges[
#         edges['type'].apply(lambda type: not type.startswith("depends_on_") and not type.startswith("execute_when_"))
#     ]
#
#     return edges
#
# def get_leaf_nodes(nodes, edges):
#     leafs = set(edges['src'].values) - set(edges['dst'].values)
#
#     leaf_nodes = nodes[
#         nodes['id'].apply(lambda id_: id_ in leafs)
#     ]
#     return leaf_nodes


class CallEdgesCache:
    def __init__(self):
        self.cache = dict()
        self.cache_hit = 0
        self.cache_miss = 0

    def get_calls_without_recursions(self, edges, start_node):
        query_string = f"src == {start_node} and dst != {start_node}"  # exclude recursions
        if query_string not in self.cache:
            self.cache[query_string] = edges.query(query_string)
            self.cache_miss += 1
        else:
            self.cache_hit += 1
        return self.cache[query_string]


class HierarchyDetector:
    def __init__(self, nodes, edges, max_cycle_depth):
        self.nodes = nodes
        self.call_edges = edges.query(f"type == '{edge_types[8]}'")
        self.max_cycle_depth = max_cycle_depth
        self.call_edges_cache = CallEdgesCache()
        self.call_graph = nx.convert_matrix.from_pandas_edgelist(self.call_edges, source='src', target='dst', create_using=nx.DiGraph)
        cycles = list(nx.simple_cycles(self.call_graph))
        self.cycle_neighbours = {}
        for c in cycles:
            for n in c:
                if n not in self.cycle_neighbours:
                    self.cycle_neighbours[n] = set()
                self.cycle_neighbours[n].update(c)


    def assign_initial_hierarchy_level(self):
        only_functions = self.nodes.query(
            f"type == '{node_types[4096]}' or type == '{node_types[8192]}' or type == '{node_types[1]}'"
        )
        initial_level = set(only_functions['id'].tolist()) - set(self.call_edges['src'].tolist())

        function_levels = {}

        for f_id in initial_level:
            function_levels[f_id] = 0

        return function_levels

    def get_edges_with_depth(self, start, depth):
        local_edges = self.call_edges_cache.get_calls_without_recursions(self.call_edges, start)
        dsts = [dst for dst in local_edges['dst']]

        for depth_level in range(depth):
            if len(dsts) > 0:
                new_edges = pd.concat([self.call_edges_cache.get_calls_without_recursions(
                    self.call_edges, start_node=dst
                ) for dst in dsts])
                # new_edges = self.call_edges.query(" or ".join(f"src == {dst}" for dst in dsts))
                dsts = new_edges['dst']
                local_edges = local_edges.append(new_edges)

        return local_edges

    # def get_cycles(self, edges, start):
    def get_cycles(self, start, cycle_depth):
        subgraph_nodes = list(self.call_graph[start].keys()) + [start]

        cycle_dsts = subgraph_nodes
        for i in range(cycle_depth):
            new_dsts = []
            for dst in cycle_dsts:
                new_dsts.extend(list(self.call_graph[dst].keys()))
            cycle_dsts = new_dsts
            subgraph_nodes.extend(new_dsts)

        # g = nx.DiGraph()
        # g.add_nodes_from((n, self.call_graph.nodes[n]) for n in subgraph_nodes)
        # g.add_edges_from((n, nbr, d)
        #       for n, nbrs in self.call_graph.adj.items() if n in subgraph_nodes
        #       for nbr, d in nbrs.items() if nbr in subgraph_nodes)
        # g.graph.update(self.call_graph.graph)

        g = self.call_graph.subgraph(subgraph_nodes)

        # if start in self.cycle_neighbours:
        #     return self.cycle_neighbours[start]
        # else:
        #     return None

        # g = nx.convert_matrix.from_pandas_edgelist(edges, source='src', target='dst', create_using=nx.DiGraph)
        try:
            cycle_edges = nx.algorithms.cycles.find_cycle(g, source=start)
            cycle_nodes = set(chain.from_iterable(cycle_edges))
        except nx.NetworkXNoCycle:
            cycle_nodes = None
        return cycle_nodes

    def assign_hierarchy_levels(self):
        # only_calls = edges.query(f"type == '{edge_types[8]}'")

        function_levels = self.assign_initial_hierarchy_level()

        call_groups = dict(list(self.call_edges.groupby("src")))

        it = 0
        unresolved = -1
        detect_cycles = False
        cycle_depth = 0

        while len(call_groups) > 0:
            to_pop = []

            for func_id, func_edges in custom_tqdm(
                    call_groups.items(), total=len(call_groups), message=f"Iteration {it}"
            ):
                dsts = [dst for dst in func_edges['dst'] if dst != func_id]  # exclude recursive dst

                cycle_nodes = None
                if detect_cycles:
                    # func_call_neighbourhood = self.get_edges_with_depth(start=func_id, depth=cycle_depth)
                    # cycle_nodes = self.get_cycles(func_call_neighbourhood, start=func_id)
                    cycle_nodes = self.get_cycles(start=func_id, cycle_depth=cycle_depth)

                if cycle_nodes is not None:  # break cycles
                    dsts = [dst for dst in dsts if dst not in cycle_nodes]

                destinations = list(dst in function_levels for dst in dsts if dst != func_id)

                if len(destinations) == 0:
                    # only recursion calls
                    function_levels[func_id] = 0
                    to_pop.append(func_id)
                else:
                    if all(destinations):
                        function_levels[func_id] = max(function_levels[dst] for dst in dsts if dst != func_id) + 1
                        to_pop.append(func_id)

                # if cycle_nodes is not None and func_id in function_levels:
                #     for dst in cycle_nodes:
                #         function_levels[dst] = function_levels[func_id]
                #         to_pop.append(dst)
                if cycle_depth >= self.max_cycle_depth:
                    logging.warning("Maximum cycle depth reached, assigning functions using best guess")
                    function_levels[func_id] = max(function_levels[dst] for dst in dsts if dst in function_levels) + 1

            for func_id in to_pop:
                if func_id in call_groups:
                    call_groups.pop(func_id)

            print(
                f"\nIteration: {it}, unassigned functions: {len(call_groups)}, "
                f"maximum level: {max(function_levels.values())}",
                end="\n")

            detect_cycles = False

            if unresolved == len(call_groups):
                detect_cycles = True
                cycle_depth += 1

            unresolved = len(call_groups)
            it += 1
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

    hierarchy_detector = HierarchyDetector(nodes, edges, max_cycle_depth=args.max_cycle_depth)
    hierarchy_levels = hierarchy_detector.assign_hierarchy_levels()

    # hierarchy_levels = assign_hierarchy_levels(nodes, edges, max_cycle_depth=args.max_cycle_depth)

    persist(hierarchy_levels, os.path.join(os.path.dirname(args.nodes), "hierarchies.csv"))

    print(
        hierarchy_levels.groupby("hierarchy_level")\
            .count().rename({'id': 'count'}, axis=1)\
            .sort_values(by='count', ascending=False).to_string()
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('nodes', help='')
    parser.add_argument('edges', help='')
    parser.add_argument('--max_cycle_depth', dest='max_cycle_depth', default=10, help='')
    args = parser.parse_args()

    main(args)
