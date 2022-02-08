import argparse

from SourceCodeTools.code.data.dataset.Dataset import load_data
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import edge_types
from SourceCodeTools.code.data.file_utils import *
import networkx as nx

from functools import lru_cache

from SourceCodeTools.code.data.sourcetrail.common import custom_tqdm


# class CallEdgesCache:
#     def __init__(self):
#         self.cache = dict()
#         self.cache_hit = 0
#         self.cache_miss = 0
#
#     def get_calls_without_recursions(self, edges, start_node):
#         query_string = f"src == {start_node} and dst != {start_node}"  # exclude recursions
#         if query_string not in self.cache:
#             self.cache[query_string] = edges.query(query_string)
#             self.cache_miss += 1
#         else:
#             self.cache_hit += 1
#         return self.cache[query_string]


class HierarchyDetector:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.call_edges = edges.query(f"type == '{edge_types[8]}'")
        self.call_graph = nx.convert_matrix.from_pandas_edgelist(self.call_edges, source='src', target='dst', create_using=nx.DiGraph)
        self.detect_cycle_groups()

    def detect_cycle_groups(self):
        cycles = list(map(set, nx.simple_cycles(self.call_graph)))
        keep_merging = True

        while keep_merging:
            merged_cycles = []
            before_merging = len(cycles)

            while len(cycles) > 1:
                if cycles[0].isdisjoint(cycles[1]):
                    merged_cycles.append(cycles.pop(0))
                else:
                    merged_cycles.append(cycles.pop(0) | cycles.pop(0))
            if len(cycles) > 0:
                merged_cycles.append(cycles.pop(0))

            assert len(cycles) == 0

            after_merging = len(merged_cycles)

            if before_merging == after_merging:
                keep_merging = False

            cycles = merged_cycles

        # run full overlap scan

        cycle_groups = []

        while len(cycles) > 0:

            current_cycle_group = cycles.pop(0)

            position = 0

            while position < len(cycles):
                if current_cycle_group.isdisjoint(cycles[position]):
                    position += 1
                else:
                    current_cycle_group |= cycles.pop(position)

            cycle_groups.append(current_cycle_group)

        self.cycle_groups = cycle_groups
        self.nodes_in_cycles = set()

        for group in self.cycle_groups:
            self.nodes_in_cycles |= group

    def assign_initial_hierarchy_level(self):
        initial_level = set(self.call_edges['dst'].tolist()) - set(self.call_edges['src'].tolist())

        for f_id in initial_level:
            self.call_graph.nodes[f_id]['level'] = 0

    @lru_cache(maxsize=5000)
    def get_call_neighbours(self, start):
        return list(self.call_graph[start].keys())

    @lru_cache(maxsize=5000)
    def get_call_neighbours_with_depth(self, start, depth):
        if depth == 0:
            outer_nodes = self.get_call_neighbours(start)
            return set(outer_nodes + [start]), outer_nodes
        else:
            subgraph_nodes, outer_nodes = self.get_call_neighbours_with_depth(start, depth - 1)
            new_outer_nodes = []
            for dst in outer_nodes:
                new_outer_nodes.extend(self.get_call_neighbours(dst))

            subgraph_nodes.update(new_outer_nodes)

            return subgraph_nodes, new_outer_nodes

    def get_cycle_dsts(self, start):
        if start in self.nodes_in_cycles:
            cycle_group = None
            for group in self.cycle_groups:
                if start in group:
                    cycle_group = group
                    break

            dsts = set()
            for n in cycle_group:
                dsts.update(self.get_call_neighbours(n))

            dsts -= cycle_group

            return dsts, cycle_group
        else:
            return None, None

    def assign_hierarchy_levels(self):

        self.assign_initial_hierarchy_level()

        call_groups = dict(list(self.call_edges.groupby("src")))

        it = 0
        unresolved = -1
        detect_cycles = False
        max_level = 0

        while len(call_groups) > 0:
            to_pop = []

            for func_id, func_edges in custom_tqdm(
                    call_groups.items(), total=len(call_groups), message=f"Iteration {it}"
            ):
                dsts = [dst for dst in func_edges['dst'] if dst != func_id]  # exclude recursive dst

                cycle_group = None
                if detect_cycles:
                    cycle_dsts, cycle_group = self.get_cycle_dsts(func_id)
                    if cycle_dsts is None:
                        continue
                    else:
                        dsts = cycle_dsts

                destinations = list('level' in self.call_graph.nodes[dst] for dst in dsts if dst != func_id)

                if len(destinations) == 0:
                    # only recursion calls
                    self.call_graph.nodes[func_id]['level'] = 0
                    to_pop.append(func_id)
                else:
                    if all(destinations):
                        new_level = max(self.call_graph.nodes[dst]['level'] for dst in dsts if dst != func_id) + 1
                        if new_level > max_level:
                            max_level = new_level
                        if cycle_group is not None:
                            nodes_to_assign = cycle_group
                            assert func_id in cycle_group
                        else:
                            nodes_to_assign = [func_id]
                        for n in nodes_to_assign:
                            self.call_graph.nodes[n]['level'] = new_level
                            to_pop.append(n)

            for func_id in to_pop:
                if func_id in call_groups:
                    call_groups.pop(func_id)

            print(
                f"\nIteration: {it}, unassigned functions: {len(call_groups)}, "
                f"maximum level: {max_level}",
                end="\n")

            detect_cycles = False

            if unresolved == len(call_groups):
                detect_cycles = True

            unresolved = len(call_groups)
            it += 1
        print()

        hierarchy_levels = []
        for func_id, data in self.call_graph.nodes.items():
            hierarchy_levels.append({
                'id': func_id,
                'hierarchy_level': data['level']
            })

        return pd.DataFrame(hierarchy_levels)


def main(args):
    nodes, edges = load_data(args.nodes, args.edges)

    hierarchy_detector = HierarchyDetector(nodes, edges)
    hierarchy_levels = hierarchy_detector.assign_hierarchy_levels()

    persist(hierarchy_levels, os.path.join(os.path.dirname(args.nodes), "hierarchies.csv"))

    print(
        hierarchy_levels.groupby("hierarchy_level")\
            .count().rename({'id': 'count'}, axis=1)\
            .sort_values(by='count', ascending=False).to_string()
    )

    # nodes = pd.read_parquet("call_nodes.parquet")
    # nodes = nodes.rename({'level': 'type'}, axis=1)
    # nodes['type'] = nodes['type'].apply(lambda x: f"type_{x}")
    # nodes['serialized_name'] = ""
    # nodes.to_pickle("call_nodes.bz2")
    #
    # edges = pd.read_parquet("call_edges.parquet")
    # edges['type'] = edges['appears_in_cycle'].apply(lambda x: "regular" if x is None else "cycle")
    # edges.to_pickle("call_edges.bz2")

    # pd.DataFrame([{"id": node,
    #                "type": f"lvl_{self.call_graph.nodes[node]['level'] if 'level' in self.call_graph.nodes[node] else None}",
    #                "serialized_name": id2name[node]} for node in self.call_graph.nodes]).to_pickle("call_nodes.bz2")
    # pd.DataFrame([{"source_node_id": e1, "target_node_id": e2, "type": "regular" if "appears_in_cycle" not in self.call_graph.edges[e1, e2] else "cycle"}for e1, e2 in self.call_graph.edges]).to_pickle("call_edges.bz2")

    # match p=shortestPath( (n{id:6028322})-[*1..10]->(e:lvl_None) ) where e.id <> 6028322 return [n IN nodes(p) WHERE 'lvl_None' IN labels(n)]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('nodes', help='')
    parser.add_argument('edges', help='')
    args = parser.parse_args()

    main(args)
