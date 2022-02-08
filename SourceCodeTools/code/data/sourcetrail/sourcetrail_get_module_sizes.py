import os
from collections import Counter
from pprint import pprint

from SourceCodeTools.code.data.file_utils import unpersist


# def estimate_module_sizes(nodes_path):
#     nodes = unpersist(nodes_path)
#
#     def is_part_of_module(type):
#         module_types = {val for _, val in node_types.items()}
#         return type in module_types
#
#     module_nodes = nodes.query("type.map(@is_part_of_module)", local_dict={"is_part_of_module": is_part_of_module})
#
#     def get_module_name(name):
#         return name.split(".")[0]
#
#     module_nodes.eval("serialized_name = serialized_name.map(@get_module_name)", local_dict={"get_module_name": get_module_name}, inplace=True)
#
#     id2module = dict(zip(module_nodes["id"], module_nodes["serialized_name"]))
#
#     def node_location(id):
#         return id2module.get(id, pd.NA)
#
#     nodes.eval("mentioned_in = mentioned_in.map(@node_location)", local_dict={"node_location": node_location}, inplace=True)
#     nodes.dropna(axis=0, inplace=True)
#
#     module_sizes = nodes.groupby("mentioned_in").count()
#
#     print()


def estimate_module_sizes(path):
    module_count = Counter()
    for dir in os.listdir(path):
        module_path = os.path.join(path, dir)
        if not os.path.isdir(module_path):
            continue

        nodes_path = os.path.join(module_path, "nodes_with_ast.bz2")

        if os.path.isfile(nodes_path):
            module_count[dir] = unpersist(nodes_path).shape[0]

    pprint(module_count.most_common())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("environments")
    args = parser.parse_args()
    estimate_module_sizes(args.environments)