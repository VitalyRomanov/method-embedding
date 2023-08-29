from collections import Counter

from SourceCodeTools.code.data.file_utils import unpersist


def main():
    import sys

    node_path = sys.argv[1]
    edge_path = sys.argv[2]

    edges = unpersist(edge_path)

    all_node_mention = edges["source_node_id"].tolist()
    all_node_mention.extend(edges["target_node_id"].tolist())

    counts = Counter(all_node_mention)

    total_degree = 0
    total_nodes = 0

    for node, count in counts.items():
        total_degree += count
        total_nodes += 1

    print("Average degree:", total_degree / total_nodes)

if __name__ == "__main__":
    main()