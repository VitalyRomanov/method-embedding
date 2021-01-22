from SourceCodeTools.code.data.sourcetrail.Dataset import load_data
import sys

node_path = sys.argv[1]
edge_path = sys.argv[2]

nodes, edges = load_data(node_path, edge_path)

node_set = set(nodes['id'].tolist())
assert len(node_set) == len(nodes['id'].tolist())

for ind, row in edges.iterrows():
    if row["src"] in node_set and row["dst"] in node_set:
        pass
    else:
        print(f"Missing: {row}")

    print(f"{ind}/{len(edges)}", end="\r")