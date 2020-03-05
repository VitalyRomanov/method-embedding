import sys
import pandas

edge_path = sys.argv[1]
edge_type = int(sys.argv[2])

edges = pandas.read_csv(edge_path)

edges = edges[edges['type'] == edge_type][['source_node_id', 'target_node_id']]

edges.to_csv(f"edges_type_{edge_type}.tsv", sep ="\t", index=False, header=False)