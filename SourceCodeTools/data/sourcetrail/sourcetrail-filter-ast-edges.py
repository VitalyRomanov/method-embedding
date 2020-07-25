import pandas as pd
import sys, os

edges_path = sys.argv[1]
edges = pd.read_csv(edges_path)

edges.query("type >= 0").to_csv(os.path.join(os.path.dirname(edges_path), "edges_no_ast.csv"), index=False)
edges.query("type < 0").to_csv(os.path.join(os.path.dirname(edges_path), "edges_ast.csv"), index=False)