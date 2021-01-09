import pandas as pd
import os, sys

edges = pd.read_csv(sys.argv[1])

edges.query("type >= 0").to_csv(os.path.join(os.path.dirname(sys.argv[1]), "edges_no_ast.csv"), index=False)
edges.query("type < 0").to_csv(os.path.join(os.path.dirname(sys.argv[1]), "edges_ast.csv"), index=False)