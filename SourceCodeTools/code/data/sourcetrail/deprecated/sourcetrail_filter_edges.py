import sys, os
import pandas as pd
from csv import QUOTE_NONNUMERIC

allowed = sys.argv[2:]

edges = pd.read_csv(sys.argv[1])

edges['allow_src'] = edges["Source"].apply(lambda x: x in allowed)
edges['allow_dst'] = edges["Target"].apply(lambda x: x in allowed)
edges['allow'] = edges['allow_src'] | edges['allow_dst']

edges = edges[edges['allow']]

edges[["Source","Target"]].to_csv(os.path.join(os.path.dirname(sys.argv[1]), "allowed_edges.csv"), index=False, quoting=QUOTE_NONNUMERIC)