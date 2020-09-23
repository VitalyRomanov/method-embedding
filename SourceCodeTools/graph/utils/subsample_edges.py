import pandas as pd
import sys, os

frac = float(sys.argv[2])
comp = int(sys.argv[3])

edges = pd.read_csv(sys.argv[1])
sample = edges.sample(frac=frac)
sample.to_csv(os.path.join(os.path.dirname(sys.argv[1]), f"edges_{frac}.csv"), index=False)

all_other = edges.query(f"type != {comp}")
comp_edge = edges.query(f"type == {comp}")
N = len(comp_edge) - (len(edges) - len(sample))

assert N > 0, f"{len(edges)} {len(sample)} {len(comp_edge)} {N}"

pd.concat([all_other, comp_edge.sample(n=N)]).to_csv(os.path.join(os.path.dirname(sys.argv[1]), f"edges_{comp}_{frac}.csv"), index=False)