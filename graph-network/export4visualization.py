from Embedder import Embedder
import pickle
import sys
import pandas
import numpy as np
import os

model_path = sys.argv[1]

if len(sys.argv) > 2:
    max_embs = sys.argv[2]
else:
    max_embs = 5000

nodes_path = os.path.join(model_path, "nodes.csv")
embedders_path = os.path.join(model_path, "embeddings.pkl")

embedders = pickle.load(open(embedders_path, "rb"))
nodes = pandas.read_csv(nodes_path)

ids = nodes['id'].values
names = nodes['label'].values

id_name_map = list(zip(ids, names))

ind_mapper = embedders[0].ind

id_name_map = sorted(id_name_map, key=lambda x: ind_mapper[x[0]])

ids, names = zip(*id_name_map)

print(f"Limiting to {max_embs} embeddings")

print("Writing meta...", end="")
with open(os.path.join(model_path, "emb4proj_meta.tsv"), "w") as meta:
    for name in names[:max_embs]:
        meta.write(f"{name}\n")
print("done")

for ind, e in enumerate(embedders):
    print(f"Writing embedding layer {ind}...", end="")
    np.savetxt(os.path.join(model_path, f"emb4proj{ind}.tsv"), e.e[:max_embs], delimiter="\t")
    print("done")