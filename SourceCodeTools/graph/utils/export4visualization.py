from SourceCodeTools.graph.model.Embedder import Embedder
import pickle
import sys
import pandas
import numpy as np
import os
from collections import Counter

model_path = sys.argv[1]

if len(sys.argv) > 2:
    max_embs = int(sys.argv[2])
else:
    max_embs = 5000

nodes_path = os.path.join(model_path, "nodes.csv")
edges_path = os.path.join(model_path, "edges.csv")
embedders_path = os.path.join(model_path, "embeddings.pkl")

embedders = pickle.load(open(embedders_path, "rb"))
nodes = pandas.read_csv(nodes_path)

edges = pandas.read_csv(edges_path)
degrees = Counter(edges['src'].tolist()) + Counter(edges['dst'].tolist())

ids = nodes['id'].values
names = nodes['name']#.apply(lambda x: x.split(".")[-1]).values

id_name_map = list(zip(ids, names))
id_name_map_d = dict(id_name_map)

ind_mapper = embedders[0].ind

id_name_map = sorted(id_name_map, key=lambda x: ind_mapper[x[0]])

ids, names = zip(*id_name_map)

print(f"Limiting to {max_embs} embeddings")


# emb0 = []
# emb1 = []
# emb2 = []

for group, gr_ in nodes.groupby("type_backup"):
    c_ = 0

    names = []
    embs = [[] for _ in embedders]

    nodes_in_group = set(gr_['id'].tolist())
    for ind, (id_, count) in enumerate(degrees.most_common()):
        if id_ not in nodes_in_group: continue
        names.append(id_name_map_d[id_])
        for emb_, embedder in zip(embs, embedders):
            emb_.append(embedder.e[embedder.ind[id_]])
        # emb0.append(embedders[0].e[embedders[0].ind[id_]])
        # emb1.append(embedders[1].e[embedders[1].ind[id_]])
        # emb2.append(embedders[2].e[embedders[2].ind[id_]])
        c_ += 1
        if c_ >= max_embs - 1: break
        # if ind >= max_embs-1: break

    # np.savetxt(os.path.join(model_path, "emb4proj_meta.tsv"), np.array(names).reshape(-1, 1))
    for ind, emb_ in enumerate(embs):
        np.savetxt(os.path.join(model_path, f"emb4proj{ind}_{group}.tsv"), np.array(emb_), delimiter="\t")
    # np.savetxt(os.path.join(model_path, "emb4proj0.tsv"), np.array(emb0), delimiter="\t")
    # np.savetxt(os.path.join(model_path, "emb4proj1.tsv"), np.array(emb1), delimiter="\t")
    # np.savetxt(os.path.join(model_path, "emb4proj2.tsv"), np.array(emb2), delimiter="\t")

    print("Writing meta...", end="")
    with open(os.path.join(model_path, f"emb4proj_meta_{group}.tsv"), "w") as meta:
        for name in names[:max_embs]:
            meta.write(f"{name}\n")
    print("done")

# with open(os.path.join(model_path, "emb4proj2w2v.txt"), "w") as w2v:
#     for ind, name in enumerate(names):
#         w2v.write("%s " % name)
#         for j, v in enumerate(emb2[ind]):
#             if j < len(emb2[ind]) - 1:
#                 w2v.write("%f " % v)
#             else:
#                 w2v.write("%f\n" % v)


# for ind, e in enumerate(embedders):
#     print(f"Writing embedding layer {ind}...", end="")
#     np.savetxt(os.path.join(model_path, f"emb4proj{ind}.tsv"), e.e[:max_embs], delimiter="\t")
#     print("done")
