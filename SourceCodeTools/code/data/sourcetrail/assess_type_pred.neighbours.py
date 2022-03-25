import pickle

import sys

import numpy as np

from SourceCodeTools.code.data.file_utils import unpersist

type_annotations_path = sys.argv[1]
nodes = sys.argv[2]
embeddings_path = sys.argv[3]

nodes = unpersist(nodes)
type_annotations = unpersist(type_annotations_path)
embs = pickle.load(open(embeddings_path, "rb"))# [0]

def normalize(typeann):
    return typeann.strip("\"").strip("'").split("[")[0]

node2nodetype = dict(zip(nodes["id"], nodes["type"]))

type_annotations["dst"] = type_annotations["dst"].apply(normalize)

type_annotations = type_annotations[
    type_annotations["src"].apply(lambda x: node2nodetype[x] == "mention")
]

seed_nodes = type_annotations["src"]
node2type = dict(zip(type_annotations["src"], type_annotations["dst"]))

seed_embs = []
missing_emb = []
for nid in seed_nodes:
    if nid in embs:
        seed_embs.append(embs[nid])
    else:
        missing_emb.append(nid)

type_embs = np.vstack(seed_embs)

added = set()
same_or_not = dict()

for nid, emb in zip(seed_nodes, type_embs):
    diff = emb.reshape(1, -1) - type_embs
    diff2 = np.square(diff)
    dist = np.sqrt(np.sum(diff2, axis=1))
    order = np.argsort(dist)
    min_pos = order[1]
    closest = seed_nodes.iloc[min_pos]
    if (nid, closest) not in added and (closest, nid) not in added:
        added.add((nid, closest))
        added.add((closest, nid))
        same_or_not[(nid, closest)] = node2type[nid] == node2type[closest]

print(sum(list(same_or_not.values())) / len(same_or_not))

