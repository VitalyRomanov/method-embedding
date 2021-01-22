import pickle
import sys
import pandas
import numpy as np
import os
from collections import Counter
from sklearn.preprocessing import normalize

model_path = sys.argv[1]

nodes_path = os.path.join(model_path, "nodes.csv")
edges_path = os.path.join(model_path, "edges.csv")
embedders_path = os.path.join(model_path, "embeddings.pkl")

embedders = pickle.load(open(embedders_path, "rb"))
nodes = pandas.read_csv(nodes_path)
edges = pandas.read_csv(edges_path)
degrees = Counter(edges['src'].tolist()) + Counter(edges['dst'].tolist())

def create_name_maps(nodes, degrees):
    ids = nodes['id'].values
    names = nodes['name']#.apply(lambda x: x.split(".")[-1]).values

    id2name = dict()
    name2id = dict()

    for id_, name_ in zip(ids, names):
        if degrees[id_] > 1:
            id2name[id_] = name_
            name2id[name_] = id_
    # id2name = dict(zip(ids, names))
    # name2id = dict(zip(names, ids))
    return id2name, name2id

id2name, name2id = create_name_maps(nodes, degrees)


def get_embeddings_from_groups(nodes, embedder, id2name):
    ids_ = nodes['id'].values

    emb_ = []
    group_ids = []

    for id_ in ids_:
        if id_ in id2name:
            emb_.append(embedder.e[embedder.ind[id_]])
            group_ids.append(id_)

    # normalize(in_vectors, axis=1)
    return group_ids, normalize(np.array(emb_), axis=1)


groups = []

for group, gr_ in nodes.groupby("type_backup"):
    # if group == 4096:
    group_ids, group_embs = get_embeddings_from_groups(gr_, embedders[-1], id2name)

    groups.append((group, group_ids, group_embs))

del embedders

while True:
    query = input("Enter query: ")
    query = query.strip()

    if query in name2id:
        id_ = name2id[query]
    else:
        print("Name not found\n")
        continue

    for group_name, group_ids, group_embs in groups:
        if id_ in group_ids:
            q_v = group_embs[group_ids.index(id_), :]
            score = group_embs @ q_v
            ind = np.flip(np.argsort(score))[:20]

            for i in ind:
                print("%s\t%.4f" % (id2name[group_ids[i]],
                                    group_embs[group_ids.index(id_)].dot(group_embs[i])))
            break
    else:
        print("Nothing found")