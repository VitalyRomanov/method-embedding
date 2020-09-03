from Embedder import Embedder
import numpy as np
import sys
import pickle

def load_w2v_map(w2v_path):

    embs = []
    w_map = dict()

    with open(w2v_path) as w2v:
        n_vectors, n_dims = map(int, w2v.readline().strip().split())
        for ind in range(n_vectors):
            e = w2v.readline().strip().split()

            word = e[0]
            w_map[word] = len(w_map)

            embs.append(list(map(float, e[1:])))

    return Embedder(w_map, np.array(embs))

w2v_path = sys.argv[1]
out_path = sys.argv[2]

emb = load_w2v_map(w2v_path)
pickle.dump(emb, open(out_path, "wb"))