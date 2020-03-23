import numpy as np

class Embedder:
    def __init__(self, id_map, embeddings):
        self.e = embeddings
        self.ind = id_map
        aid, iid = zip(*id_map.items())
        self.inv = dict(zip(iid, aid))

    def __getitem__(self, key):
        if type(key) == int:
            return self.e[self.ind[key], :]
        elif type(key) == np.ndarray:
            return self.e[np.array([self.ind[k] for k in key], dtype=np.int32), :]
        else:
            raise TypeError("Unknown type:", type(key))