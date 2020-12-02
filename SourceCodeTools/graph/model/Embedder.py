import numpy as np

class Embedder:
    def __init__(self, id_map, embeddings):
        self.e = embeddings
        self.ind = id_map
        aid, iid = zip(*id_map.items())
        self.inv = dict(zip(iid, aid))


    def __getitem__(self, key):
        # if not hasattr(self, "map_id"):
        #     self.map_id = np.vectorize(lambda id: self.ind[id])
        # TODO
        # support for str key type
        if type(key) == int:
            return self.e[self.ind[key], :]
        elif type(key) == np.ndarray:
            # return self.e[self.map_id(key), :]
            if isinstance(self.e, np.ndarray):
                return self.e[np.array([self.ind[k] for k in key], dtype=np.int32), :]
            # elif isinstance(self.e, tf.ResourceVariable):
            else: # this is assumed to be tensorflow Variable, need to work out how to do type checking
                import tensorflow as tf # is this a good practice?
                slices = np.array([self.ind[k] for k in key], dtype=np.int32)
                return tf.gather(self.e, slices, axis=0)
            # else:
            #     raise TypeError("Problem with embedder internal type:", type(self.e))
        else:
            raise TypeError("Unknown type:", type(key))

    # @staticmethod
    # def load_word2vec(path):
    #     vecs = []
    #     id_map = dict()
    #
    #     with open(path) as vectors:
    #         n_vectors, n_dims = map(int, vectors.readline().strip().split())
    #
    #
    #         for ind in range(n_vectors):
    #             elements = vectors.readline().strip().split()
    #             id_ = int(elements[0])
    #             vec = list(map(float, elements[1:]))
    #             assert len(vec) == n_dims
    #             id_map[id_] = ind
    #             vecs.append(vec)
    #
    #     embs = np.array(vecs)
    #     return Embedder(id_map, embs)