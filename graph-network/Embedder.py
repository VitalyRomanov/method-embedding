import numpy as np
import tensorflow as tf

class Embedder:
    def __init__(self, id_map, embeddings):
        self.e = embeddings
        self.ind = id_map
        aid, iid = zip(*id_map.items())
        self.inv = dict(zip(iid, aid))


    def __getitem__(self, key):
        # if not hasattr(self, "map_id"):
        #     self.map_id = np.vectorize(lambda id: self.ind[id])
        if type(key) == int:
            return self.e[self.ind[key], :]
        elif type(key) == np.ndarray:
            # return self.e[self.map_id(key), :]
            if isinstance(self.e, np.ndarray):
                return self.e[np.array([self.ind[k] for k in key], dtype=np.int32), :]
            # elif isinstance(self.e, tf.ResourceVariable):
            else: # this is assumed to be tensorflow Variable, neet to work out how to do type checking
                slices = np.array([self.ind[k] for k in key], dtype=np.int32)
                return tf.gather(self.e, slices, axis=0)
            # else:
            #     raise TypeError("Problem with embedder internal type:", type(self.e))
        else:
            raise TypeError("Unknown type:", type(key))