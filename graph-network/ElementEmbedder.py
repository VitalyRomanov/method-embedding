import torch
import torch.nn as nn
import numpy as np
import random as rnd

def compact_property(values):
    uniq = np.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    # prop2pid = dict(list(zip(uniq, list(range(uniq.size)))))
    return prop2pid

class ElementEmbedder(nn.Module):
    # TODO:
    # 1. Data arrive in a DataFrame. First column stores src, the second - dst
    # 2. Need to group using the value in the first column, and enumerate possible dst
    # 3. When id is provided, forward pass retrieves one of the possible dst for the partial
    #       src (randomly).
    # 4. Forward function should have a flag that allows retreiving negative samples. This will be
    #       used to generate negative samples ourside of this class (in the training procedure).
    # 5. Alternatively, I can generate dst for random src and keep the logic in this class simple. I think
    #       this will be better
    def __init__(self, elements, emb_size):
        super(ElementEmbedder, self).__init__()

        self.elements = elements.copy()
        self.elem2id = compact_property(elements['dst'])
        self.elements['emb_id'] = self.elements['dst'].apply(lambda x: self.elem2id[x])

        self.element_lookup = {}
        for name, group in self.elements.groupby('id'):
            self.element_lookup[name] = group['emb_id'].tolist()

        # self.element_lookup = dict(zip(self.elements['id'], self.elements['emb_id']))

        n_elems = self.elements['emb_id'].unique().size

        self.embed = nn.Embedding(n_elems, emb_size)

        self.emb_size = emb_size
        self.n_elements = len(self.elem2id)

    def __getitem__(self, ids):
        return np.array([rnd.choice(self.element_lookup[id]) for id in ids])

    def __len__(self):
        return len(self.element_lookup)

    def forward(self, input, **kwargs):
        return self.embed(input)

if __name__ == '__main__':
    import pandas as pd
    test_data = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 4, 5],
        "dst": [6, 11, 12, 11, 14, 15, 16]
    })

    ee = ElementEmbedder(test_data, 5)

    rand_ind = np.random.randint(low=0, high=len(ee), size=20)
    sample = ee[rand_ind]
    from pprint import pprint
    pprint(list(zip(rand_ind, sample)))