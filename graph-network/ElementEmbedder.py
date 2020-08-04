import torch
import torch.nn as nn
import numpy as np
import random as rnd

from ElementEmbedderBase import ElementEmbedderBase

class ElementEmbedder(ElementEmbedderBase, nn.Module):
    def __init__(self, elements, emb_size, compact_dst=True):
        ElementEmbedderBase.__init__(self, elements=elements, compact_dst=compact_dst)
        nn.Module.__init__(self)

        # self.elements = elements.copy()
        # self.elem2id = compact_property(elements['dst'])
        #
        # if compact_dst:
        #     self.elements['emb_id'] = self.elements['dst'].apply(lambda x: self.elem2id[x])
        # else:
        #     self.elements['emb_id'] = self.elements['dst']
        #
        # self.element_lookup = {}
        # for name, group in self.elements.groupby('id'):
        #     self.element_lookup[name] = group['emb_id'].tolist()
        #
        # # self.element_lookup = dict(zip(self.elements['id'], self.elements['emb_id']))
        #
        # self.emb_size = emb_size
        # self.n_elements = len(self.elem2id)
        #
        # self.init_neg_sample()

        self.emb_size = emb_size
        n_elems = self.elements['emb_id'].unique().size
        self.embed = nn.Embedding(n_elems, emb_size)
        self.norm = nn.BatchNorm1d(emb_size)

    # def init_neg_sample(self):
    #     WORD2VEC_SAMPLING_POWER = 3 / 4
    #
    #     # compute distribution of dst elements
    #     counts = self.elements['emb_id'].value_counts(normalize=True)
    #     # idxs = list(map(lambda x: self.elem2id[x], counts.index))
    #     self.idxs = counts.index
    #     self.neg_prob = counts.to_numpy()
    #     self.neg_prob **= WORD2VEC_SAMPLING_POWER
    #     # ind_freq = list(zip(idxs, freq))
    #     # ind_freq = sorted(ind_freq, key=lambda x:x[0])
    #     # _, self.elem_probs = zip(*ind_freq)
    #     # self.elem_probs = np.power(self.elem_probs, WORD2VEC_SAMPLING_POWER)
    #     self.neg_prob /= sum(self.neg_prob)
    #     # self.random_indices = np.arange(0, len(self.elem2id))
    #
    # def sample_negative(self, size):
    #     return np.random.choice(self.idxs, size, replace=True, p=self.neg_prob)

    def __getitem__(self, ids):
        return torch.LongTensor(ElementEmbedderBase.__getitem__(self, ids=ids))
        # return torch.LongTensor(np.array([rnd.choice(self.element_lookup[id]) for id in ids]))

    # def __len__(self):
    #     return len(self.element_lookup)

    def forward(self, input, **kwargs):
        return self.embed(input)
        # return self.norm(self.embed(input))

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
    print(ee.elem_probs)
    print(ee.elem2id)
    print(ee.sample_negative(3))