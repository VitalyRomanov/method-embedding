import torch
import torch.nn as nn
import numpy as np
import random as rnd
import hashlib

from SourceCodeTools.graph.model.ElementEmbedderBase import ElementEmbedderBase

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
        # self.norm = nn.BatchNorm1d(emb_size)

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


def hashstr(s, num_buckets):
    return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % num_buckets


def window(x, gram_size):
    x = "<" + x + ">"
    length = len(x)
    return (x[i:i + gram_size] for i in range(0, length) if i+gram_size<=length)


def create_fixed_length(parts, length):
    empty = np.zeros((length,), dtype=np.int32)

    empty[0:min(parts.size, length)] = parts[0:min(parts.size, length)]
    return empty


class ElementEmbedderWithSubwords(ElementEmbedderBase, nn.Module):
    def __init__(self, elements, emb_size, num_buckets=5000, max_len=100, gram_size=3):
        ElementEmbedderBase.__init__(self, elements=elements, compact_dst=False)
        nn.Module.__init__(self)

        self.emb_size = emb_size

        names = elements['dst']
        reprs = names.map(lambda x: window(x, gram_size)) \
            .map(lambda grams: (hashstr(g, num_buckets) for g in grams)) \
            .map(lambda int_grams: np.fromiter(int_grams, dtype=np.int32))\
            .map(lambda parts: create_fixed_length(parts, max_len))

        self.name2repr = dict(zip(names, reprs))

        self.embed = nn.Embedding(num_buckets, emb_size)

    def __getitem__(self, ids):
        candidates = [rnd.choice(self.element_lookup[id]) for id in ids]
        emb_matr = np.array([self.name2repr[c] for c in candidates], dtype=np.int32)
        return torch.LongTensor(emb_matr)

    def forward(self, input, **kwargs):
        x = self.embed(input)
        return torch.mean(x, dim=1)

if __name__ == '__main__':
    import pandas as pd
    test_data = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 4, 5],
        # "dst": [6, 11, 12, 11, 14, 15, 16]
        "dst": ["hello", "how", "are", "you", "doing", "today", "?"]
    })

    # ee = ElementEmbedder(test_data, 5)
    ee = ElementEmbedderWithSubwords(test_data, 5, max_len=10)

    rand_ind = [0, 1, 2, 3, 4, 4, 5]#np.random.randint(low=0, high=len(ee), size=20)
    sample = ee[rand_ind]
    ee(sample)
    from pprint import pprint
    pprint(list(zip(rand_ind, sample)))
    print(ee.elem_probs)
    print(ee.elem2id)
    print(ee.sample_negative(3))