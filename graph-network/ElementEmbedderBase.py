import numpy as np
import random as rnd

def compact_property(values):
    uniq = np.unique(values)
    prop2pid = dict(zip(uniq, range(uniq.size)))
    # prop2pid = dict(list(zip(uniq, list(range(uniq.size)))))
    return prop2pid

class ElementEmbedderBase:
    def __init__(self, elements, compact_dst=True):

        self.elements = elements.copy()

        if compact_dst:
            elem2id = compact_property(elements['dst'])
            self.elements['emb_id'] = self.elements['dst'].apply(lambda x: elem2id[x])
        else:
            self.elements['emb_id'] = self.elements['dst']

        self.element_lookup = {}
        for name, group in self.elements.groupby('id'):
            self.element_lookup[name] = group['emb_id'].tolist()

        self.init_neg_sample()

    def init_neg_sample(self):
        WORD2VEC_SAMPLING_POWER = 3 / 4

        # compute distribution of dst elements
        counts = self.elements['emb_id'].value_counts(normalize=True)
        # idxs = list(map(lambda x: self.elem2id[x], counts.index))
        self.idxs = counts.index
        self.neg_prob = counts.to_numpy()
        self.neg_prob **= WORD2VEC_SAMPLING_POWER
        # ind_freq = list(zip(idxs, freq))
        # ind_freq = sorted(ind_freq, key=lambda x:x[0])
        # _, self.elem_probs = zip(*ind_freq)
        # self.elem_probs = np.power(self.elem_probs, WORD2VEC_SAMPLING_POWER)
        self.neg_prob /= sum(self.neg_prob)
        # self.random_indices = np.arange(0, len(self.elem2id))

    def sample_negative(self, size):
        return np.random.choice(self.idxs, size, replace=True, p=self.neg_prob)

    def __getitem__(self, ids):
        return np.fromiter((rnd.choice(self.element_lookup[id]) for id in ids), dtype=np.int32)

    def __len__(self):
        return len(self.element_lookup)