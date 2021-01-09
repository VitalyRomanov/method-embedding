from typing import Tuple

import numpy as np
import random as rnd

from SourceCodeTools.common import compact_property


# def create_idx_pools(splits: Tuple, pool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     train_idx, val_idx, test_idx = splits
#     train_idx = np.fromiter(pool.intersection(train_idx.tolist()), dtype=np.int64)
#     val_idx = np.fromiter(pool.intersection(val_idx.tolist()), dtype=np.int64)
#     test_idx = np.fromiter(pool.intersection(test_idx.tolist()), dtype=np.int64)
#     return train_idx, val_idx, test_idx


# def compact_property(values):
#     uniq = np.unique(values)
#     prop2pid = dict(zip(uniq, range(uniq.size)))
#     # prop2pid = dict(list(zip(uniq, list(range(uniq.size)))))
#     return prop2pid


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
        # TODO
        # Try other distributions
        return np.random.choice(self.idxs, size, replace=True, p=self.neg_prob)

    def __getitem__(self, ids):
        return np.fromiter((rnd.choice(self.element_lookup[id]) for id in ids), dtype=np.int32)

    def get_src_pool(self, ntypes=None):
        # if ntypes is None:
        #     return set(self.elements['id'].to_list())
        # # elif ntypes == ['_U']:
        # #     # this case processes graphs with no specific node types https://docs.dgl.ai/en/latest/api/python/heterograph.html
        # #     return {"_U": set(self.elements['id'].to_list())}
        # else:
        return {ntype: set(self.elements.query(f"src_type == '{ntype}'")['src_typed_id'].tolist()) for ntype in ntypes}

    def _create_pools(self, train_idx, val_idx, test_idx, pool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_idx = np.fromiter(pool.intersection(train_idx.tolist()), dtype=np.int64)
        val_idx = np.fromiter(pool.intersection(val_idx.tolist()), dtype=np.int64)
        test_idx = np.fromiter(pool.intersection(test_idx.tolist()), dtype=np.int64)
        return train_idx, val_idx, test_idx

    def create_idx_pools(self, train_idx, val_idx, test_idx):
        # if isinstance(train_idx, dict):
        train_pool = {}
        test_pool = {}
        val_pool = {}

        pool = self.get_src_pool(ntypes=list(train_idx.keys()))

        for ntype in train_idx.keys():
            train, test, val = self._create_pools(train_idx[ntype], val_idx[ntype], test_idx[ntype], pool[ntype])
            train_pool[ntype] = train
            test_pool[ntype] = test
            val_pool[ntype] = val

        return train_pool, val_pool, test_pool
        # else:
        #     return self._create_pools(train_idx, test_idx, val_idx, self.get_src_pool())

    def __len__(self):
        return len(self.element_lookup)