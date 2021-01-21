import logging
import sys
from typing import Tuple

import numpy as np
import random as rnd

from SourceCodeTools.common import compact_property


class ElementEmbedderBase:
    def __init__(self, elements, nodes, compact_dst=True):

        self.elements = self.preprocess_element_data(elements.copy(), nodes, compact_dst)

        if compact_dst:
            elem2id = compact_property(elements['dst'])
            self.elements['emb_id'] = self.elements['dst'].apply(lambda x: elem2id[x])
        else:
            self.elements['emb_id'] = self.elements['dst']

        self.element_lookup = {}
        for name, group in self.elements.groupby('id'):
            self.element_lookup[name] = group['emb_id'].tolist()

        self.init_neg_sample()

    def preprocess_element_data(self, element_data, nodes, compact_dst):
        if len(element_data) == 0:
            logging.error(f"Not enough data for the embedder: {len(element_data)}. Exiting...")
            sys.exit()

        id2nodeid = dict(zip(nodes['id'].tolist(), nodes['global_graph_id'].tolist()))
        id2typedid = dict(zip(nodes['id'].tolist(), nodes['typed_id'].tolist()))
        id2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))

        element_data['id'] = element_data['src'].apply(lambda x: id2nodeid.get(x, None))
        element_data['src_type'] = element_data['src'].apply(lambda x: id2type.get(x, None))
        element_data['src_typed_id'] = element_data['src'].apply(lambda x: id2typedid.get(x, None))
        element_data = element_data.astype({
            'id': 'Int32',
            'src_type': 'category',
            'src_typed_id': 'Int32',
        })

        if compact_dst is False:  # creating api call embedder
            # element_data = element_data.rename({'dst': 'dst_orig'}, axis=1)
            # element_data['dst'] = element_data['dst_orig'].apply(lambda x: id2nodeid.get(x, None))
            # element_data['dst_type'] = element_data['dst_orig'].apply(lambda x: id2type.get(x, None))
            # element_data['dst_typed_id'] = element_data['dst_orig'].apply(lambda x: id2typedid.get(x, None))
            element_data.drop_duplicates(['id', 'dst'], inplace=True,
                                         ignore_index=True)  # this line apparenly filters parallel edges
            # element_data = element_data.astype({
            #     'dst': 'Int32',
            #     'dst_type': 'category',
            #     'dst_typed_id': 'Int32',
            # })

        # element_data = element_data.dropna(axis=0)
        return element_data

    def init_neg_sample(self, word2vec_sampling_power=0.75):
        # compute distribution of dst elements
        counts = self.elements['emb_id'].value_counts(normalize=True)
        self.idxs = counts.index
        self.neg_prob = counts.to_numpy()
        self.neg_prob **= word2vec_sampling_power

        self.neg_prob /= sum(self.neg_prob)

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