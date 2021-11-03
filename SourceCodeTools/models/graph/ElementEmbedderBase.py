import logging
import sys
from typing import Tuple

import numpy as np
import random as rnd

from SourceCodeTools.tabular.common import compact_property


class ElementEmbedderBase:
    def __init__(self, elements, nodes, compact_dst=True, dst_to_global=False):
        self.elements = self.preprocess_element_data(elements.copy(), nodes, compact_dst, dst_to_global=dst_to_global)
        self.init(compact_dst)

    def init(self, compact_dst):
        if compact_dst:
            elem2id, self.inverse_dst_map = compact_property(self.elements['dst'], return_order=True)
            self.elements['emb_id'] = self.elements['dst'].apply(lambda x: elem2id.get(x, -1))
            assert -1 not in self.elements['emb_id'].tolist()
        else:
            self.elements['emb_id'] = self.elements['dst']

        self.element_lookup = {}
        # for name, group in self.elements.groupby('id'):
        #     self.element_lookup[name] = group['emb_id'].tolist()
        for id_, emb_id in self.elements[["id", "emb_id"]].values:
            if id_ in self.element_lookup:
                self.element_lookup[id_].append(emb_id)
            else:
                self.element_lookup[id_] = [emb_id]

        self.init_neg_sample()

    def preprocess_element_data(self, element_data, nodes, compact_dst, dst_to_global=False):
        """
        Takes the mapping from the original ids in the graph to the target embedding, maps dataset ids to graph ids,
        creates structures that will allow mapping from the global graph id to the desired embedding
        :param element_data:
        :param nodes:
        :param compact_dst:
        :param dst_to_global:
        :return:
        """
        if len(element_data) == 0:
            logging.error(f"Not enough data for the embedder: {len(element_data)}. Exiting...")
            sys.exit()

        id2nodeid = dict(zip(nodes['id'].tolist(), nodes['global_graph_id'].tolist()))
        id2typedid = dict(zip(nodes['id'].tolist(), nodes['typed_id'].tolist()))
        id2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))

        def get_node_pools(element_data):
            node_typed_pools = {}
            for orig_node_id in element_data['src']:
                global_id = id2nodeid.get(orig_node_id, None)

                if global_id is None:
                    continue

                src_type = id2type.get(orig_node_id, None)
                src_typed_id = id2typedid.get(orig_node_id, None)

                if src_type not in node_typed_pools:
                    node_typed_pools[src_type] = []

                node_typed_pools[src_type].append(src_typed_id)

            return node_typed_pools

        self.node_typed_pools = get_node_pools(element_data)

        # map to global graph id
        element_data['id'] = element_data['src'].apply(lambda x: id2nodeid.get(x, None))
        # # save type id to allow pooling nodes of certain types
        # element_data['src_type'] = element_data['src'].apply(lambda x: id2type.get(x, None))
        # element_data['src_typed_id'] = element_data['src'].apply(lambda x: id2typedid.get(x, None))

        if dst_to_global:
            element_data['dst'] = element_data['dst'].apply(lambda x: id2nodeid.get(x, None))

        element_data = element_data.astype({
            'id': 'Int32',
            # 'src_type': 'category',
            # 'src_typed_id': 'Int32',
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

    def init_neg_sample(self, skipgram_sampling_power=0.75):
        # compute distribution of dst elements
        counts = self.elements['emb_id'].value_counts(normalize=True)
        self.idxs = counts.index
        self.neg_prob = counts.to_numpy()
        self.neg_prob **= skipgram_sampling_power

        self.neg_prob /= sum(self.neg_prob)

    def sample_negative(self, size):
        # TODO
        # Try other distributions
        return np.random.choice(self.idxs, size, replace=True, p=self.neg_prob)

    def __getitem__(self, ids):
        return np.fromiter((rnd.choice(self.element_lookup[id]) for id in ids), dtype=np.int32)

    def get_src_pool(self, ntypes=None):
        """
        Get pool of nodes present in the elements represented as typed_id. For graphs without node types, typed and
        global ids match.
        :param ntypes:
        :return:
        """
        return {ntype: set(self.node_typed_pools[ntype]) for ntype in ntypes if ntype in self.node_typed_pools}
        # return {ntype: set(self.elements.query(f"src_type == '{ntype}'")['src_typed_id'].tolist()) for ntype in ntypes}

    def _create_pools(self, train_idx, val_idx, test_idx, pool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_idx = np.fromiter(pool.intersection(train_idx.reshape((-1,)).tolist()), dtype=np.int64)
        val_idx = np.fromiter(pool.intersection(val_idx.reshape((-1,)).tolist()), dtype=np.int64)
        test_idx = np.fromiter(pool.intersection(test_idx.reshape((-1,)).tolist()), dtype=np.int64)
        return train_idx, val_idx, test_idx

    def create_idx_pools(self, train_idx, val_idx, test_idx):
        """
        Given split ids, filter only those that are given in elements.
        The format of splits is dict {node_type: typed_ids}.
        For graphs with single node type, global and typed ids match.
        :param train_idx:
        :param val_idx:
        :param test_idx:
        :return:
        """
        train_pool = {}
        test_pool = {}
        val_pool = {}

        node_types = set(train_idx.keys()) | set(val_idx.keys()) | set(test_idx.keys())

        pool = self.get_src_pool(ntypes=node_types)

        for ntype in train_idx.keys():
            if ntype in pool:
                train, test, val = self._create_pools(train_idx[ntype], val_idx[ntype], test_idx[ntype], pool[ntype])
                train_pool[ntype] = train
                test_pool[ntype] = test
                val_pool[ntype] = val

        return train_pool, val_pool, test_pool
        # else:
        #     return self._create_pools(train_idx, test_idx, val_idx, self.get_src_pool())

    def __len__(self):
        return len(self.element_lookup)