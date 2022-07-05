import logging
import random
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import random as rnd


from SourceCodeTools.tabular.common import compact_property


import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class FaissIndex:
    def __init__(self, X, method="inner_prod", *args, **kwargs):
        import faiss
        self.method = method
        if method == "inner_prod":
            self.index = faiss.IndexFlatIP(X.shape[1])
        elif method == "l2":
            self.index = faiss.IndexFlatL2(X.shape[1])
        else:
            raise NotImplementedError()
        self.index.add(X.astype(np.float32))

    def query_inner_prod(self, X, k):
        X = normalize(X, axis=1)
        return self.index.search(X.astype(np.float32), k=k)

    def query_l2(self, X, k):
        return self.index.search(X.astype(np.float32), k=k)

    def query(self, X, k):
        if self.method == "inner_prod":
            return self.query_inner_prod(X, k)
        elif self.method == "l2":
            return self.query_l2(X, k)
        else:
            raise NotImplementedError()


class Brute:
    def __init__(self, X, method="inner_prod", device="cpu", *args, **kwargs):
        self.vectors = X
        self.method = method
        self.device = device

    def query_inner_prod(self, X, k):
        X = torch.Tensor(normalize(X, axis=1)).to(self.device)
        vectors = torch.Tensor(self.vectors).to(self.device)
        score = (vectors @ X.T).reshape(-1,).to("cpu").numpy()
        ind = np.flip(np.argsort(score))[:k]
        return score[ind][:k], ind

    def query_l2(self, X, k):
        vectors = torch.Tensor(self.vectors).to(self.device)
        X = torch.Tensor(X).to(self.device)
        score = torch.norm(vectors - X, dim=-1).to("cpu").numpy()
        # score = np.linalg.norm(vectors - X).reshape(-1,)
        ind = np.argsort(score)[:k]
        return score[ind][:k], ind

    def query(self, X, k):
        assert X.shape[0] == 1
        if self.method == "inner_prod":
            dist, ind = self.query_inner_prod(X, k)
        elif self.method == "l2":
            dist, ind = self.query_l2(X, k)
        else:
            raise NotImplementedError()

        return dist, ind  # .reshape((-1,1))


class TargetEmbeddingProximity:
    def __init__(
            self, unique_targets, emb_size, index_backend="brute", scoring_method="inner_prod", device="cpu"
    ):
        self._unique_targets = unique_targets
        self._emb_size = emb_size
        self._scoring_method = scoring_method
        self._index_backend = index_backend
        self._device = device

        self._init_proximity_search()

    def _init_proximity_search(self):
        self._target2index = dict(zip(self._unique_targets, range(len(self._unique_targets))))
        self._target_embedding_cache = normalize(np.ones((len(self._unique_targets), self._emb_size)), axis=1) # large matrix, high memory consumption
        # Track whether all embeddings were updated at least once. After each embedding was initialized,
        # can start proximity negative sampling
        self._cold_start_status = list(False for _ in range(len(self._unique_targets)))
        self._all_embeddings_ready = False

    def update_index(self):
        if self._scoring_method == "nn":
            self.scorer_index = None
            return
        if self._index_backend == "sklearn":
            self._scorer_index = NearestNeighbors()
            self._scorer_index.fit(self._target_embedding_cache)
        elif self._index_backend == "faiss":
            self._scorer_index = FaissIndex(self._target_embedding_cache, method=self._scoring_method)
        elif self._index_backend == "brute":
            self._scorer_index = Brute(self._target_embedding_cache, method=self._scoring_method, device=self._device)
        else:
            raise ValueError(f"Unsupported backend: {self._index_backend}. Supported backends are: sklearn|faiss")

    def set_embed(self, ids, embs):
        ids = [self._target2index[id_] for id_ in ids]  # all ids here are supposed to be global, so not sure if needed
        if self._all_embeddings_ready is False:
            for id_ in ids:
                self._cold_start_status[id_] = True
            if all(self._cold_start_status):
                self._all_embeddings_ready = True
                self.update_index()
        self._target_embedding_cache[ids, :] = normalize(embs, axis=1) if self._scoring_method == "inner_prod" else embs

    @property
    def num_embeddings(self):
        return self._target_embedding_cache.shape[0]

    @property
    def all_embeddings_ready(self):
        return self._all_embeddings_ready

    def query(self, key, k):
        return self._scorer_index.query(
            self._target_embedding_cache[key].reshape(1, -1), k=k
        )

    def __getitem__(self, item):
        return self._target_embedding_cache[item]


class LabelDenseEncoder:
    def __init__(self, labels):
        self._compact_target(labels)

    def _compact_target(self, targets):
        self._target2target_id, self._inverse_target_map = compact_property(targets['dst'], return_order=True)

    def __getitem__(self, item):
        return self._target2target_id[item]

    def __contains__(self, item):
        return item in self._target2target_id

    def get(self, item, default):
        if item in self:
            return self[item]
        else:
            return default

    def encoded_labels(self):
        return sorted(list(self._target2target_id.values()))

    def num_unique_labels(self):
        return len(self._inverse_target_map)

    def get_unique_labels(self):
        return self._inverse_target_map

    def get_original_targets(self):
        labels = self.get_unique_labels()
        return dict(zip(range(len(labels)), labels))


class TargetLoader:
    def __init__(
            self, targets, label_encoder, *, compact_dst=True, target_embedding_proximity=None, emb_size=None,
            device="cpu", logger_path=None, logger_name=None, use_ns_groups=False, restrict_targets_to_src=False, **kwargs
    ):
        self._compact_targets = compact_dst
        self._emb_size = emb_size
        self._label_encoder = label_encoder
        self._use_ns_groups = use_ns_groups
        self._unique_targets = self._label_encoder.encoded_labels()
        self.device = device
        self._restrict_targets_to_src = restrict_targets_to_src
        self._init(targets)
        self._prepare_logger(logger_path, logger_name)
        self._target_embedding_proximity = target_embedding_proximity

        assert self._unique_targets[-1] == len(self._unique_targets) - 1
        self.num_unique_targets = len(self._unique_targets)

        self.extra_args = kwargs

    @property
    def embedding_proximity_ready(self):
        return self._target_embedding_proximity.all_embeddings_ready

    @property
    def target_embeddings(self):
        return self._target_embedding_proximity._target_embedding_cache

    def _prepare_logger(self, logger_path, logger_name):
        if logger_path is not None:
            _logger_path = Path(logger_path).joinpath(logger_name)
            logger_targets = Path(str(_logger_path) + "_targets.txt")
            with open(logger_targets, "w") as s:
                for t in self._label_encoder.get_original_targets():
                    s.write(f"{self._label_encoder._inverse_target_map[t]}\n")

            self._ns_logger = open(Path(str(_logger_path) + "_ns.txt"), "w")
        else:
            self._ns_logger = None

    def _drop_duplicates(self, targets):
        len_before = len(targets)
        targets.drop_duplicates(inplace=True)
        len_after = len(targets)

        if len_after != len_before:
            logging.info(f"Elements contain duplicate entries. {len_before - len_after} entries were removed")

    def _init(self, targets):
        if len(targets) == 0:
            logging.error(f"Not enough data for the embedder: {len(targets)}. Exiting...")
            sys.exit()

        self._drop_duplicates(targets)
        # self._compact_target(targets)

        compacted = targets['dst'].apply(lambda x: self._label_encoder.get(x, -1))

        self._element_lookup = defaultdict(list)
        for src, dst in zip(targets["src"], compacted):
            self._element_lookup[src].append(dst)

        self._groups = None
        if "group" in targets.columns:
            self._groups = defaultdict(set)
            for dst, group in zip(compacted, targets["group"]):
                self._groups[group].add(dst)

        self._prepare_excluded_targets(targets)

        self._init_w2v_ns(compacted)

    def _prepare_excluded_targets(self, targets):
        pass
        # code below does not work
        if self._restrict_targets_to_src is True:
            assert self._compact_targets is False
            self._excluded_targets = set(
                self._label_encoder.get(dst, -1) for dst in targets['dst']
                if dst not in self._element_lookup and self._label_encoder.get(dst, -1) != -1
            )
            # compacted_dst = set(targets['dst'].apply(lambda x: self._label_encoder.get(x, -1)))
            # compacted_src = targets['src'].apply(lambda x: self._label_encoder.get(x, pd.NA)).dropna()
            # self._excluded_targets = set(compacted_src)
        else:
            self._excluded_targets = set()

    def _init_w2v_ns(self, elements, skipgram_sampling_power=0.75):
        # compute distribution of dst elements
        counts = elements.value_counts(normalize=True)
        self._ns_idxs = counts.index
        _neg_prob = counts.to_numpy()
        _neg_prob **= skipgram_sampling_power

        # _neg_prob /= sum(_neg_prob)
        # self._neg_prob = dict()
        self._general_sample_group_key = "all"
        self._neg_prob = _neg_prob / sum(_neg_prob)
        # self._neg_prob[self._general_sample_group_key] = _neg_prob
        self._id_loc = dict(zip(self._ns_idxs, range(len(self._ns_idxs))))
        # if self._use_ns_groups:
        #     self._id_loc = dict(zip(self._ns_idxs, range(len(self._ns_idxs))))
        #     for group, dsts in self._groups.items():
        #         positions = np.fromiter((id_loc[loc] for loc in dsts), dtype=np.int32)
        #         group_probs = np.zeros_like(_neg_prob)
        #         group_probs[positions] = _neg_prob[positions]
        #         self._neg_prob[group] = group_probs / np.sum(group_probs)

    @lru_cache(5)
    def _get_ns_w2v_weights(self, group):
        if group == self._general_sample_group_key:
            return self._neg_prob
        else:
            positions = np.fromiter((self._id_loc[loc] for loc in self._groups[group]), dtype=np.int32)
            group_probs = np.zeros_like(self._neg_prob)
            group_probs[positions] = self._neg_prob[positions]
            return group_probs / np.sum(group_probs)


    def __len__(self):
        return len(self._element_lookup)

    def _sample_closest_negative(self, ids, k=1, current_group=None):
        assert self._emb_size is not None, "Sampling closest negative is not initialized, try passing `emb_size` to initializer"
        assert ids is not None
        num_emb = self._target_embedding_proximity.num_embeddings
        if k > num_emb:
            logging.warning("Requested number of negative samples is larger than total number of targets")
            k = num_emb

        negative_candidates = []
        for _ in range(k):
            for id_ in ids:
                positive = set(self._element_lookup[id_])
                negative_candidates.append(self._get_closest_to_key(
                    id_,
                    # need to choose candidates so that there are more of them than the number of positive examples
                    # also need to make sure there are more candidates available in case one node is constantly
                    # present among candidates
                    k=10 + len(positive),
                    exclude=positive, current_group=current_group)
                )

        negative = []
        for neg in negative_candidates:
            negative.extend(random.choices(neg, k=k))
        return negative

    def _get_closest_to_key(self, key, k=None, exclude=None, current_group=None):
        _, closest_keys = self._target_embedding_proximity.query(
            self._target_embedding_proximity[key].reshape(1, -1), k=k
        )
        try:
            # Try to remove the key itself from the list of neighbours
            closest_keys = list(set(closest_keys) - {key} - exclude)
        except ValueError:
            # index throws value error if the items is not in the list
            pass

        if self._use_ns_groups:
            closest_keys = [key for key in closest_keys if key in self._groups[current_group]]

        if len(closest_keys) == 0:
            # backup strategy
            closest_keys = random.choices(list(set(self._unique_targets)), k=k)
            while key in closest_keys:
                closest_keys = random.choices(list(set(self._unique_targets)), k=k)
        return closest_keys

    def sample_negative_w2v(self, ids, k=1, current_group=None):

        if not self._use_ns_groups:
            current_group = self._general_sample_group_key

        negative = []
        for _ in range(k):
            for id_ in ids:
                excluded  = self._element_lookup[id_] + [id_]

                def get_negative(group):
                    return np.random.choice(self._ns_idxs, 1, replace=True, p=self._get_ns_w2v_weights(group)).astype(np.int32)[0]

                neg = get_negative(current_group)
                attempt_counter = 10
                while neg in excluded and attempt_counter > 0:
                    neg = get_negative(current_group)
                    attempt_counter -= 1
                negative.append(neg)
        return np.array(negative, dtype=np.int32)
        # return np.random.choice(self._ns_idxs, size, replace=True, p=self._neg_prob).astype(np.int32)

    def sample_positive(self, ids):
        return np.fromiter((rnd.choice(self._element_lookup[id_]) for id_ in ids), dtype=np.int32)

    def set_embed(self, ids, embs):
        assert self._target_embedding_proximity is not None

        if self._compact_targets is False:  # when is False ids are from the graph, need to map them to dense index
            ids = [self._label_encoder._target2target_id[id] for id in ids]

        self._target_embedding_proximity.set_embed(ids, embs)

    def sample_negative(self, ids, k=1, strategy="w2v", current_group=None):
        num_ids = len(ids)

        if strategy == "w2v" or self._target_embedding_proximity.all_embeddings_ready is False:
            if strategy == "closest":
                logging.info("Proximity negative sampling is not ready yet, falling back to w2v")
            negative = self.sample_negative_w2v(ids, k=k, current_group=current_group)
        elif strategy == "closest":
            negative = self._sample_closest_negative(ids, k=k, current_group=current_group)
        else:
            raise ValueError(f"Unsupported negative sampling strategy: {strategy}. Supported values are w2v|closest")

        assert len(negative) == num_ids * k

        if self._ns_logger is not None:
            for i, n in zip(ids, negative):
                self._ns_logger.write(f"{i}\t{self._label_encoder._inverse_target_map[n]}\n")

        return negative

    def update_index(self):
        self._target_embedding_proximity.update_index()

    def has_label_mask(self):
        return {key: True for key in self._element_lookup}

    def get_groups(self):
        if self._groups is None:
            return None
        else:
            return list(self._groups.keys())


class GraphLinkTargetLoader(TargetLoader):
    def __init__(
            self, *args, **kwargs
    ):
        # assert emb_size is not None
        # assert compact_dst is False
        TargetLoader.__init__(
            self, *args, **kwargs
        )

    def sample_positive(self, ids):
        return np.fromiter(
            (self._label_encoder._inverse_target_map[rnd.choice(self._element_lookup[id_])] for id_ in ids),
            dtype=np.int32
        )

    def sample_negative(self, ids, k=1, strategy="w2v", current_group=None):
        negative = super().sample_negative(ids, k=k, strategy=strategy, current_group=current_group)

        return np.fromiter((self._label_encoder._inverse_target_map[neg] for neg in negative), dtype=np.int32)
