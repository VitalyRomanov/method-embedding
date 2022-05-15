import logging
import random
import sys
from collections import defaultdict
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
            self, targets, label_encoder, *, compact_dst=True, ns_index_backend="brute",
            scoring_method="inner_prod", emb_size=None, device="cpu", logger_path=None, logger_name=None, **kwargs
    ):
        self._compact_targets = compact_dst
        self._ns_index_backend = ns_index_backend
        self._scoring_method = scoring_method
        self._emb_size = emb_size
        self._label_encoder = label_encoder
        self._unique_targets = self._label_encoder.encoded_labels()
        self.device = device
        self._init(targets)
        self._prepare_logger(logger_path, logger_name)

        assert self._unique_targets[-1] == len(self._unique_targets) - 1
        self.num_unique_targets = len(self._unique_targets)

        self.extra_args = kwargs

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

    # def _compact_target(self, targets):
    #     if self._compact_targets:
    #         self._target2target_id, self._inverse_target_map = compact_property(targets['dst'], return_order=True)
    #     else:
    #         self._inverse_target_map = list(range(targets['dst'].max()))
    #         self._target2target_id = dict(zip(self._inverse_target_map, self._inverse_target_map))

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
            self._groups = defaultdict(list)
            for src, group in zip(targets["src"], targets["group"]):
                self._groups[group].append(src)

        self._init_w2v_ns(compacted)
        if self._emb_size is not None:
            self._init_proximity_ns(compacted)

    def _init_proximity_ns(self, targets):
        num_targets = self._label_encoder.num_unique_labels()
        self._target_embedding_cache = normalize(np.ones((num_targets, self._emb_size)), axis=1)
        # Track whether all embeddings were updated at least once. After each embedding was initialized,
        # can start proximity negative sampling
        self._cold_start_status = list(False for _ in range(len(self._unique_targets)))
        self._proximity_ns_ready = False

    def _init_w2v_ns(self, elements, skipgram_sampling_power=0.75):
        # compute distribution of dst elements
        counts = elements.value_counts(normalize=True)
        self._ns_idxs = counts.index
        self._neg_prob = counts.to_numpy()
        self._neg_prob **= skipgram_sampling_power

        self._neg_prob /= sum(self._neg_prob)

    def __len__(self):
        return len(self._element_lookup)

    def _prepare_ns_index(self):
        if self._scoring_method == "nn":
            self.scorer_index = None
            return
        if self._ns_index_backend == "sklearn":
            self._scorer_index = NearestNeighbors()
            self._scorer_index.fit(self._target_embedding_cache)
        elif self._ns_index_backend == "faiss":
            self._scorer_index = FaissIndex(self._target_embedding_cache, method=self._scoring_method)
        elif self._ns_index_backend == "brute":
            self._scorer_index = Brute(self._target_embedding_cache, method=self._scoring_method, device=self.device)
        else:
            raise ValueError(f"Unsupported backend: {self._ns_index_backend}. Supported backends are: sklearn|faiss")

    def _sample_closest_negative(self, ids, k=None):
        assert self._emb_size is not None, "Sampling closest negative is not initialized, try passing `emb_size` to initializer"
        assert ids is not None
        num_emb = self._target_embedding_cache.shape[0]
        if k > num_emb:
            logging.warning("Requested number of negative samples is larger than total number of targets")
            k = num_emb

        negative_candidates = []
        for id_ in ids:
            positive = set(self._element_lookup[id_])
            negative_candidates.append(self._get_closest_to_key(id_, k=k + len(positive) + 1, exclude=positive))

        negative = []
        for neg in negative_candidates:
            negative.extend(random.choices(neg, k=k))
        return negative

    def _get_closest_to_key(self, key, k=None, exclude=None):
        _, closest_keys = self._scorer_index.query(
            self._target_embedding_cache[key].reshape(1, -1), k=k
        )
        try:
            # Try to remove the key itself from the list of neighbours
            closest_keys = list(set(closest_keys) - {key} - exclude)
        except ValueError:
            # index throws value error if the items is not in the list
            pass

        if len(closest_keys) == 0:
            # backup strategy
            closest_keys = random.choices(list(set(self._unique_targets)), k=k)
            while key in closest_keys:
                closest_keys = random.choices(list(set(self._unique_targets)), k=k)
        return closest_keys

    def sample_negative_w2v(self, ids):
        # size = len(ids)

        negative = []
        for id_ in ids:
            excluded  = self._element_lookup[id_] + [id_]

            def get_negative():
                return np.random.choice(self._ns_idxs, 1, replace=True, p=self._neg_prob).astype(np.int32)[0]

            neg = get_negative()
            attempt_counter = 10
            while neg in excluded and attempt_counter > 0:
                neg = get_negative()
                attempt_counter -= 1
            negative.append(neg)
        return np.array(negative, dtype=np.int32)
        # return np.random.choice(self._ns_idxs, size, replace=True, p=self._neg_prob).astype(np.int32)

    def sample_positive(self, ids):
        return np.fromiter((rnd.choice(self._element_lookup[id_]) for id_ in ids), dtype=np.int32)

    def set_embed(self, ids, embs):
        if self._proximity_ns_ready is False:
            for id_ in ids:
                self._cold_start_status[id_] = True
            if all(self._cold_start_status):
                self._proximity_ns_ready = True
                self.update_index()
        self._target_embedding_cache[ids, :] = normalize(embs, axis=1) if self._scoring_method == "inner_prod" else embs

    def sample_negative(self, ids, strategy="w2v"):
        if strategy == "w2v":
            negative = self.sample_negative_w2v(ids)
        elif strategy == "closest":
            if self._proximity_ns_ready is False:
                logging.info("Proximity negative sampling is not ready yet, falling back to w2v")
                negative = self.sample_negative_w2v(ids)
            else:
                negative = self._sample_closest_negative(ids, k=len(ids) // len(ids))
                assert len(negative) == len(ids)
        else:
            raise ValueError(f"Unsupported negative sampling strategy: {strategy}. Supported values are w2v|closest")

        if self._ns_logger is not None:
            for i, n in zip(ids, negative):
                self._ns_logger.write(f"{i}\t{self._label_encoder._inverse_target_map[n]}\n")

        return negative

    def update_index(self):
        self._prepare_ns_index()

    def has_label_mask(self):
        return {key: True for key in self._element_lookup}

    def get_groups(self):
        if self._groups is None:
            return None
        else:
            return list(self._groups.keys())


class GraphLinkSampler(TargetLoader):
    def __init__(
            self, targets, *, compact_dst=True, ns_index_backend="brute",
            scoring_method="inner_prod", emb_size=None, device="cpu", **kwargs
    ):
        assert emb_size is not None
        assert compact_dst is False
        TargetLoader.__init__(
            self, targets, compact_dst=compact_dst, ns_index_backend=ns_index_backend, scoring_method=scoring_method,
            emb_size=emb_size, device=device, **kwargs
        )
