import random
import time
from collections import Iterable, defaultdict
from typing import Dict, List

import torch
import numpy as np
from sklearn.metrics import ndcg_score, top_k_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors._ball_tree import BallTree
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

        return dist, ind#.reshape((-1,1))


class Scorer:
    """
    Implements sampler for triplet loss. This sampler is useful when the loss is based on the neighbourhood
    similarity. It becomes less useful when the decision is made by neural network because it does not need to mode
    points to learn how to make correct decisions.
    """
    def __init__(
            self, num_embs, emb_size, src2dst: Dict[int, List[int]], neighbours_to_sample=5, index_backend="brute",
            method = "inner_prod", device="cpu", ns_groups=None
    ):
        """
        Creates an embedding table, the embeddings in this table are updated once during an epoch. Embeddings from this
        table are used for nearest neighbour queries during negative sampling. We avoid keeping track of all possible
        embeddings by knowing that only part of embeddings are eligible as DST.
        :param num_embs: number of unique DST
        :param emb_size: embedding dimensionality
        :param src2dst: Mapping from SRC to all DST, need this to find the hardest negative example for all DST at once
        :param neighbours_to_sample: default number of neighbours
        :param index_backend: Choose between sklearn and faiss
        """
        self.scorer_num_emb = num_embs
        self.scorer_emb_size = emb_size
        self.scorer_src2dst = src2dst  # mapping from src to all possible dst
        self.scorer_index_backend = index_backend
        self.scorer_method = method
        self.scorer_device = device

        self.scorer_all_emb = normalize(np.ones((num_embs, emb_size)), axis=1)  # unique dst embedding table
        self.scorer_all_keys = self.get_cand_to_score_against(None)
        self.scorer_key_order = dict(zip(self.scorer_all_keys, range(len(self.scorer_all_keys))))
        self.scorer_index = None
        self.neighbours_to_sample = min(neighbours_to_sample, self.scorer_num_emb)
        self.prepare_ns_groups(ns_groups)

    def prepare_ns_groups(self, ns_groups):
        if ns_groups is None:
            return

        self.scorer_node2ns_group = {}
        self.scorer_ns_group2nodes = defaultdict(list)

        unique_dst = set()
        for dsts in self.scorer_src2dst.values():
            for dst in dsts:
                if isinstance(dst, tuple):
                    unique_dst.add(dst[0])
                else:
                    unique_dst.add(dst)

        for id, mentioned_in in ns_groups.values:
            id_ = self.id2nodeid[id]
            mentioned_in_ = self.id2nodeid[mentioned_in]
            self.scorer_node2ns_group[id_] = mentioned_in_
            if id_ in unique_dst:
                self.scorer_ns_group2nodes[mentioned_in_].append(id_)

    def sample_negative_from_groups(self, key_groups, k):
        possible_targets = []
        for key_group in key_groups:
            any_key = key_group[0]
            possible_negative = self.scorer_ns_group2nodes[self.scorer_node2ns_group[any_key]]
            possible_negative_ = list(set(possible_negative) - set(key_group))
            if len(possible_negative_) < k:
                # backup strategy
                possible_negative_.extend(
                    random.choices(list(set(self.scorer_all_keys) - set(key_group)), k=k - len(possible_negative_))
                )
            possible_targets.append(possible_negative_)
        return possible_targets


    def prepare_index(self, override_strategy=None):
        if self.scorer_method == "nn":
            self.scorer_index = None
            return
        if self.scorer_index_backend == "sklearn":
            self.scorer_index = NearestNeighbors()
            self.scorer_index.fit(self.scorer_all_emb)
            # self.scorer_index = BallTree(self.scorer_all_emb, leaf_size=1)
            # self.scorer_index = BallTree(normalize(self.scorer_all_emb, axis=1), leaf_size=1)
        elif self.scorer_index_backend == "faiss":
            self.scorer_index = FaissIndex(self.scorer_all_emb, method=self.scorer_method)
        elif self.scorer_index_backend == "brute":
            self.scorer_index = Brute(self.scorer_all_emb, method=self.scorer_method, device=self.scorer_device)
        else:
            raise ValueError(f"Unsupported backend: {self.scorer_index_backend}. Supported backends are: sklearn|faiss")

    def sample_closest_negative(self, ids, k=None):
        if k is None:
            k = self.neighbours_to_sample

        assert ids is not None
        seed_pool = []
        for id in ids:
            pool = self.scorer_src2dst[id]
            if len(pool) > 0 and isinstance(pool[0], tuple):
                pool = [x[0] for x in pool]
            seed_pool.append(pool)
            if id in self.scorer_key_order:
                seed_pool[-1] = seed_pool[-1] + [id]  # make sure that original list is not changed
        # [seed_pool.append(self.scorer_src2dst[id]) for id in ids]
        if hasattr(self, "scorer_ns_group2nodes"):
            nested_negative = self.sample_negative_from_groups(seed_pool, k=k+1)
        else:
            nested_negative = self.get_closest_to_keys(seed_pool, k=k+1)

        negative = []
        for neg in nested_negative:
            negative.extend(random.choices(neg, k=k))
        return negative

    def get_closest_to_keys(self, key_groups, k=None):
        possible_targets = []
        for key_group in key_groups:
            closest_keys = []
            for key in key_group:
                _, closest = self.scorer_index.query(
                    self.scorer_all_emb[self.scorer_key_order[key]].reshape(1, -1), k=k
                )
                closest_keys.extend(self.scorer_all_keys[c] for c in closest.ravel())
            # ensure that negative samples do not come from positive edges
            closest_keys_ = list(set(closest_keys) - set(key_group))
            if len(closest_keys_) == 0:
                # backup strategy
                closest_keys_ = random.choices(list(set(self.scorer_all_keys) - set(key_group)), k=k)
            possible_targets.append(closest_keys_)
            # possible_targets.extend(self.scorer_all_keys[c] for c in closest.ravel())
        return possible_targets
        # return list(set(possible_targets) - set(keys))

    def set_embed(self, ids, embs):

        ids = np.array(list(map(self.scorer_key_order.get, ids.tolist())))
        self.scorer_all_emb[ids, :] = normalize(embs, axis=1) if self.scorer_method == "inner_prod" else embs

        # for ind, id in enumerate(ids):
        #     self.all_embs[self.key_order[id], :] = embs[ind, :]

    def score_candidates_cosine(self, to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, at=None):

        to_score_embs = to_score_embs / to_score_embs.norm(p=2, dim=1, keepdim=True)
        embs_to_score_against = embs_to_score_against / embs_to_score_against.norm(p=2, dim=1, keepdim=True)

        score_matr = (to_score_embs @ embs_to_score_against.t())
        score_matr = (score_matr + 1.) / 2.
        # score_matr = score_matr - self.margin
        # score_matr[score_matr < 0.] = 0.
        y_pred = score_matr.cpu().tolist()

        return y_pred

    def set_margin(self, margin):
        self.margin = margin

    def score_candidates_l2(self, to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, at=None):

        y_pred = []
        for i in range(len(to_score_ids)):
            input_embs = to_score_embs[i, :].reshape(1, -1)
            score_matr = torch.norm(embs_to_score_against - input_embs, dim=-1)
            score_matr = 1. / (1. + score_matr)
            # score_matr = score_matr + self.margin
            # score_matr[score_matr < 0.] = 0
            y_pred.append(score_matr.cpu().tolist())

        # embs_to_score_against = embs_to_score_against.unsqueeze(0)
        # to_score_embs = to_score_embs.unsqueeze(1)
        #
        # score_matr = -torch.norm(embs_to_score_against - to_score_embs, dim=-1)
        # score_matr = score_matr + self.margin
        # score_matr[score_matr < 0.] = 0
        # y_pred = score_matr.cpu().tolist()

        return y_pred

    def score_candidates_lp(
            self, to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, link_predictor, at=None,
            with_types=None
    ):

        if with_types is None:
            y_pred = []
            for i in range(len(to_score_ids)):
                input_embs = to_score_embs[i, :].repeat((embs_to_score_against.shape[0], 1))
                # predictor_input = torch.cat([input_embs, all_emb], dim=1)
                y_pred.append(
                    torch.nn.functional.softmax(link_predictor(input_embs, embs_to_score_against), dim=1)[:, 1].tolist()
                )  # 0 - negative, 1 - positive

            return y_pred

        else:
            y_pred = []
            for i in range(len(to_score_ids)):
                y_pred.append(dict())
                for type in with_types[i]:
                    input_embs = to_score_embs[i, :].unsqueeze(0)
                    # predictor_input = torch.cat([input_embs, all_emb], dim=1)

                    labels = torch.LongTensor([type]).to(link_predictor.proj_matr.weight.device)
                    weights = link_predictor.proj_matr(labels).reshape((-1, link_predictor.rel_dim, link_predictor.input_dim))
                    rels = link_predictor.rel_emb(labels)
                    m_a = (weights * input_embs.unsqueeze(1)).sum(-1)
                    m_s = (weights * embs_to_score_against.unsqueeze(1)).sum(-1)

                    transl = m_a + rels
                    sim = torch.norm(transl - m_s, dim=-1)
                    sim = 1./ (1. + sim)
                    y_pred[-1][type] = sim.cpu().tolist()

            return y_pred


    def get_gt_candidates(self, ids):
        candidates = [set(list(self.scorer_src2dst[id])) for id in ids]
        return candidates

    def get_cand_to_score_against(self, ids):
        """
        Generate sorted list of all possible DST. These will be used as possible targets during NDCG calculation
        :param ids:
        :return:
        """
        all_keys = set()

        for key in self.scorer_src2dst:
            cand = self.scorer_src2dst[key]
            for c in cand:
                if isinstance(c, tuple):  # happens with graphlinkclassifier objectives
                    all_keys.add(c[0])
                else:
                    all_keys.add(c)

        # [all_keys.update(self.scorer_src2dst[key]) for key in self.scorer_src2dst]
        return sorted(list(all_keys))  # list(self.elem2id[a] for a in all_keys)

    def get_embeddings_for_scoring(self, device, **kwargs):
        """
        Get all embeddings as a tensor
        :param device:
        :param kwargs:
        :return:
        """
        return torch.Tensor(self.scorer_all_emb).to(device)

    def get_keys_for_scoring(self):
        return self.scorer_all_keys

    def hits_at_k(self, y_true, y_pred, k):
        correct = y_true
        predicted = y_pred
        result = []
        for y_true, y_pred in zip(correct, predicted):
            ind_true = set([ind for ind, y_t in enumerate(y_true) if y_t == 1.])
            ind_pred = set(list(np.flip(np.argsort(y_pred))[:k]))
            result.append(len(ind_pred.intersection(ind_true)) / min(len(ind_true), k))

        return sum(result) / len(result)

    def mean_rank(self, y_true, y_pred):
        true_ranks = []
        reciprocal_ranks = []

        correct = y_true
        predicted = y_pred
        for y_true, y_pred in zip(correct, predicted):
            ranks = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
            for ind, (true, pred) in enumerate(ranks):
                if true > 0.:
                    true_ranks.append(ind + 1)
                    reciprocal_ranks.append(1 / (ind + 1))
                    break  # should consider only first result https://en.wikipedia.org/wiki/Mean_reciprocal_rank

        return sum(true_ranks) / len(true_ranks), sum(reciprocal_ranks) / len(reciprocal_ranks)

    def mean_average_precision(self, y_true, y_pred):
        correct = y_true
        predicted = y_pred
        map = 0.
        for y_true, y_pred in zip(correct, predicted):
            ranks = sorted(zip(y_true, y_pred), key=lambda x: x[1], reverse=True)
            found_relevant = 0
            avep = 0.
            for ind, (true, pred) in enumerate(ranks):
                if true > 0.:
                    found_relevant += 1
                    avep += found_relevant / (ind + 1)  # precision@k
            avep /= found_relevant

            map += avep

        map /= len(correct)

        return map

    def get_y_true_from_candidate_list(self, candidates, keys_to_score_against):
        y_true = [[1. if key in cand else 0. for key in keys_to_score_against] for cand in candidates]
        return y_true

    def get_y_true_from_candidates(self, candidates, keys_to_score_against):

        has_types = isinstance(list(candidates[0])[0], tuple)

        if not has_types:
            return self.get_y_true_from_candidate_list(candidates, keys_to_score_against)

        candidate_dicts = []
        for cand in candidates:
            candidate_dicts.append(dict())
            for ent, type in cand:
                if type not in candidate_dicts[-1]:
                    candidate_dicts[-1][type] = []
                candidate_dicts[-1][type].append(ent)

        y_true = []
        for cand in candidate_dicts:
            y_true.append(dict())
            for type, ents in cand.items():
                y_true[-1][type] = [1. if key in ents else 0. for key in keys_to_score_against]
                assert sum(y_true[-1][type]) > 0.

        return y_true

    def flatten_pred(self, y):

        flattened = []
        for x in y:
            for key, scores in x.items():
                flattened.append(scores)

        return flattened

    def score_candidates(self, to_score_ids, to_score_embs, link_predictor=None, at=None, type=None, device="cpu"):

        if at is None:
            at = [1, 3, 5, 10]

        start = time.time()

        to_score_ids = to_score_ids.tolist()

        candidates = self.get_gt_candidates(to_score_ids)  # positive candidates
        # keys_to_score_against = self.get_cand_to_score_against(to_score_ids)
        keys_to_score_against = self.get_keys_for_scoring()

        y_true = self.get_y_true_from_candidates(candidates, keys_to_score_against)

        embs_to_score_against = self.get_embeddings_for_scoring(device=to_score_embs.device)

        if type == "nn":
            has_types = isinstance(y_true[0], dict)

            y_pred = self.score_candidates_lp(
                to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against,
                link_predictor, at=at, with_types=y_true if has_types else None
            )
        elif type == "inner_prod":
            y_pred = self.score_candidates_cosine(
                to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, at=at
            )
        elif type == "l2":
            y_pred = self.score_candidates_l2(
                to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, at=at
            )
        else:
            raise ValueError(f"`type` can be either `nn` or `inner_prod` but `{type}` given")

        has_types = isinstance(y_pred[0], dict)

        if has_types:
            y_true = self.flatten_pred(y_true)
            y_pred = self.flatten_pred(y_pred)

        scores = {}
        # y_true_onehot = np.array(y_true)
        # labels=list(range(y_true_onehot.shape[1]))

        if isinstance(at, Iterable):
            scores.update({f"hits@{k}": self.hits_at_k(y_true, y_pred, k=k) for k in at})
            scores.update({f"ndcg@{k}": ndcg_score(y_true, y_pred, k=k) for k in at})
            # scores = {f"ndcg@{k}": ndcg_score(y_true, y_pred, k=k) for k in at}
        else:
            scores.update({f"hits@{at}": self.hits_at_k(y_true, y_pred, k=at)})
            scores.update({f"ndcg@{at}": ndcg_score(y_true, y_pred, k=at)})
            # scores = {f"ndcg@{at}": ndcg_score(y_true, y_pred, k=at)}

        mr, mrr = self.mean_rank(y_true, y_pred)
        map = self.mean_average_precision(y_true, y_pred)
        scores["mr"] = mr
        scores["mrr"] = mrr
        scores["map"] = map
        scores["scoring_time"] = time.time() - start
        return scores
