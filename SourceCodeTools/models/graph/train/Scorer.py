import random
from collections import Iterable

import torch
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.neighbors._ball_tree import BallTree
from sklearn.preprocessing import normalize


class FaissIndex:
    def __init__(self, X, *args, **kwargs):
        import faiss
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def query(self, X, k):
        return self.index.search(X.astype(np.float32), k=k)


class Scorer:
    """
    Implements sampler for triplet loss. This sampler is useful when the loss is based on the neighbourhood
    similarity. It becomes less useful when the decision is made by neural network because it does not need to mode
    points to learn how to make correct decisions.
    """
    def __init__(self, num_embs, emb_size, src2dst, neighbours_to_sample=5, index_backend="sklearn"):
        self.scorer_num_emb = num_embs
        self.scorer_emb_size = emb_size
        self.scorer_src2dst = src2dst
        self.scorer_index_backend = index_backend

        self.scorer_all_emb = np.ones((num_embs, emb_size))
        self.scorer_all_keys = self.get_cand_to_score_against(None)
        self.scorer_key_order = dict(zip(self.scorer_all_keys, range(len(self.scorer_all_keys))))
        self.scorer_index = None
        self.neighbours_to_sample = min(neighbours_to_sample, self.scorer_num_emb)

    def prepare_index(self, override_strategy=None):
        self.override_strategy=override_strategy
        if self.scorer_index_backend == "sklearn":
            self.scorer_index = BallTree(self.scorer_all_emb, leaf_size=1)
            # self.scorer_index = BallTree(normalize(self.scorer_all_emb, axis=1), leaf_size=1)
        elif self.scorer_index_backend == "faiss":
            self.scorer_index = FaissIndex(self.scorer_all_emb)
        else:
            raise ValueError(f"Unsupported backend: {self.scorer_index_backend}. Supported backends are: sklearn|faiss")

    def sample_closest_negative(self, ids, k=None):
        if k is None:
            k = self.neighbours_to_sample

        assert ids is not None
        seed_pool = []
        [seed_pool.append(self.scorer_src2dst[id]) for id in ids]
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
            closest_keys_ = list(set(closest_keys) - set(key_group))
            if len(closest_keys_) == 0:
                # backup strategy
                closest_keys_ = random.choices(list(set(self.scorer_all_keys) - set(key_group)), k=k-1)
            possible_targets.append(closest_keys_)
            # possible_targets.extend(self.scorer_all_keys[c] for c in closest.ravel())
        return possible_targets
        # return list(set(possible_targets) - set(keys))

    def set_embed(self, ids, embs):

        ids = np.array(list(map(self.scorer_key_order.get, ids.tolist())))
        self.scorer_all_emb[ids, :] = embs# normalize(embs, axis=1)

        # for ind, id in enumerate(ids):
        #     self.all_embs[self.key_order[id], :] = embs[ind, :]

    def score_candidates_cosine(self, to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, at=None):

        score_matr = (to_score_embs @ embs_to_score_against.t()) / \
                     to_score_embs.norm(p=2, dim=1, keepdim=True) / \
                     embs_to_score_against.norm(p=2, dim=1, keepdim=True).t()
        y_pred = score_matr.tolist()

        return y_pred

    def score_candidates_lp(self, to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, link_predictor, at=None):

        y_pred = []
        for i in range(len(to_score_ids)):
            input_embs = to_score_embs[i, :].repeat((embs_to_score_against.shape[0], 1))
            # predictor_input = torch.cat([input_embs, all_emb], dim=1)
            y_pred.append(torch.nn.functional.softmax(link_predictor(input_embs, embs_to_score_against), dim=1)[:, 1].tolist())  # 0 - negative, 1 - positive

        return y_pred

    def get_gt_candidates(self, ids):
        candidates = [set(list(self.scorer_src2dst[id])) for id in ids]
        return candidates

    def get_cand_to_score_against(self, ids):
        all_keys = set()

        [all_keys.update(self.scorer_src2dst[key]) for key in self.scorer_src2dst]
        return sorted(list(all_keys))  # list(self.elem2id[a] for a in all_keys)

    def get_embeddings_for_scoring(self, device, **kwargs):
        return torch.Tensor(self.scorer_all_emb).to(device)

    def get_keys_for_scoring(self):
        return self.scorer_all_keys

    def score_candidates(self, to_score_ids, to_score_embs, link_predictor=None, at=None, type=None, device="cpu"):

        if at is None:
            at = [1, 3, 5, 10]

        to_score_ids = to_score_ids.tolist()

        candidates = self.get_gt_candidates(to_score_ids)
        # keys_to_score_against = self.get_cand_to_score_against(to_score_ids)
        keys_to_score_against = self.get_keys_for_scoring()

        y_true = [[1. if key in cand else 0. for key in keys_to_score_against] for cand in candidates]

        embs_to_score_against = self.get_embeddings_for_scoring(device=to_score_embs.device)

        if type == "nn":
            y_pred = self.score_candidates_lp(
                to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against,
                link_predictor, at=at
            )
        elif type == "inner_prod":
            y_pred = self.score_candidates_cosine(
                to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, at=at
            )
        else:
            raise ValueError(f"`type` can be either `nn` or `inner_prod` but `{type}` given")

        if isinstance(at, Iterable):
            scores = {f"ndcg@{k}": ndcg_score(y_true, y_pred, k=k) for k in at}
        else:
            scores = {f"ndcg@{at}": ndcg_score(y_true, y_pred, k=at)}
        return scores
