import logging
import time
from collections import Iterable

import torch
import numpy as np
from sklearn.metrics import ndcg_score, top_k_accuracy_score


class Scorer:
    """
    Implements sampler for triplet loss. This sampler is useful when the loss is based on the neighbourhood
    similarity. It becomes less useful when the decision is made by neural network because it does not need to mode
    points to learn how to make correct decisions.
    """
    def __init__(
            self, target_loader, margin, device="cpu"
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
        self.target_loader = target_loader
        self.device = device
        self.margin = margin

    def score_candidates_cosine(self, to_score_ids, to_score_embs, keys_to_score_against, embs_to_score_against, at=None):

        to_score_embs = to_score_embs / to_score_embs.norm(p=2, dim=1, keepdim=True)
        embs_to_score_against = embs_to_score_against / embs_to_score_against.norm(p=2, dim=1, keepdim=True)

        score_matr = (to_score_embs @ embs_to_score_against.t())
        score_matr = (score_matr + 1.) / 2.
        # score_matr = score_matr - self.margin
        # score_matr[score_matr < 0.] = 0.
        y_pred = score_matr.cpu().tolist()

        return y_pred

    # def set_margin(self, margin):
    #     self.margin = margin

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

    def get_ground_truth_candidates(self, ids):
        candidates = [set(list(self.target_loader._element_lookup[id])) for id in ids]
        return candidates

    # def get_cand_to_score_against(self):
    #     return list(range(len(self.target_loader._inverse_target_map)))

    def get_embeddings_for_scoring(self, device, **kwargs):
        return torch.Tensor(self.target_loader.target_embeddings).to(device)

    def get_keys_for_scoring(self):
        return list(range(len(self.target_loader._unique_targets)))

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

        if self.target_loader.embedding_proximity_ready is False:
            logging.warning("Scoring is unreliable")

        if at is None:
            at = [1, 3, 5, 10]

        start = time.time()

        # to_score_ids = to_score_ids.tolist()

        candidates = self.get_ground_truth_candidates(to_score_ids)  # positive candidates
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
