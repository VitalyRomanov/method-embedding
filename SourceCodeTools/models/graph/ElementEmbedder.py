from collections import Iterable

import torch
import torch.nn as nn
import numpy as np
import random as rnd

from sklearn.metrics import ndcg_score
from sklearn.neighbors._ball_tree import BallTree
from sklearn.preprocessing import normalize

from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase

class ElementEmbedder(ElementEmbedderBase, nn.Module):
    def __init__(self, elements, nodes, emb_size, compact_dst=True):
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes, compact_dst=compact_dst)
        nn.Module.__init__(self)

        self.emb_size = emb_size
        n_elems = self.elements['emb_id'].unique().size
        self.embed = nn.Embedding(n_elems, emb_size)

    def __getitem__(self, ids):
        return torch.LongTensor(ElementEmbedderBase.__getitem__(self, ids=ids))

    def forward(self, input, **kwargs):
        return self.embed(input)


# def hashstr(s, num_buckets):
#     return int(hashlib.md5(s.encode('utf8')).hexdigest(), 16) % num_buckets


# def window(x, gram_size):
#     x = "<" + x + ">"
#     length = len(x)
#     return (x[i:i + gram_size] for i in range(0, length) if i+gram_size<=length)


def create_fixed_length(parts, length, padding_value):
    empty = np.ones((length,), dtype=np.int32) * padding_value

    empty[0:min(parts.size, length)] = parts[0:min(parts.size, length)]
    return empty


from SourceCodeTools.nlp import token_hasher
from SourceCodeTools.nlp.embed.fasttext import char_ngram_window


class ElementEmbedderWithCharNGramSubwords(ElementEmbedderBase, nn.Module):
    def __init__(self, elements, nodes, emb_size, num_buckets=5000, max_len=100, gram_size=3):
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes, compact_dst=False)
        nn.Module.__init__(self)

        self.emb_size = emb_size

        names = elements['dst']
        reprs = names.map(lambda x: char_ngram_window(x, gram_size)) \
            .map(lambda grams: (token_hasher(g, num_buckets) for g in grams)) \
            .map(lambda int_grams: np.fromiter(int_grams, dtype=np.int32))\
            .map(lambda parts: create_fixed_length(parts, max_len, 0))

        self.name2repr = dict(zip(names, reprs))

        self.embed = nn.Embedding(num_buckets, emb_size, padding_idx=0)

    def __getitem__(self, ids):
        """
        Get possible targets
        :param ids: Takes a list of original ids
        :return: Matrix with subwords for passing to embedder
        """
        candidates = [rnd.choice(self.element_lookup[id]) for id in ids]
        emb_matr = np.array([self.name2repr[c] for c in candidates], dtype=np.int32)
        return torch.LongTensor(emb_matr)

    def sample_negative(self, size):
        # TODO
        # Try other distributions
        emb_matr = np.array([self.name2repr[c] for c in np.random.choice(self.idxs, size, replace=True, p=self.neg_prob)])
        return torch.LongTensor(emb_matr)

    def forward(self, input, **kwargs):
        x = self.embed(input)
        return torch.mean(x, dim=1)

    def prepare_index(self):
        all_keys = [key for key in self.name2repr]
        emb_matr = np.array([self.name2repr[key] for key in all_keys], dtype=np.int32)
        all_emb = self(torch.LongTensor(emb_matr)).detach().numpy()
        self.all_emb = normalize(all_emb, axis=1)
        self.ball_tree = BallTree(all_emb)

    def score_candidates_cosine(self, to_score_ids, to_score_embs, at=None):
        if at is None:
            at = [1, 3, 5, 10]

        ids = to_score_ids.tolist()
        candidates = [set(list(self.element_lookup[id])) for id in ids]
        all_keys = [key for key in self.name2repr]
        # emb_matr = np.array([self.name2repr[key] for key in all_keys], dtype=np.int32)
        # all_emb = self(torch.LongTensor(emb_matr)).detach().numpy()
        # all_emb = normalize(all_emb, axis=1)
        # ball_tree = BallTree(all_emb)

        y_true = [[1. if all_keys[i] in cand else 0. for i in range(len(all_keys))] for cand in candidates]

        y_pred = []
        to_score_embs = to_score_embs.detach().numpy()
        to_score_embs = normalize(to_score_embs, axis=1)
        for ind in range(to_score_embs.shape[0]):
            curr_to_score = to_score_embs[ind].reshape(1, -1)
            _, closest = self.ball_tree.query(curr_to_score, k=at if type(at) is int else max(at))
            scores = (curr_to_score @ self.all_emb[closest.ravel(), :].T).ravel()
            closest = closest.ravel().tolist()
            y_pred.append([])
            for i in range(len(all_keys)):
                if i in closest:
                    y_pred[-1].append(scores[closest.index(i)])
                else:
                    y_pred[-1].append(-1.)

        # score_matr = (to_score_embs @ all_emb.t()) / \
        #             to_score_embs.norm(p=2, dim=1, keepdim=True) / \
        #             all_emb.norm(p=2, dim=1, keepdim=True).t()
        # y_pred = score_matr.tolist()

        if isinstance(at, Iterable):
            scores = {f"ndcg@{k}": ndcg_score(y_true, y_pred, k=k) for k in at}
        else:
            scores = {f"ndcg@{at}": ndcg_score(y_true, y_pred, k=at)}
        return scores

    def score_candidates_lp(self, to_score_ids, to_score_embs, link_predictor, at=None):
        if at is None:
            at = [1, 3, 5, 10]

        ids = to_score_ids.tolist()
        candidates = [set(list(self.element_lookup[id])) for id in ids]
        all_keys = [key for key in self.name2repr]
        emb_matr = np.array([self.name2repr[key] for key in all_keys], dtype=np.int32)
        all_emb = self(torch.LongTensor(emb_matr))

        y_true = [[1. if all_keys[i] in cand else 0. for i in range(len(all_keys))] for cand in candidates]

        y_pred = []
        for i in range(len(to_score_ids)):
            input_embs = to_score_embs[i, :].repeat((all_emb.shape[0], 1))
            # predictor_input = torch.cat([input_embs, all_emb], dim=1)
            y_pred.append(link_predictor(input_embs, all_emb)[:, 1].tolist())  # 0 - negative, 1 - positive

        if isinstance(at, Iterable):
            scores = {f"ndcg@{k}": ndcg_score(y_true, y_pred, k=k) for k in at}
        else:
            scores = {f"ndcg@{at}": ndcg_score(y_true, y_pred, k=at)}
        return scores

    def score_candidates(self, to_score_ids, to_score_embs, link_predictor=None, at=None, type=None):
        if type == "nn":
            scores = self.score_candidates_lp(to_score_ids, to_score_embs, link_predictor, at=at)
        elif type == "inner_prod":
            scores = self.score_candidates_cosine(to_score_ids, to_score_embs, at=at)
        else:
            raise ValueError(f"`type` can be either `nn` or `inner_prod` but `{type}` given")
        return scores





class ElementEmbedderWithBpeSubwords(ElementEmbedderWithCharNGramSubwords, nn.Module):
    def __init__(self, elements, nodes, emb_size, tokenizer_path, num_buckets=100000, max_len=10):
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes, compact_dst=False)
        nn.Module.__init__(self)

        self.emb_size = emb_size

        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
        tokenize = make_tokenizer(load_bpe_model(tokenizer_path))

        names = elements['dst']
        reprs = names.map(tokenize) \
            .map(lambda tokens: (token_hasher(t, num_buckets) for t in tokens)) \
            .map(lambda int_tokens: np.fromiter(int_tokens, dtype=np.int32))\
            .map(lambda parts: create_fixed_length(parts, max_len, 0))

        self.name2repr = dict(zip(names, reprs))

        self.embed = nn.Embedding(num_buckets, emb_size, padding_idx=0)


def test_ElementEmbedderWithCharNGramSubwords_score_candidates():
    import pandas as pd
    test_nodes = pd.DataFrame({
        "id": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        "global_graph_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "typed_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "type": ["node_", "node_", "node_", "node_", "node_", "node_", "node_", "node_", "node_", "node_"]
    })
    test_data = pd.DataFrame({
        "src": [0, 1, 2, 3, 4, 4, 5, 9],
        "dst": ["hello", "how", "are", "you", "doing", "today", "?", "?!"]
    })
    ee = ElementEmbedderWithCharNGramSubwords(test_data, test_nodes, 5, max_len=10)
    # provide original ids as input
    ee.score_candidates(torch.LongTensor([9, 7, 8]), torch.Tensor(np.random.rand(3, 5)))





if __name__ == '__main__':
    import pandas as pd
    test_data = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 4, 5, 9],
        # "dst": [6, 11, 12, 11, 14, 15, 16]
        "dst": ["hello", "how", "are", "you", "doing", "today", "?", "?!"]
    })

    # ee = ElementEmbedder(test_data, 5)
    ee = ElementEmbedderWithCharNGramSubwords(test_data, 5, max_len=10)

    rand_ind = [0, 1, 2, 3, 4, 4, 5]#np.random.randint(low=0, high=len(ee), size=20)
    sample = ee[rand_ind]
    ee(sample)
    from pprint import pprint
    pprint(list(zip(rand_ind, sample)))
    print(ee.elem_probs)
    print(ee.elem2id)
    print(ee.sample_negative(3))