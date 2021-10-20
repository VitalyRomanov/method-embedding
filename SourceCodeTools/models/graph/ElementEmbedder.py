import random
from collections import Iterable

import torch
import torch.nn as nn
import numpy as np
import random as rnd

from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.train.Scorer import Scorer
from SourceCodeTools.models.nlp.TorchEncoder import LSTMEncoder, Encoder


class GraphLinkSampler(ElementEmbedderBase, Scorer):
    def __init__(self, elements, nodes, compact_dst=True, dst_to_global=True, emb_size=None):
        assert emb_size is not None
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes, compact_dst=compact_dst, dst_to_global=dst_to_global)
        Scorer.__init__(self, num_embs=len(self.elements["dst"].unique()), emb_size=emb_size, src2dst=self.element_lookup)

    def sample_negative(self, size, ids=None, strategy="closest"):
        if strategy == "w2v":
            negative = ElementEmbedderBase.sample_negative(self, size)
        else:
            negative = Scorer.sample_closest_negative(self, ids, k=size // len(ids))
            assert len(negative) == size
        return negative


class ElementEmbedder(ElementEmbedderBase, nn.Module, Scorer):
    def __init__(self, elements, nodes, emb_size, compact_dst=True):
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes, compact_dst=compact_dst)
        nn.Module.__init__(self)
        Scorer.__init__(self, num_embs=len(self.elements["dst"].unique()), emb_size=emb_size,
                        src2dst=self.element_lookup)

        self.emb_size = emb_size
        n_elems = self.elements['emb_id'].unique().size
        self.embed = nn.Embedding(n_elems, emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def __getitem__(self, ids):
        return torch.LongTensor(ElementEmbedderBase.__getitem__(self, ids=ids))

    def sample_negative(self, size, ids=None, strategy="closest"):
        # TODO
        # Try other distributions
        if strategy == "w2v":
            negative = ElementEmbedderBase.sample_negative(self, size)
        else:
            ### negative = random.choices(Scorer.sample_closest_negative(self, ids), k=size)
            negative = Scorer.sample_closest_negative(self, ids, k=size // len(ids))
            assert len(negative) == size

        return torch.LongTensor(negative)

    def forward(self, input, **kwargs):
        return self.norm(self.embed(input))

    def set_embed(self):
        all_keys = self.get_keys_for_scoring()
        with torch.set_grad_enabled(False):
            self.scorer_all_emb = self(torch.LongTensor(all_keys).to(self.embed.weight.device)).detach().cpu().numpy()

    def prepare_index(self):
        self.set_embed()
        Scorer.prepare_index(self)


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


class ElementEmbedderWithCharNGramSubwords(ElementEmbedderBase, nn.Module, Scorer):
    def __init__(self, elements, nodes, emb_size, num_buckets=5000, max_len=100, gram_size=3):
        ElementEmbedderBase.__init__(self, elements=elements, nodes=nodes, compact_dst=False)
        nn.Module.__init__(self)
        Scorer.__init__(self, num_embs=len(self.elements["dst"].unique()), emb_size=emb_size,
                        src2dst=self.element_lookup)

        self.gram_size = gram_size
        self.emb_size = emb_size
        self.init_subwords(elements, num_buckets=num_buckets, max_len=max_len)

    def init_subwords(self, elements, num_buckets, max_len):
        names = elements['dst']
        reprs = names.map(lambda x: char_ngram_window(x, self.gram_size)) \
            .map(lambda grams: (token_hasher(g, num_buckets) for g in grams)) \
            .map(lambda int_grams: np.fromiter(int_grams, dtype=np.int32)) \
            .map(lambda parts: create_fixed_length(parts, max_len, 0))

        self.name2repr = dict(zip(names, reprs))

        self.embed = nn.Embedding(num_buckets, self.emb_size, padding_idx=0)
        self.norm = nn.LayerNorm(self.emb_size)

    def __getitem__(self, ids):
        """
        Get possible targets
        :param ids: Takes a list of original ids
        :return: Matrix with subwords for passing to embedder
        """
        candidates = [rnd.choice(self.element_lookup[id]) for id in ids]
        emb_matr = np.array([self.name2repr[c] for c in candidates], dtype=np.int32)
        return torch.LongTensor(emb_matr)

    def sample_negative(self, size, ids=None, strategy="closest"):
        # TODO
        # Try other distributions
        if strategy == "w2v":
            negative = ElementEmbedderBase.sample_negative(self, size)
        else:
            ### negative = random.choices(Scorer.sample_closest_negative(self, ids), k=size)
            negative = Scorer.sample_closest_negative(self, ids, k=size // len(ids))
            assert len(negative) == size

        emb_matr = np.array([self.name2repr[c] for c in negative])
        return torch.LongTensor(emb_matr)

    def forward(self, input, **kwargs):
        x = self.embed(input)
        return self.norm(torch.mean(x, dim=1))

    def set_embed(self):
        all_keys = self.get_keys_for_scoring()
        emb_matr = np.array([self.name2repr[key] for key in all_keys], dtype=np.int32)
        with torch.set_grad_enabled(False):
            self.scorer_all_emb = self(torch.LongTensor(emb_matr).to(self.embed.weight.device)).detach().cpu().numpy()

    def prepare_index(self):
        self.set_embed()
        Scorer.prepare_index(self)


class ElementEmbedderWithBpeSubwords(ElementEmbedderWithCharNGramSubwords, nn.Module):
    def __init__(self, elements, nodes, emb_size, tokenizer_path, num_buckets=100000, max_len=10):
        self.tokenizer_path = tokenizer_path
        ElementEmbedderWithCharNGramSubwords.__init__(
            self, elements=elements, nodes=nodes, emb_size=emb_size, num_buckets=num_buckets,
            max_len=max_len
        )

    def init_subwords(self, elements, num_buckets, max_len):
        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
        tokenize = make_tokenizer(load_bpe_model(self.tokenizer_path))

        names = elements['dst']
        reprs = names.map(tokenize) \
            .map(lambda tokens: (token_hasher(t, num_buckets) for t in tokens)) \
            .map(lambda int_tokens: np.fromiter(int_tokens, dtype=np.int32))\
            .map(lambda parts: create_fixed_length(parts, max_len, 0))

        self.name2repr = dict(zip(names, reprs))

        self.embed = nn.Embedding(num_buckets, self.emb_size, padding_idx=0)
        self.norm = nn.LayerNorm(self.emb_size)


class DocstringEmbedder(ElementEmbedderWithBpeSubwords):
    def __init__(self, elements, nodes, emb_size, tokenizer_path, num_buckets=100000, max_len=100):
        super().__init__(
            elements=elements, nodes=nodes, emb_size=emb_size, tokenizer_path=tokenizer_path,
            num_buckets=num_buckets, max_len=max_len
        )
        self.encoder = LSTMEncoder(embed_dim=emb_size, num_layers=1, dropout_in=0.1, dropout_out=0.1)
        self.norm = nn.LayerNorm(emb_size)
        # self.encoder = Encoder(emb_size, emb_size, nheads=1, layers=1)

    def forward(self, input, **kwargs):
        x = self.embed(input)
        x = self.encoder(x)
        return self.norm(torch.mean(x, dim=1))  #x[:,-1,:]


class NameEmbedderWithGroups(ElementEmbedderWithBpeSubwords):
    def __init__(self, elements, nodes, emb_size, tokenizer_path, num_buckets=100000, max_len=10):
        super(NameEmbedderWithGroups, self).__init__(elements, nodes, emb_size, tokenizer_path, num_buckets, max_len)

        self.group_lookup = {}
        for dst, group_ in self.elements[["dst", "group"]].values:
            if dst not in self.element_lookup:
                self.group_lookup[group_] = []

            self.group_lookup[group_].append(dst)

        self.id2group = dict(zip(self.elements["id"], self.elements["group"]))

    def get_cand_to_score_against(self, ids):
        # TODO this way the score depends on the batch size which is bad
        all_keys = []
        [all_keys.extend(self.group_lookup[self.id2group[id]]) for id in ids]
        all_keys = list(set(all_keys))
        return all_keys




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