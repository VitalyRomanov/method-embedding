import torch
import torch.nn as nn
import numpy as np
import random as rnd

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


if __name__ == '__main__':
    import pandas as pd
    test_data = pd.DataFrame({
        "id": [0, 1, 2, 3, 4, 4, 5],
        # "dst": [6, 11, 12, 11, 14, 15, 16]
        "dst": ["hello", "how", "are", "you", "doing", "today", "?"]
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