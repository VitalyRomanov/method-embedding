import numpy as np
import pandas as pd
import torch
from torch import nn

from SourceCodeTools.models.nlp.TorchEncoder import LSTMEncoder
from SourceCodeTools.nlp import token_hasher
from SourceCodeTools.nlp.embed.fasttext import char_ngram_window


class TargetEmbedder(nn.Module):
    def __init__(
            self, ind2repr, *, emb_size=None, num_buckets=200000, **kwargs
    ):
        nn.Module.__init__(self)
        self.ind2repr = ind2repr
        self.emb_size = emb_size
        self.num_buckets = num_buckets

        self.unique_targets = sorted(list(set(ind2repr.values())))
        self._create_target_input_features()
        self._create_embedding_model()

    def keys(self):
        return list(self.ind2repr.keys())

    def _create_target_input_features(self):
        # need a structure that supports indexing (dict style) and
        # return the same value as the index
        # when overriden should return actual features
        self.feature_map = list(range(len(self.unique_targets)))

    def _create_embedding_model(self):
        self.embedding_model = nn.Embedding(len(self.unique_targets), self.emb_size)
        self.norm = nn.LayerNorm(self.emb_size)

    def forward(self, indices, **kwargs):
        embs = self.embedding_model(indices)
        embs = self.norm(embs)
        return embs

    def embed_target(self, indices, **kwargs):
        features = np.array([self.feature_map[ind] for ind in indices], dtype=np.int32)
        embs = self(features, **kwargs)
        return embs


class TargetEmbedderWithCharNGramSubwords(TargetEmbedder):
    def __init__(
            self, ind2repr, *, emb_size=None, num_buckets=200000, max_len=20, **kwargs
    ):
        self.max_len = max_len
        self.gram_size = 3
        super().__init__(
            ind2repr, emb_size=emb_size, num_buckets=num_buckets, **kwargs
        )

    @staticmethod
    def _create_fixed_length(parts, length, padding_value):
        empty = np.ones((length,), dtype=np.int32) * padding_value

        empty[0:min(parts.size, length)] = parts[0:min(parts.size, length)]
        return empty

    def _create_target_input_features(self):
        names = pd.Series(self.unique_targets)
        reprs = names.map(lambda x: char_ngram_window(x, self.gram_size)) \
            .map(lambda grams: (token_hasher(g, self.num_buckets) for g in grams)) \
            .map(lambda int_grams: np.fromiter(int_grams, dtype=np.int32)) \
            .map(lambda parts: self._create_fixed_length(parts, self.max_len, self.num_buckets))

        self.feature_map = dict()
        for name, repr_ in zip(names, reprs):
            self.feature_map[self._target2target_id[name]] = repr_

    def _create_embedding_model(self):
        self.embedding_model = nn.Embedding(self.num_buckets + 1, self.emb_size, padding_idx=self.num_buckets)
        self.norm = nn.LayerNorm(self.emb_size)


class TargetEmbedderWithBpeSubwords(TargetEmbedderWithCharNGramSubwords, nn.Module):
    def __init__(
            self, ind2repr, *, emb_size=None, num_buckets=200000, max_len=20, tokenizer_path=None, **kwargs
    ):
        self.tokenizer_path = tokenizer_path
        super().__init__(
            ind2repr, emb_size=emb_size, num_buckets=num_buckets, max_len=max_len, **kwargs
        )

    def _create_target_input_features(self):
        from SourceCodeTools.nlp import create_tokenizer
        tokenize = create_tokenizer("bpe", bpe_path=self.tokenizer_path)

        inds = list(self.ind2repr.keys())
        targets = list(self.ind2repr.values())

        names = pd.Series(targets)
        reprs = names.map(tokenize) \
            .map(lambda tokens: (token_hasher(t, self.num_buckets) for t in tokens)) \
            .map(lambda int_tokens: np.fromiter(int_tokens, dtype=np.int32))\
            .map(lambda parts: self._create_fixed_length(parts, self.max_len, self.num_buckets))

        self.feature_map = dict()
        for ind, repr_ in zip(inds, reprs):
            self.feature_map[ind] = repr_


class DocstringEmbedder(TargetEmbedderWithBpeSubwords):
    def __init__(
            self, ind2repr, *, emb_size=None, num_buckets=200000, max_len=100, tokenizer_path=None, **kwargs
    ):
        super().__init__(
            ind2repr, emb_size=emb_size, num_buckets=num_buckets, max_len=max_len,
            tokenizer_path=tokenizer_path, **kwargs
        )

    def _create_embedding_model(self):
        self.embedding_model = nn.Embedding(self.num_buckets + 1, self.emb_size, padding_idx=self.num_buckets)
        self.norm = nn.LayerNorm(self.emb_size)
        self.encoder = LSTMEncoder(embed_dim=self.emb_size, num_layers=1, dropout_in=0.1, dropout_out=0.1)

    def forward(self, indices, **kwargs):
        x = self.embedding_model(indices)
        x = self.encoder(x)
        return self.norm(torch.mean(x, dim=1))  # x[:,-1,:]
