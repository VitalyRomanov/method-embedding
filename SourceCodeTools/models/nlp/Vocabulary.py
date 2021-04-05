from collections import Counter
from random import choice

import numpy as np
import pickle as p


class Vocabulary:
    def __init__(self, pad='<pad>', eos='</s>', unk='▁<unk>'):
        self.count = Counter()
        self.ids = {}
        self.inv_ids = []
        self.prob_valid = False

        self.pad = pad
        self.eos = eos
        self.unk = unk

        self.add(["<Lua heritage>", self.pad, self.eos, self.unk])

        self.pad_id = self.ids[pad]
        self.eos_id = self.ids[eos]
        self.unk_id = self.ids[unk]

    def add(self, tokens):
        for token in tokens:
            self.add_token(token)

        self.prob_valid = False

    def add_token(self, token):
        if token in self.ids:
            self.count[self.ids[token]] += 1
        else:
            new_id = len(self.ids)
            self.ids[token] = new_id
            self.inv_ids.append(token)
            self.count[new_id] = 1

    def drop_oov(self, tokens):
        return (self.is_oov(t) for t in tokens)

    def is_oov(self, token):
        return token in self.ids

    @property
    def total_words(self):
        if not self.prob_valid:
            self.calculate_prob()
        return self.word_count_sum

    def tokens2ids(self, tokens):
        return [self.ids.get(token, 0) for token in tokens]

    def ids2tokens(self, ids):
        return [self.inv_ids[token_id] if token_id < len(self.inv_ids) else self.unk for token_id in ids]

    def __len__(self):
        return len(self.count)

    def __getitem__(self, item):
        if type(item) == str:
            return self.ids[item]
        elif type(item) == int:
            return self.inv_ids[item] if item < len(self.inv_ids) else self.unk
        else:
            raise KeyError("")

    def most_common(self, length=None):
        if length is None:
            length = len(self)
        return [(token_id, self.inv_ids[token_id], freq) for token_id, freq in self.count.most_common(length)]

    def values(self):
        return self.count.values()

    def calculate_prob(self):
        p = np.array(list(self.count.values()))
        self.word_count_sum = np.sum(p)
        self.p = p / self.word_count_sum
        self.prob_valid = True

    def sample(self, n_samples, limit_top=-1):
        # make sure that counts follow the order of ids

        if limit_top == -1:
            limit_top = len(self.count)

        if not self.prob_valid:
            self.calculate_prob()

        if limit_top != -1:
            p = self.p[:limit_top] / np.sum(self.p[:limit_top])
        else:
            p = self.p

        return list(choice(limit_top, size=n_samples, p=p))

    def save(self, destination):
        p.dump(self, open(destination, "wb"))

    @classmethod
    def load(self, location):
        return p.load(open(location, "rb"))