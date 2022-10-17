from abc import ABC
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScorerObjectiveOptimizationDirection(Enum):
    minimize = 0
    maximize = 1


class AbstractLinkScorer(nn.Module, ABC):
    def __init__(self, input_dim1, input_dim2, num_classes, h_dim, **kwargs):
        super(AbstractLinkScorer, self).__init__()
        self._input_dim1 = input_dim1
        self._input_dim2 = input_dim2
        self._num_classes = num_classes
        self._h_dim = h_dim

        self._default_positive_label = 1.0
        self._default_negative_label = 0.0
        self._optimization_direction = None
        self._default_label_dtype = None

    def forward(self, x1, x2, **kwargs):
        # Similarity score depends on the type of the metric used
        #
        raise NotImplementedError()

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def label_dtype(self):
        return self._default_label_dtype

    @property
    def optimization_direction(self):
        return self._optimization_direction

    @property
    def positive_label(self):
        return self._default_positive_label

    @property
    def negative_label(self):
        return self._default_negative_label


class AbstractCosineLinkScorer(AbstractLinkScorer, ABC):
    def __init__(self, input_dim1, input_dim2, num_classes, h_dim, **kwargs):
        super(AbstractCosineLinkScorer, self).__init__(input_dim1, input_dim2, num_classes, h_dim)

        self._optimization_direction = ScorerObjectiveOptimizationDirection.maximize
        self._default_positive_label = 1.0
        self._default_negative_label = -1.0
        self._default_label_dtype = torch.float32


class AbstractNNLinkScorer(AbstractLinkScorer, ABC):
    def __init__(self, input_dim1, input_dim2, num_classes, h_dim, **kwargs):
        super(AbstractNNLinkScorer, self).__init__(input_dim1, input_dim2, num_classes, h_dim)

        self._optimization_direction = ScorerObjectiveOptimizationDirection.minimize
        self._default_positive_label = 1
        self._default_negative_label = 0
        self._default_label_dtype = torch.long

    @property
    def positive_label(self):
        if self._num_classes > 2:
            return None
        return self._default_positive_label


class AbstractL2LinkScorer(AbstractLinkScorer, ABC):
    def __init__(self, input_dim1, input_dim2, num_classes, h_dim, **kwargs):
        super(AbstractL2LinkScorer, self).__init__(input_dim1, input_dim2, num_classes, h_dim)

        self._optimization_direction = ScorerObjectiveOptimizationDirection.minimize
        self._default_positive_label = 1.  # 0
        self._default_negative_label = -1.  # float("-Inf")
        self._default_label_dtype = torch.float32


class LinkClassifier(AbstractNNLinkScorer):
    def __init__(self, input_dim1, input_dim2, num_classes=2, h_dim=300, **kwargs):
        super(LinkClassifier, self).__init__(input_dim1, input_dim2, num_classes, h_dim)
        concat_emb_size = self._input_dim1 + self._input_dim1

        # self.norm = nn.BatchNorm1d(concat_emb_size) # LayerNorm
        self.l1 = nn.Linear(concat_emb_size, self._h_dim)
        self.logits = nn.Linear(self._h_dim, num_classes)

    def forward(self, x1, x2, **kwargs):
        x = torch.cat([x1, x2], dim=-1)
        # x = self.norm(x)
        x = F.relu(self.l1(x))
        return self.logits(x)


class BilinearLinkClassifier(AbstractNNLinkScorer):
    def __init__(self, input_dim1, input_dim2, num_classes=2, h_dim=200, **kwargs):
        super(BilinearLinkClassifier, self).__init__(input_dim1, input_dim2, num_classes, h_dim)

        self.l1_1 = nn.Linear(self._input_dim1, self._h_dim, bias=False)
        self.l1_2 = nn.Linear(self._h_dim, self._h_dim, bias=False)
        self.l2_1 = nn.Linear(self._input_dim2, self._h_dim, bias=False)
        self.l2_2 = nn.Linear(self._h_dim, self._h_dim, bias=False)

        self.bilinear = nn.Bilinear(self._h_dim, self._h_dim, self._num_classes)

    def forward(self, x1, x2, **kwargs):

        x1_l1 = self.l1_2(F.relu(self.l1_1(x1)))
        x2_l2 = self.l2_2(F.relu(self.l2_1(x2)))

        return self.bilinear(x1_l1, x2_l2)


class UndirectedCosineLinkScorer(AbstractCosineLinkScorer):
    def __init__(self, input_dim1, input_dim2, **kwargs):
        if "num_classes" in kwargs:
            kwargs.pop("num_classes")
        if "h_dim" in kwargs:
            kwargs.pop("h_dim")
        super(UndirectedCosineLinkScorer, self).__init__(input_dim1, input_dim2, num_classes=1, h_dim=None, **kwargs)
        self.cos = nn.CosineSimilarity(dim=-1)
        assert input_dim1 == input_dim2

    def forward(self, x1, x2, **kwargs):
        return self.cos(x1, x2)


class DirectedCosineLinkScorer(AbstractCosineLinkScorer):
    def __init__(self, input_dim1, input_dim2, h_dim=200, **kwargs):
        if "num_classes" in kwargs:
            kwargs.pop("num_classes")
        super(DirectedCosineLinkScorer, self).__init__(input_dim1, input_dim2, num_classes=1, h_dim=h_dim, **kwargs)
        self.cos = nn.CosineSimilarity(dim=-1)
        assert input_dim1 == input_dim2
        self.l1_1 = nn.Linear(input_dim1, h_dim)
        self.l1_2 = nn.Linear(h_dim, h_dim)
        self.l2_1 = nn.Linear(input_dim2, h_dim)
        self.l2_2 = nn.Linear(h_dim, h_dim)

    def forward(self, x1, x2, **kwargs):
        x1 = self.l1_2(F.relu(self.l1_1(x1)))
        x2 = self.l2_2(F.relu(self.l2_1(x2)))
        return self.cos(x1, x2)


class UndirectedL2LinkScorer(AbstractL2LinkScorer):
    def __init__(self, input_dim1, input_dim2, **kwargs):
        if "num_classes" in kwargs:
            kwargs.pop("num_classes")
        if "h_dim" in kwargs:
            kwargs.pop("h_dim")
        super(UndirectedL2LinkScorer, self).__init__(input_dim1, input_dim2, num_classes=1, h_dim=None, **kwargs)
        assert input_dim1 == input_dim2

    def forward(self, x1, x2, **kwargs):
        return torch.norm(x1 - x2, dim=-1, keepdim=True)


class DirectedL2LinkScorer(AbstractL2LinkScorer):
    def __init__(self, input_dim1, input_dim2, h_dim=200, **kwargs):
        if "num_classes" in kwargs:
            kwargs.pop("num_classes")
        super(DirectedL2LinkScorer, self).__init__(input_dim1, input_dim2, num_classes=1, h_dim=h_dim, **kwargs)
        assert input_dim1 == input_dim2

        self.l1_1 = nn.Linear(input_dim1, h_dim)
        self.l1_2 = nn.Linear(h_dim, h_dim)
        self.l2_1 = nn.Linear(input_dim2, h_dim)
        self.l2_2 = nn.Linear(h_dim, h_dim)

    def forward(self, x1, x2, **kwargs):
        x1 = self.l1_2(F.relu(self.l1_1(x1)))
        x2 = self.l1_2(F.relu(self.l1_2(x2)))
        return torch.norm(x1 - x2, dim=-1, keepdim=True)


class TranslationLinkScorer(AbstractL2LinkScorer):
    def __init__(self, input_dim1, input_dim2, h_dim=200, **kwargs):
        if "num_classes" in kwargs:
            kwargs.pop("num_classes")
        super(TranslationLinkScorer, self).__init__(input_dim1, input_dim2, num_classes=1, h_dim=h_dim, **kwargs)

        self.rel_emb = nn.Parameter(torch.randn(1, self._h_dim) / self._h_dim)
        self.src_proj = nn.Linear(self._input_dim1, self._h_dim, bias=False)
        self.dst_proj = nn.Linear(self._input_dim1, self._h_dim, bias=False)

    def forward(self, x1, x2, **kwargs):
        x1 = self.src_proj(x1) + self.rel_emb
        x2 = self.dst_proj(x2)
        return torch.norm(x1 - x2, dim=-1, keepdim=True)


class TransRLinkScorer(AbstractL2LinkScorer):
    def __init__(self, input_dim1, input_dim2, num_classes, h_dim=200, **kwargs):
        super(TransRLinkScorer, self).__init__(input_dim1, input_dim2, num_classes, h_dim=h_dim, **kwargs)
        assert input_dim1 == input_dim2

        self.rel_emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=self._h_dim)
        self.proj_matr = nn.Embedding(num_embeddings=self._num_classes, embedding_dim=input_dim1 * self._h_dim)

    def forward(self, x1, x2, labels=None, **kwargs):
        weights = self.proj_matr(labels).reshape((-1, self.rel_dim, self.input_dim))
        rels = self.rel_emb(labels)
        m_s = (weights * x1.unsqueeze(1)).sum(-1)
        m_t = (weights * x2.unsqueeze(1)).sum(-1)

        transl = m_s + rels
        return torch.norm(transl - m_t, dim=-1, keepdim=True)



