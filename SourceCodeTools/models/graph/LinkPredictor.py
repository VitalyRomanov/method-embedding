import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    def __init__(self, input_dimensionality):
        super(LinkPredictor, self).__init__()

        self.norm = nn.BatchNorm1d(input_dimensionality) # LayerNorm

        self.l1 = nn.Linear(input_dimensionality, 20)
        self.l1_a = nn.Sigmoid()

        self.logits = nn.Linear(20, 2)

    # def forward(self, x, **kwargs):
    #     x = self.norm(x)
    #     x = F.relu(self.l1(x))
    #     return self.logits(x)

    def forward(self, x1, x2, **kwargs):
        x = torch.cat([x1, x2], dim=1)
        x = self.norm(x)
        x = F.relu(self.l1(x))
        return self.logits(x)


class LinkClassifier(nn.Module):
    def __init__(self, input_dimensionality, num_classes):
        super(LinkClassifier, self).__init__()

        self.norm = nn.BatchNorm1d(input_dimensionality) # LayerNorm

        self.l1 = nn.Linear(input_dimensionality, 20)
        self.l1_a = nn.Sigmoid()

        self.logits = nn.Linear(20, num_classes)

    def forward(self, x1, x2, **kwargs):
        x = torch.cat([x1, x2], dim=1)
        x = self.norm(x)
        x = F.relu(self.l1(x))
        return self.logits(x)


class BilinearLinkPedictor(nn.Module):
    def __init__(self, embedding_dim_1, embedding_dim_2, target_classes=2):
        super(BilinearLinkPedictor, self).__init__()

        self.l1 = nn.Linear(embedding_dim_1, 300)
        self.l2 = nn.Linear(embedding_dim_2, 300)
        self.act = nn.Sigmoid()
        self.bilinear = nn.Bilinear(300, 300, target_classes)

    def forward(self, x1, x2):
        return self.bilinear(self.act(self.l1(x1)), self.act(self.l2(x2)))


class CosineLinkPredictor(nn.Module):
    def __init__(self, margin=0.):
        """
        Dummy link predictor, using to keep API the same
        """
        super(CosineLinkPredictor, self).__init__()

        self.cos = nn.CosineSimilarity()
        self.max_margin = torch.Tensor([margin])[0]

    def forward(self, x1, x2):
        if self.max_margin.device != x1.device:
            self.max_margin = self.max_margin.to(x1.device)
        # this will not train
        logit = (self.cos(x1, x2) > self.max_margin).float().unsqueeze(1)
        return torch.cat([1 - logit, logit], dim=1)


class L2LinkPredictor(nn.Module):
    def __init__(self, margin=1.):
        super(L2LinkPredictor, self).__init__()
        self.margin = torch.Tensor([margin])[0]

    def forward(self, x1, x2):
        if self.margin.device != x1.device:
            self.margin = self.margin.to(x1.device)
        # this will not train
        logit = (torch.norm(x1 - x2, dim=-1, keepdim=True) < self.margin).float()
        return torch.cat([1 - logit, logit], dim=1)


class TransRLinkPredictor(nn.Module):
    def __init__(self, input_dim, rel_dim, num_relations, margin=0.3):
        super(TransRLinkPredictor, self).__init__()
        self.rel_dim = rel_dim
        self.input_dim = input_dim
        self.margin = margin

        self.rel_emb = nn.Embedding(num_embeddings=num_relations, embedding_dim=rel_dim)
        self.proj_matr = nn.Embedding(num_embeddings=num_relations, embedding_dim=input_dim * rel_dim)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, a, p, n, labels):
        weights = self.proj_matr(labels).reshape((-1, self.rel_dim, self.input_dim))
        rels = self.rel_emb(labels)
        m_a = (weights * a.unsqueeze(1)).sum(-1)
        m_p = (weights * p.unsqueeze(1)).sum(-1)
        m_n = (weights * n.unsqueeze(1)).sum(-1)

        transl = m_a + rels

        sim = torch.norm(torch.cat([transl - m_p, transl - m_n], dim=0), dim=-1)

        return self.triplet_loss(transl, m_p, m_n), sim < self.margin



