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
    def __init__(self):
        super(CosineLinkPredictor, self).__init__()

        self.cos = nn.CosineSimilarity()
        self.max_margin = torch.Tensor([0.4])

    def forward(self, x1, x2):
        if self.max_margin.device != x1.device:
            self.max_margin = self.max_margin.to(x1.device)
        logit = (self.cos(x1, x2) > self.max_margin).float().unsqueeze(1)
        return torch.cat([1 - logit, logit], dim=1)