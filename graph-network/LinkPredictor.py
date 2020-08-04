import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LinkPredictor(nn.Module):
    def __init__(self, input_dimensionality):
        super(LinkPredictor, self).__init__()

        self.l1 = nn.Linear(input_dimensionality, 20)
        self.l1_a = nn.Sigmoid()

        self.logits = nn.Linear(20, 2)

    def forward(self, x, **kwargs):
        x = F.relu(self.l1(x))
        return self.logits(x)