import torch
import torch.nn as nn
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
        x = torch.cat([x1, x2], dim=-1)
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
        x = torch.cat([x1, x2], dim=-1)
        x = self.norm(x)
        x = F.relu(self.l1(x))
        return self.logits(x)


class BilinearLinkPedictor(nn.Module):
    def __init__(self, embedding_dim_1, embedding_dim_2, target_classes=2):
        super(BilinearLinkPedictor, self).__init__()
        self.num_target_classes = target_classes

        # self.src_l1 = nn.Linear(embedding_dim_1, embedding_dim_1)
        # self.src_l2 = nn.Linear(embedding_dim_1, embedding_dim_1)
        # self.src_l3 = nn.Linear(embedding_dim_1, embedding_dim_1)
        #
        # self.dst_l1 = nn.Linear(embedding_dim_2, embedding_dim_2)
        # self.dst_l2 = nn.Linear(embedding_dim_2, embedding_dim_2)
        # self.dst_l3 = nn.Linear(embedding_dim_2, embedding_dim_2)

        self.l1 = nn.Linear(embedding_dim_1, 300, bias=False)
        self.l2 = nn.Linear(embedding_dim_2, 300, bias=False)
        self.act = nn.Sigmoid()
        self.bilinear = nn.Bilinear(300, 300, target_classes)

    def forward(self, x1, x2):

        # x1 = self.act(self.src_l1(x1))
        # x1 = self.act(self.src_l2(x1))
        # x1 = self.act(self.src_l3(x1))
        #
        # x2 = self.act(self.dst_l1(x2))
        # x2 = self.act(self.dst_l2(x2))
        # x2 = self.act(self.dst_l3(x2))

        if len(x1.shape) == 3 and x1.size(1) != x2.size(1):
            x1 = torch.tile(x1, (1, x2.size(1), 1))

        return self.bilinear(self.act(self.l1(x1)), self.act(self.l2(x2))).reshape(-1, self.num_target_classes)


class CosineUndirectedLinkPredictor(nn.Module):
    def __init__(self, margin=0.):
        """
        Dummy link predictor, using to keep API the same
        """
        super(CosineUndirectedLinkPredictor, self).__init__()

        self.cos = nn.CosineSimilarity(dim=-1)
        self.max_margin = torch.Tensor([margin])[0]

    def forward(self, x1, x2):
        if self.max_margin.device != x1.device:
            self.max_margin = self.max_margin.to(x1.device)
        # this will not train
        logit = (self.cos(x1, x2) > self.max_margin).float().unsqueeze(-1)
        return torch.cat([1 - logit, logit], dim=-1)


class L2UndirectedLinkPredictor(nn.Module):
    def __init__(self, margin=1.):
        super(L2UndirectedLinkPredictor, self).__init__()
        self.margin = torch.Tensor([margin])[0]

    def forward(self, x1, x2):
        if self.margin.device != x1.device:
            self.margin = self.margin.to(x1.device)
        # this will not train
        logit = (torch.norm(x1 - x2, dim=-1, keepdim=True) < self.margin).float()
        return torch.cat([1 - logit, logit], dim=-1)


class TranslationLinkPredictor(nn.Module):
    def __init__(self, input_dim, rel_dim, margin=0.3):
        super(TranslationLinkPredictor, self).__init__()
        self.rel_dim = rel_dim
        self.input_dim = input_dim
        self.margin = margin

        self.rel_emb = nn.Parameter(torch.randn(1, rel_dim) / rel_dim)
        self.src_proj = nn.Linear(input_dim, rel_dim, bias=False)
        self.dst_proj = nn.Linear(input_dim, rel_dim, bias=False)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

    def forward(self, a, p, n, labels):

        transl = self.src_proj(a) + self.rel_emb
        m_p = self.dst_proj(p)
        m_n = self.dst_proj(n)

        sim = torch.norm(torch.cat([transl - m_p, transl - m_n], dim=0), dim=-1)

        return self.triplet_loss(transl, m_p, m_n), sim < self.margin


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

        # pos_diff = torch.norm(transl - m_p, dim=-1)
        # neg_diff = torch.norm(transl - m_n, dim=-1)

        # loss = pos_diff + torch.maximum(torch.tensor([0.]).to(neg_diff.device), self.margin - neg_diff)
        # return loss.mean(), sim
        return self.triplet_loss(transl, m_p, m_n), sim < self.margin



