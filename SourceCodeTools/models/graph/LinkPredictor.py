import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractLinkClassifier(nn.Module):
    def __init__(self, input_dim1, input_dim2, num_classes, **kwargs):
        super(AbstractLinkClassifier, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.num_classes = num_classes

    def forward(self, x1, x2, **kwargs):
        # Similarity score depends on the type of the metric used
        #
        raise NotImplementedError()


# class LinkPredictor(AbstractLinkClassifier):
#     def __init__(self, input_dim1, input_dim2, num_classes, **kwargs):
#         super(LinkPredictor, self).__init__(input_dim1, input_dim2, num_classes)
#         concat_emb_size = self.input_dim1 + self.input_dim1
#         self.norm = nn.BatchNorm1d(concat_emb_size) # LayerNorm
#         self.l1 = nn.Linear(concat_emb_size, 20)
#         self.logits = nn.Linear(20, 2)
#
#     # def forward(self, x, **kwargs):
#     #     x = self.norm(x)
#     #     x = F.relu(self.l1(x))
#     #     return self.logits(x)
#
#     def forward(self, x1, x2, **kwargs):
#         x = torch.cat([x1, x2], dim=-1)
#         x = self.norm(x)
#         x = F.relu(self.l1(x))
#         return self.logits(x)


class LinkClassifier(AbstractLinkClassifier):
    def __init__(self, input_dim1, input_dim2, num_classes=2, **kwargs):
        super(LinkClassifier, self).__init__(input_dim1, input_dim2, num_classes)
        concat_emb_size = self.input_dim1 + self.input_dim1

        self.norm = nn.BatchNorm1d(concat_emb_size) # LayerNorm
        self.l1 = nn.Linear(concat_emb_size, 20)
        self.logits = nn.Linear(20, num_classes)

    def forward(self, x1, x2, **kwargs):
        x = torch.cat([x1, x2], dim=-1)
        x = self.norm(x)
        x = F.relu(self.l1(x))
        return self.logits(x)


class BilinearLinkClassifier(AbstractLinkClassifier):
    def __init__(self, input_dim1, input_dim2, num_classes=2, **kwargs):
        super(BilinearLinkClassifier, self).__init__(input_dim1, input_dim2, num_classes)

        # self.src_l1 = nn.Linear(embedding_dim_1, embedding_dim_1)
        # self.src_l2 = nn.Linear(embedding_dim_1, embedding_dim_1)
        # self.src_l3 = nn.Linear(embedding_dim_1, embedding_dim_1)
        #
        # self.dst_l1 = nn.Linear(embedding_dim_2, embedding_dim_2)
        # self.dst_l2 = nn.Linear(embedding_dim_2, embedding_dim_2)
        # self.dst_l3 = nn.Linear(embedding_dim_2, embedding_dim_2)

        self.l1 = nn.Linear(self.input_dim1, 300, bias=False)
        self.l2 = nn.Linear(self.input_dim2, 300, bias=False)
        self.act = nn.Sigmoid()
        self.bilinear = nn.Bilinear(300, 300, self.num_classes)

    def forward(self, x1, x2, **kwargs):

        x1_l1 = self.act(self.l1(x1))
        x2_l2 = self.act(self.l2(x2))

        return self.bilinear(x1_l1, x2_l2)


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



