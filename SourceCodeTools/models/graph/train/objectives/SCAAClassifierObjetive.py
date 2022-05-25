from collections import OrderedDict
from itertools import chain

import torch
import torch.nn.functional as F

from SourceCodeTools.models.graph.train.objectives.SubgraphEmbedderObjective import SubgraphMatchingObjective


class PoolingLayer(torch.nn.Module):

    def __init__(self, k, shape=1):
        super().__init__()
        self.learnable_vector = torch.nn.Parameter(torch.randn(shape))
        self.learnable_vector.requires_grad = True
        self.k = k

    def forward(self, x):
        length = torch.norm(self.learnable_vector)
        y = torch.mm(x, self.learnable_vector)/length
        if (y.shape[0] < self.k):
            y = F.pad(input=y, pad=(
                0, 0, 0, self.k - y.shape[0]), mode='constant', value=0)
            x = F.pad(input=x, pad=(
                0, 0, 0, self.k - x.shape[0]), mode='constant', value=0)
        idx = torch.topk(y, self.k, dim=0)
        y = torch.gather(y, 0, idx.indices)
        y = torch.sigmoid(y)
        x_part = torch.gather(x, 0, idx.indices)
        return torch.mul(x_part, y)


class SCAAClassifierObjective(SubgraphMatchingObjective):

    def __init__(
        self, name, graph_model, node_embedder, nodes, data_loading_func, device,
        sampling_neighbourhood_size, batch_size,
        tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod",
        masker=None, measure_scores=False, dilate_scores=1,
        early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
        ns_groups=None, subgraph_mapping=None, subgraph_partition=None
    ):
        SubgraphMatchingObjective.__init__(self,
                                           name, graph_model, node_embedder, nodes, data_loading_func, device,
                                           sampling_neighbourhood_size, batch_size,
                                           tokenizer_path, target_emb_size, link_predictor_type,
                                           masker, measure_scores, dilate_scores, early_stopping, early_stopping_tolerance, nn_index,
                                           ns_groups, subgraph_mapping, subgraph_partition
                                           )
        self.pooler = PoolingLayer(100, (100, 1))
        # self.pooler = torch.nn.MultiheadAttention(
        #     100, 2, device=device, batch_first=True)

    def parameters(self, recurse: bool = True):
        return chain(self.link_predictor.parameters(), self.pooler.parameters())

    def pooling_fn(self, node_embeddings):
        return self.pooler(node_embeddings)

        # node_embeddings = torch.reshape(node_embeddings, (1, -1, 100))
        # pool = self.pooler(node_embeddings, node_embeddings,
        #                    node_embeddings, need_weights=False)[0][0]
        # return torch.transpose(torch.mean(pool, dim=0, keepdim=True), 0, 1)

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.link_predictor.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        for k, v in self.pooler.state_dict().items():
            state_dict[f"pooler.{k}"] = v
        return state_dict

    def custom_load_state_dict(self, state_dicts):
        self.link_predictor.load_state_dict(
            self.get_prefix("classifier", state_dicts)
        )
        self.pooler.load_state_dict(
            self.get_prefix("pooler", state_dicts)
        )
