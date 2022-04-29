from collections import OrderedDict
from itertools import chain

import torch

from SourceCodeTools.models.graph.train.objectives.SubgraphClassifierObjective import SubgraphClassifierObjective


class PoolingLayer(torch.nn.Module):

    def __init__(self, k, shape=1):
        super().__init__()
        self.learnable_vector = torch.nn.Parameter(torch.randn(shape))
        self.learnable_vector.requires_grad = True
        self.k = k

    def forward(self, x):
        print('in', x.shape, end='\t')
        length = torch.norm(self.learnable_vector)
        y = torch.mm(x, self.learnable_vector)/length
        idx = torch.topk(y, self.k)
        y = torch.gather(y, -1, idx.indices)
        y = torch.sigmoid(y)
        x_part = torch.gather(x, -1, idx.indices)
        print('out', torch.mul(x_part, y).shape)
        return torch.mul(x_part, y)


class SCAAClassifierObjective(SubgraphClassifierObjective):

    def __init__(
        self, name, graph_model, node_embedder, nodes, data_loading_func, device,
        sampling_neighbourhood_size, batch_size,
        tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod",
        masker=None, measure_scores=False, dilate_scores=1,
        early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
        ns_groups=None, subgraph_mapping=None, subgraph_partition=None
    ):
        SubgraphClassifierObjective.__init__(self,
                                             name, graph_model, node_embedder, nodes, data_loading_func, device,
                                             sampling_neighbourhood_size, batch_size,
                                             tokenizer_path, target_emb_size, link_predictor_type,
                                             masker, measure_scores, dilate_scores, early_stopping, early_stopping_tolerance, nn_index,
                                             ns_groups, subgraph_mapping, subgraph_partition
                                             )
        self.pooler = PoolingLayer(50, (100, 100))

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters(), self.pooler.parameters())

    def pooling_fn(self, node_embeddings):
        return self.pooler(node_embeddings)

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.classifier.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        for k, v in self.pooler.state_dict().items():
            state_dict[f"pooler.{k}"] = v
        return state_dict

    def custom_load_state_dict(self, state_dicts):
        self.classifier.load_state_dict(
            self.get_prefix("classifier", state_dicts)
        )
        self.pooler.load_state_dict(
            self.get_prefix("pooler", state_dicts)
        )
