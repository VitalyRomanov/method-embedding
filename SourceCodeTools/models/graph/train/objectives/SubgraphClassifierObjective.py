from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import chain
from typing import Dict

import dgl
import torch
from torch import nn
from torch.utils import checkpoint

from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective, GNNOutput
from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import NodeClassifierObjective


class PoolingLayerUnet(torch.nn.Module):
    # From : http://proceedings.mlr.press/v97/gao19a/gao19a.pdf

    def __init__(self, k, emb_dim):
        super().__init__()
        self.learnable_vector = torch.nn.Parameter(torch.randn(emb_dim, 1))
        self.learnable_vector.requires_grad = True
        self.k = k
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def do_stuff(self, x):
        length = torch.norm(self.learnable_vector)
        y = (x @ self.learnable_vector / length).reshape(-1)
        top_k = min(self.k, y.shape[0])
        y, idx = torch.topk(y, top_k, dim=0)
        y = torch.sigmoid(y)
        x_part = x[idx]
        return (x_part * y.reshape(-1, 1)).sum(dim=0, keepdim=True)

    def custom(self):
        def custom_forward(*inputs):
            x, dummy = inputs
            return self.do_stuff(x)
        return custom_forward

    def forward(self, x):
        # if self.use_checkpoint:
        return checkpoint.checkpoint(self.custom(), x, self.dummy_tensor)
        # else:
        #     return self.do_stuff(x)


class PoolingLayerWithTrans(torch.nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(emb_size, 10)
        self.multihead_attn2 = nn.MultiheadAttention(emb_size, 10)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def do_stuff(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.multihead_attn(x, x, x)[0])
        x = torch.relu(self.multihead_attn2(x, x, x)[0])
        x = x.squeeze(1)
        assert len(x.shape) == 2
        return torch.mean(x, dim=0, keepdim=True)

    def custom(self):
        def custom_forward(*inputs):
            x, dummy = inputs
            return self.do_stuff(x)
        return custom_forward

    def forward(self, x):
        # if self.use_checkpoint:
        return checkpoint.checkpoint(self.custom(), x, self.dummy_tensor)
        # else:
        #     return self.do_stuff(x)


class SubgraphAbstractObjective(AbstractObjective, ABC):
    def __init__(self, *args, **kwargs):
        super(SubgraphAbstractObjective, self).__init__(*args, **kwargs)
        self.update_embeddings_for_queries = False
        self.create_pooling_layer()

    @abstractmethod
    def create_pooling_layer(self):
        pass

    @abstractmethod
    def pooling_fn(self, node_embeddings):
        pass

    def parameters(self, recurse: bool = True):
        return chain(self.target_embedder.parameters(), self.link_predictor.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.target_embedder.state_dict().items():
            state_dict[f"target_embedder.{k}"] = v
        for k, v in self.link_predictor.state_dict().items():
            state_dict[f"link_predictor.{k}"] = v
        return state_dict

    def custom_load_state_dict(self, state_dicts):
        self.target_embedder.load_state_dict(
            self.get_prefix("target_embedder", state_dicts)
        )
        self.link_predictor.load_state_dict(
            self.get_prefix("link_predictor", state_dicts)
        )

    def _verify_parameters(self):
        pass


class SubgraphEmbeddingObjective(SubgraphAbstractObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphEmbeddingObjective, self).__init__(*args, **kwargs)

    def create_pooling_layer(self):
        pass

    def pooling_fn(self, node_embeddings):
        return torch.mean(node_embeddings, dim=0, keepdim=True)


class SubgraphClassifierObjective(NodeClassifierObjective, SubgraphAbstractObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphClassifierObjective, self).__init__(*args, **kwargs)

    def create_pooling_layer(self):
        pass

    def pooling_fn(self, node_embeddings):
        return torch.mean(node_embeddings, dim=0, keepdim=True)

    def _graph_embeddings(self, input_nodes, blocks, mask=None, target_mask=None) -> GNNOutput:
        graph = blocks

        emb = self._wrap_into_dict(self._extract_embed(input_nodes, mask=mask))
        node_embs = self.graph_model(emb, blocks=None, graph=graph)

        graph.nodes["node_"].data["node_embeddings"] = node_embs["node_"]

        unbatched = dgl.unbatch(graph)

        subgraph_embs = []
        for subgraph in unbatched:
            subgraph_embs.append(self.pooling_fn(subgraph.nodes["node_"].data["node_embeddings"]))

        subgraph_embs_ = torch.cat(subgraph_embs, dim=0)
        if target_mask is not None:
            subgraph_embs_ = subgraph_embs_[target_mask]

        return GNNOutput(
            output=subgraph_embs_,
            node_embeddings=node_embs,
            input_embeddings=emb
        )

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters())

    def custom_load_state_dict(self, state_dicts):
        self.classifier.load_state_dict(
            self.get_prefix("classifier", state_dicts)
        )

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.classifier.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        return state_dict


class SubgraphClassifierObjectiveWithAttentionPooling(SubgraphClassifierObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphClassifierObjectiveWithAttentionPooling, self).__init__(*args, **kwargs)

    def create_pooling_layer(self):
        self.pooler = PoolingLayerWithTrans(self.target_emb_size).to(self.device)

    def pooling_fn(self, node_embeddings):
        return self.pooler(node_embeddings)

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters(), self.pooler.parameters())

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


class SubgraphClassifierObjectiveWithMaxPooling(SubgraphClassifierObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphClassifierObjectiveWithMaxPooling, self).__init__(*args, **kwargs)

    def pooling_fn(self, node_embeddings):
        return torch.max(node_embeddings, dim=0, keepdim=True)[0]

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.classifier.state_dict().items():
            state_dict[f"classifier.{k}"] = v
        return state_dict

    def custom_load_state_dict(self, state_dicts):
        self.classifier.load_state_dict(
            self.get_prefix("classifier", state_dicts)
        )


class SubgraphClassifierObjectiveWithUnetPool(SubgraphClassifierObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphClassifierObjectiveWithUnetPool, self).__init__(*args, **kwargs)

    def create_pooling_layer(self, num_heads=1):
        self.pooler = PoolingLayerUnet(k=300, emb_dim=self.graph_model.emb_size).to(self.device)

    def pooling_fn(self, node_embeddings):
        return self.pooler(node_embeddings)

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters(), self.pooler.parameters())

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
