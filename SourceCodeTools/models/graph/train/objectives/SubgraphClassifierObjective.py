import logging
from collections import OrderedDict, defaultdict
from itertools import chain
from typing import Optional

import dgl
import numpy as np
import torch
from sklearn.metrics import ndcg_score, top_k_accuracy_score
from torch import nn
from tqdm import tqdm

from SourceCodeTools.code.data.dataset.SubwordMasker import SubwordMasker
from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedderWithBpeSubwords
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.train.Scorer import Scorer
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective
from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import NodeClassifier, \
    NodeClassifierObjective
from SourceCodeTools.tabular.common import compact_property


class PoolingLayerUnet(torch.nn.Module):
    # From : http://proceedings.mlr.press/v97/gao19a/gao19a.pdf

    def __init__(self, k, shape=1):
        super().__init__()
        self.learnable_vector = torch.nn.Parameter(torch.randn(shape))
        self.learnable_vector.requires_grad = True
        self.k = k

    def forward(self, x):
        length = torch.norm(self.learnable_vector)
        y = (x @ self.learnable_vector / length).reshape(-1)
        top_k = min(self.k, y.shape[0])
        y, idx = torch.topk(y, top_k, dim=0)
        y = torch.sigmoid(y)
        x_part = x[idx]
        return (x_part * y.reshape(-1, 1)).sum(dim=0, keepdim=True)


class PoolingLayerWithTrans(torch.nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(emb_size, 10)
        self.multihead_attn2 = nn.MultiheadAttention(emb_size, 10)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.multihead_attn(x,x,x)[0])
        x = torch.relu(self.multihead_attn2(x,x,x)[0])
        x = x.squeeze(1)
        assert len(x.shape) == 2
        return torch.max(x,dim=0,keepdim=True)[0]

class SubgraphLoader:
    def __init__(self, ids, subgraph_mapping, loading_fn, batch_size, graph_node_types):
        self.ids = ids
        self.loading_fn = loading_fn
        self.subgraph_mapping = subgraph_mapping
        self.iterator = None
        self.batch_size = batch_size
        self.graph_node_types = graph_node_types

    def __iter__(self):
        # TODO
        # supports only nodes without types

        for i in range(0, len(self.ids), self.batch_size):

            node_ids = dict()
            subgraphs = dict()

            batch_ids = self.ids[i: i + self.batch_size]
            for id_ in batch_ids:
                subgraph_nodes = self._get_subgraph(id_)
                subgraphs[id_] = subgraph_nodes

                for type_ in subgraph_nodes:
                    if type_ not in node_ids:
                        node_ids[type_] = set()
                    node_ids[type_].update(subgraph_nodes[type_])

            for type_ in node_ids:
                node_ids[type_] = sorted(list(node_ids[type_]))

            coincidence_matrix = []
            for id_, subgraph in subgraphs.items():
                coincidence_matrix.append([])
                for type_ in self.graph_node_types:
                    subgraph_nodes = subgraph[type_]
                    for node_id in node_ids[type_]:
                        coincidence_matrix[-1].append(node_id in subgraph_nodes)

            coincidence_matrix = torch.BoolTensor(coincidence_matrix)

            loader = self.loading_fn(node_ids)

            for input_nodes, seeds, blocks in loader:
                yield input_nodes, (coincidence_matrix, torch.LongTensor(batch_ids)), blocks

    def _get_subgraph(self, id_):
        return self.subgraph_mapping[id_]


class SubgraphAbstractObjective(AbstractObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphAbstractObjective, self).__init__(*args, **kwargs)
        self.update_embeddings_for_queries = False
        self.create_pooling_layer()

    def create_pooling_layer(self):
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

    def pooling_fn(self, node_embeddings):
        return torch.mean(node_embeddings, dim=0, keepdim=True)

    def _verify_parameters(self):
        pass


class SubgraphEmbeddingObjective(SubgraphAbstractObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphEmbeddingObjective, self).__init__(*args, **kwargs)


class SubgraphClassifierObjective(NodeClassifierObjective, SubgraphAbstractObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphClassifierObjective, self).__init__(*args, **kwargs)

    def _graph_embeddings(self, input_nodes, blocks, mask=None):
        graph = blocks

        emb = self._extract_embed(input_nodes, mask=mask)
        node_embs = self.graph_model(self._wrap_into_dict(emb), blocks=None, graph=graph)

        graph.nodes["node_"].data["node_embeddings"] = node_embs["node_"]

        unbatched = dgl.unbatch(graph)

        subgraph_embs = []
        for subgraph in unbatched:
            subgraph_embs.append(self.pooling_fn(subgraph.nodes["node_"].data["node_embeddings"]))

        return torch.cat(subgraph_embs, dim=0)

    def parameters(self, recurse: bool = True):
        return chain(self.classifier.parameters())

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

class SubgraphClassifierObjectiveWithUnetPool(SubgraphClassifierObjective):
    def __init__(self,*args, **kwargs):
        super(SubgraphClassifierObjectiveWithUnetPool,self).__init__(*args, **kwargs)

    def create_pooling_layer(self):
        self.pooler = PoolingLayerUnet(300, (300, 1)).to(self.device)

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


class SubgraphElementEmbedderBase(ElementEmbedderBase):
    def __init__(self, elements, compact_dst=True):
        # super(ElementEmbedderBase, self).__init__()
        self.elements = elements.rename({"src": "id"}, axis=1)
        self.init(compact_dst)

    def preprocess_element_data(self, *args, **kwargs):
        pass

    def create_idx_pools(self, train_idx, val_idx, test_idx):
        pool = set(self.elements["id"])
        train_pool, val_pool, test_pool = self._create_pools(train_idx, val_idx, test_idx, pool)
        return train_pool, val_pool, test_pool


class SubgraphElementEmbedderWithSubwords(SubgraphElementEmbedderBase, ElementEmbedderWithBpeSubwords):
    def __init__(self, elements, emb_size, tokenizer_path, num_buckets=100000, max_len=10):
        self.tokenizer_path = tokenizer_path
        SubgraphElementEmbedderBase.__init__(self, elements=elements, compact_dst=False)
        nn.Module.__init__(self)
        Scorer.__init__(self, num_embs=len(self.elements["dst"].unique()), emb_size=emb_size,
                        src2dst=self.element_lookup)

        self.emb_size = emb_size
        self.init_subwords(elements, num_buckets=num_buckets, max_len=max_len)


class SubgraphClassifierTargetMapper(SubgraphElementEmbedderBase, Scorer):
    def __init__(self, elements):
        SubgraphElementEmbedderBase.__init__(self, elements=elements)
        self.num_classes = len(self.inverse_dst_map)

    def set_embed(self, *args, **kwargs):
        pass

    def prepare_index(self, *args):
        pass