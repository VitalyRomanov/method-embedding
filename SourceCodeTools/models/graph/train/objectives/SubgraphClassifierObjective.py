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

        # for id_ in self.ids:
        #     idx = self._get_subgraph(id_)
        #     loader = self.loading_fn(idx)
        #     for input_nodes, seeds, blocks in loader:
        #         yield input_nodes, torch.LongTensor([id_]), blocks

    def _get_subgraph(self, id_):
        return self.subgraph_mapping[id_]


class SubgraphAbstractObjective(AbstractObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphAbstractObjective, self).__init__(*args, **kwargs)
    #
    # def __init__(
    #         self, name, graph_model, node_embedder, dataset, label_load_fn, device,
    #         sampling_neighbourhood_size, batch_size, labels_for, number_of_hops, preload_for="package",
    #         masker_fn=None, label_loader_class=None, label_loader_params=None,
    #         tokenizer_path=None, target_emb_size=None, link_scorer_type="inner_prod", masker: SubwordMasker = None,
    #         measure_scores=False, dilate_scores=1, early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
    #         subgraph_mapping=None, subgraph_partition=None
    # ):
    #     assert subgraph_partition is not None, "Provide train/val/test splits with `subgraph_partition` option"
    #     self.subgraph_mapping = subgraph_mapping
    #     self.subgraph_partition = unpersist(subgraph_partition)
    #     super(SubgraphAbstractObjective, self).__init__(
    #         name, graph_model, node_embedder, dataset, label_load_fn, device,
    #         sampling_neighbourhood_size, batch_size, labels_for=labels_for, number_of_hops=number_of_hops,
    #         preload_for=preload_for, masker_fn=masker_fn, label_loader_class=label_loader_class,
    #         label_loader_params=label_loader_params,
    #         tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_scorer_type=link_scorer_type,
    #         masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores,
    #         early_stopping=False, early_stopping_tolerance=20, nn_index=nn_index
    #     )
    #
    #     self.target_embedding_fn = self.get_targets_from_embedder
    #     self.negative_factor = 1
        self.update_embeddings_for_queries = False
        self.create_pooling_layer()

    def create_pooling_layer(self):
        pass

    # def _create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
    #     self.target_embedder = SubgraphElementEmbedderWithSubwords(
    #         data_loading_func(), self.target_emb_size, tokenizer_path
    #     )

    # def _get_training_targets(self):
    #
    #     if hasattr(self.graph_model.g, 'ntypes'):
    #         self.ntypes = self.graph_model.g.ntypes
    #         self.use_types = True
    #
    #         if len(self.graph_model.g.ntypes) == 1:
    #             self.use_types = False
    #     else:
    #         # not sure when this is called
    #         raise NotImplementedError()
    #
    #     train_idx = self.subgraph_partition[
    #         self.subgraph_partition["train_mask"]
    #     ]["id"].to_numpy()
    #     val_idx = self.subgraph_partition[
    #         self.subgraph_partition["val_mask"]
    #     ]["id"].to_numpy()
    #     test_idx = self.subgraph_partition[
    #         self.subgraph_partition["test_mask"]
    #     ]["id"].to_numpy()
    #
    #     return train_idx, val_idx, test_idx

    # def _create_loaders(self):
    #     print("Number of nodes", self.graph_model.g.number_of_nodes())
    #     train_idx, val_idx, test_idx = self._get_training_targets()
    #     train_idx, val_idx, test_idx = self.target_embedder.create_idx_pools(
    #         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
    #     )
    #     logging.info(
    #         f"Pool sizes for {self.name}: train {self._idx_len(train_idx)}, "
    #         f"val {self._idx_len(val_idx)}, "
    #         f"test {self._idx_len(test_idx)}."
    #     )
    #     loaders = self._get_loaders(
    #         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    #         batch_size=self.batch_size  # batch_size_node_name
    #     )
    #     self.train_loader, self.val_loader, self.test_loader = loaders
    #
    #     # def get_num_nodes(ids):
    #     #     return sum(len(ids[key_]) for key_ in ids) // self.batch_size + 1
    #
    #     self.num_train_batches = len(train_idx) // self.batch_size + 1
    #     self.num_test_batches = len(test_idx) // self.batch_size + 1
    #     self.num_val_batches = len(val_idx) // self.batch_size + 1
    #
    # def _get_loaders(self, train_idx, val_idx, test_idx, batch_size):
    #
    #     logging.info("Batch size is ignored for subgraphs")
    #
    #     subgraph_mapping = self.subgraph_mapping
    #
    #     train_loader = SubgraphLoader(train_idx, subgraph_mapping, self._create_loader, batch_size, self.graph_model.g.ntypes)
    #     val_loader = SubgraphLoader(val_idx, subgraph_mapping, self._create_loader, batch_size, self.graph_model.g.ntypes)
    #     test_loader = SubgraphLoader(test_idx, subgraph_mapping, self._create_loader, batch_size, self.graph_model.g.ntypes)
    #
    #     return train_loader, val_loader, test_loader

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

    # def _graph_embeddings(self, input_nodes, graph, mask=None):
    #     emb = self._extract_embed(input_nodes, mask=mask)
    #     graph_emb = self.graph_model(self._wrap_into_dict(emb), blocks=None, graph=graph)
    #     return graph_emb["node_"]

    # def _evaluate_objective(self, data_split, neg_sampling_strategy=None, negative_factor=1):
    #     at = [1, 3, 5, 10]
    #     count = 0
    #
    #     scores = defaultdict(list)
    #
    #     for input_nodes, seeds, blocks in tqdm(
    #             getattr(self, f"{data_split}_loader"), total=getattr(self, f"num_{data_split}_batches")
    #     ):
    #         blocks = [blk.to(self.device) for blk in blocks]
    #
    #         subgraph_masks, seeds = seeds
    #
    #         if self.masker is None:
    #             masked = None
    #         else:
    #             masked = self.masker.get_mask(self.seeds_to_python(seeds))
    #
    #         src_embs = self._graph_embeddings(input_nodes, blocks, masked=masked, subgraph_masks=subgraph_masks)
    #         node_embs_, element_embs_, labels = self.prepare_for_prediction(
    #             src_embs, seeds, self.target_embedding_fn, negative_factor=negative_factor,
    #             neg_sampling_strategy=neg_sampling_strategy,
    #             train_embeddings=False
    #         )
    #
    #         if self.measure_scores:
    #             if count % self.dilate_scores == 0:
    #                 scores_ = self.target_embedder.score_candidates(self.seeds_to_global(seeds), src_embs,
    #                                                              self.link_predictor, at=at,
    #                                                              type=self.link_scorer_type, device=self.device)
    #                 for key, val in scores_.items():
    #                     scores[key].append(val)
    #
    #         acc, loss = self._compute_acc_loss(node_embs_, element_embs_, labels)
    #
    #         scores["Loss"].append(loss.item())
    #         scores["Accuracy"].append(acc)
    #         count += 1
    #
    #     scores = {key: self._sum_scores(val) for key, val in scores.items()}
    #     return scores

    def _verify_parameters(self):
        pass


class SubgraphEmbeddingObjective(SubgraphAbstractObjective):
    def __init__(self, *args, **kwargs):
        super(SubgraphEmbeddingObjective, self).__init__(*args, **kwargs)
    # def __init__(
    #         self, name, graph_model, node_embedder, dataset, label_load_fn, device,
    #         sampling_neighbourhood_size, batch_size, labels_for, number_of_hops, preload_for="package",
    #         masker_fn=None, label_loader_class=None, label_loader_params=None,
    #         tokenizer_path=None, target_emb_size=None, link_scorer_type="inner_prod", masker: SubwordMasker = None,
    #         measure_scores=False, dilate_scores=1, early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
    #         subgraph_mapping=None, subgraph_partition=None
    # ):
    #     super(SubgraphEmbeddingObjective, self).__init__(
    #         name, graph_model, node_embedder, dataset, label_load_fn, device,
    #         sampling_neighbourhood_size, batch_size, labels_for=labels_for, number_of_hops=number_of_hops,
    #         preload_for=preload_for, masker_fn=masker_fn, label_loader_class=label_loader_class,
    #         label_loader_params=label_loader_params,
    #         tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_scorer_type=link_scorer_type,
    #         masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores,
    #         early_stopping=False, early_stopping_tolerance=20, nn_index=nn_index
    #     )


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

    # def __init__(
    #         self, name, graph_model, node_embedder, dataset, label_load_fn, device,
    #         sampling_neighbourhood_size, batch_size, labels_for, number_of_hops, preload_for="package",
    #         masker_fn=None, label_loader_class=None, label_loader_params=None,
    #         tokenizer_path=None, target_emb_size=None, link_scorer_type="inner_prod", masker: SubwordMasker = None,
    #         measure_scores=False, dilate_scores=1, early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
    #         subgraph_mapping=None, subgraph_partition=None
    # ):
    #     SubgraphAbstractObjective.__init__(
    #         self, name, graph_model, node_embedder, dataset, label_load_fn, device,
    #         sampling_neighbourhood_size, batch_size, labels_for=labels_for, number_of_hops=number_of_hops,
    #         preload_for=preload_for, masker_fn=masker_fn, label_loader_class=label_loader_class,
    #         label_loader_params=label_loader_params,
    #         tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_scorer_type=link_scorer_type,
    #         masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores,
    #         early_stopping=False, early_stopping_tolerance=20, nn_index=nn_index
    #     )

    # def _create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
    #     self.target_embedder = SubgraphClassifierTargetMapper(
    #         elements=data_loading_func()
    #     )

    # def _evaluate_objective(self, data_split, neg_sampling_strategy=None, negative_factor=1):
    #     at = [1, 3, 5, 10]
    #     count = 0
    #     scores = defaultdict(list)
    #
    #     for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
    #         blocks = [blk.to(self.device) for blk in blocks]
    #
    #         subgraph_masks, seeds = seeds
    #
    #         if self.masker is None:
    #             masked = None
    #         else:
    #             masked = self.masker.get_mask(self.seeds_to_python(seeds))
    #
    #         src_embs = self._graph_embeddings(input_nodes, blocks, masked=masked, subgraph_masks=subgraph_masks)
    #         node_embs_, element_embs_, labels = self.prepare_for_prediction(
    #             src_embs, seeds, self.target_embedding_fn, negative_factor=negative_factor,
    #             neg_sampling_strategy=neg_sampling_strategy,
    #             train_embeddings=False
    #         )
    #         # indices = self.seeds_to_global(seeds).tolist()
    #         # labels = self.target_embedder[indices]
    #         # labels = torch.LongTensor(labels).to(self.device)
    #         acc, loss, logits = self._compute_acc_loss(node_embs_, element_embs_, labels)
    #
    #         y_pred = nn.functional.softmax(logits, dim=-1).to("cpu").numpy()
    #         y_true = np.zeros(y_pred.shape)
    #         y_true[np.arange(0, y_true.shape[0]), labels.to("cpu").numpy()] = 1.
    #
    #         if self.measure_scores:
    #             if count % self.dilate_scores == 0:
    #                 y_true_onehot = np.array(y_true)
    #                 labels = list(range(y_true_onehot.shape[1]))
    #
    #                 for k in at:
    #                     scores[f"ndcg@{k}"].append(ndcg_score(y_true, y_pred, k=k))
    #                     scores[f"acc@{k}"].append(
    #                         top_k_accuracy_score(y_true_onehot.argmax(-1), y_pred, k=k, labels=labels)
    #                     )
    #
    #         scores["Loss"].append(loss.item())
    #         scores["Accuracy"].append(acc)
    #         count += 1
    #
    #     if count == 0:
    #         count += 1
    #
    #     scores = {key: self._sum_scores(val) for key, val in scores.items()}
    #     return scores

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