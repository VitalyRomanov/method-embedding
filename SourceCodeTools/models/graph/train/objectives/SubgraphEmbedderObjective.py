from collections import OrderedDict
from itertools import chain
from typing import Optional

import numpy as np
from torch import nn

from SourceCodeTools.code.data.dataset.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedderWithBpeSubwords
from SourceCodeTools.models.graph.train.Scorer import Scorer
from SourceCodeTools.models.graph.train.objectives.SubgraphClassifierObjective import SubgraphAbstractObjective, \
    SubgraphElementEmbedderBase
import random as rnd


class SubgraphEmbeddingObjective(SubgraphAbstractObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod",
            masker: Optional[SubwordMasker] = None, measure_scores=False, dilate_scores=1,
            early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
            ns_groups=None, subgraph_mapping=None, subgraph_partition=None
    ):
        super(SubgraphEmbeddingObjective, self).__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path, target_emb_size, link_predictor_type,
            masker, measure_scores, dilate_scores, early_stopping, early_stopping_tolerance, nn_index,
            ns_groups, subgraph_mapping, subgraph_partition
        )

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = SubgraphElementEmbedderWithSubwords(
            data_loading_func(), self.target_emb_size, tokenizer_path
        )


class SubgraphMatchingObjective(SubgraphAbstractObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod",
            masker: Optional[SubwordMasker] = None, measure_scores=False, dilate_scores=1,
            early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
            ns_groups=None, subgraph_mapping=None, subgraph_partition=None
    ):
        super(SubgraphMatchingObjective, self).__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path, target_emb_size, link_predictor_type,
            masker, measure_scores, dilate_scores, early_stopping, early_stopping_tolerance, nn_index,
            ns_groups, subgraph_mapping, subgraph_partition
        )

        self.target_embedding_fn = self.get_targets_from_subgraph
        self.update_embeddings_for_queries = True

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = SubgraphElementEmbedderMatcher(
            data_loading_func(), self.target_emb_size, tokenizer_path
        )

    def get_targets_from_subgraph(
            self, positive_indices, negative_indices=None, train_embeddings=True
    ):
        def get_embeddings(loader):
            input_nodes, seeds, blocks = loader  # next(loader)
            subgraph_masks, seeds = seeds
            masked = self.masker.get_mask(self.seeds_to_python(seeds)) if self.masker is not None else None
            graph_emb = self._graph_embeddings(input_nodes, blocks, train_embeddings, masked=masked,
                                               subgraph_masks=subgraph_masks)
            return graph_emb

        def compactify(indices):
            return list(set(indices))

        def restore(indices, compact, embs):
            mapping = {i: compact.index(i) for i in indices}
            fancy = [mapping[i] for i in indices]
            return embs[fancy]

        def prepare(indices):
            indices_c = compactify(indices)
            embs = get_embeddings(self.train_loader.load_ids(indices_c))
            if len(indices_c) != len(indices):
                embs = restore(indices, indices_c, embs)
            return embs

        positive_dst = prepare(positive_indices)
        negative_dst = prepare(negative_indices) if negative_indices is not None else None

        return positive_dst, negative_dst

    def sample_negative(self, ids, k, neg_sampling_strategy):
        negative = self.target_embedder.sample_negative(
            k, ids=ids,
        )
        return negative

    def parameters(self, recurse: bool = True):
        return chain(self.link_predictor.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.link_predictor.state_dict().items():
            state_dict[f"link_predictor.{k}"] = v
        return state_dict


class SubgraphElementEmbedderWithSubwords(SubgraphElementEmbedderBase, ElementEmbedderWithBpeSubwords):
    def __init__(self, elements, emb_size, tokenizer_path, num_buckets=100000, max_len=10):
        self.tokenizer_path = tokenizer_path
        SubgraphElementEmbedderBase.__init__(self, elements=elements, compact_dst=False)
        nn.Module.__init__(self)
        Scorer.__init__(self, num_embs=len(self.elements["dst"].unique()), emb_size=emb_size,
                        src2dst=self.element_lookup)

        self.emb_size = emb_size
        self.init_subwords(elements, num_buckets=num_buckets, max_len=max_len)


class SubgraphElementEmbedderMatcher(SubgraphElementEmbedderBase, Scorer):
    def __init__(self, elements, emb_size, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        SubgraphElementEmbedderBase.__init__(self, elements=elements, compact_dst=False)
        nn.Module.__init__(self)
        Scorer.__init__(self, num_embs=len(self.elements["id"].unique()), emb_size=emb_size,
                        src2dst=self.element_lookup)

    def init(self, compact_dst):
        self.id2name = dict(zip(self.elements["id"], self.elements["dst"]))
        self.all_names = set(self.id2name.values())
        assert len(self.id2name) == len(self.elements)

        self.name2id = dict()
        for id_, name in self.elements.values:
            if name not in self.name2id:
                self.name2id[name] = []
            self.name2id[name].append(id_)

        self.element_lookup = dict()
        for id_ in self.id2name:
            self.element_lookup[id_] = list(set(self.name2id[self.id2name[id_]]) - {id_})

        self.elements["emb_id"] = self.elements["id"]

    def __getitem__(self, ids):
        return np.fromiter((rnd.choice(list(set(self.name2id[self.id2name[id]]) - {id})) for id in ids), dtype=np.int32)

    def sample_negative(self, size, ids=None):
        negative = []
        for id in ids:
            rnd_name = rnd.choice(list(self.all_names - {self.id2name[id]}))
            rnd_id = rnd.choice(list(self.name2id[rnd_name]))
            negative.append(rnd_id)
        return np.array(negative, dtype=np.int32)
