from collections import OrderedDict
import numpy as np
import random as rnd
import torch
from torch.nn import CrossEntropyLoss

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.mltools.torch import _compute_accuracy
from SourceCodeTools.models.graph.ElementEmbedder import GraphLinkSampler
from SourceCodeTools.models.graph.ElementEmbedderBase import ElementEmbedderBase
from SourceCodeTools.models.graph.LinkPredictor import LinkClassifier, BilinearLinkPedictor
from SourceCodeTools.models.graph.train.Scorer import Scorer
from SourceCodeTools.models.graph.train.objectives.GraphLinkObjective import GraphLinkObjective
# from SourceCodeTools.models.graph.train.sampling_multitask2 import _compute_accuracy
from SourceCodeTools.tabular.common import compact_property


class GraphLinkClassificationObjective(GraphLinkObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super().__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )
        self.measure_ndcg = False

    def create_graph_link_sampler(self, data_loading_func, nodes):
        self.target_embedder = TargetLinkMapper(
            elements=data_loading_func(), nodes=nodes
        )

    def create_link_predictor(self):
        self.link_predictor = BilinearLinkPedictor(self.target_emb_size, self.graph_model.emb_size, self.target_embedder.num_classes).to(self.device)
        # self.positive_label = 1
        self.negative_label = 0
        self.label_dtype = torch.long

    def _logits_nodes(self, node_embeddings,
                      elem_embedder, link_predictor, create_dataloader,
                      src_seeds, negative_factor=1, train_embeddings=True):
        k = negative_factor
        indices = self.seeds_to_global(src_seeds).tolist()
        batch_size = len(indices)

        node_embeddings_batch = node_embeddings
        dst_indices, labels_pos = elem_embedder[indices]  # this assumes indices is torch tensor

        # dst targets are not unique
        unique_dst, slice_map = self._handle_non_unique(dst_indices)
        assert unique_dst[slice_map].tolist() == dst_indices.tolist()

        dataloader = create_dataloader(unique_dst)
        input_nodes, dst_seeds, blocks = next(iter(dataloader))
        blocks = [blk.to(self.device) for blk in blocks]
        assert dst_seeds.shape == unique_dst.shape
        assert dst_seeds.tolist() == unique_dst.tolist()
        unique_dst_embeddings = self._logits_batch(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
        dst_embeddings = unique_dst_embeddings[slice_map.to(self.device)]

        # self.target_embedder.set_embed(unique_dst.detach().cpu().numpy(), unique_dst_embeddings.detach().cpu().numpy())

        node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
        # negative_indices = torch.tensor(elem_embedder.sample_negative(
        #     batch_size * k), dtype=torch.long)  # embeddings are sampled from 3/4 unigram distribution
        negative_indices = torch.tensor(elem_embedder.sample_negative(
            batch_size * k, ids=indices), dtype=torch.long)  # closest negative
        unique_negative, slice_map = self._handle_non_unique(negative_indices)
        assert unique_negative[slice_map].tolist() == negative_indices.tolist()

        dataloader = create_dataloader(unique_negative)
        input_nodes, dst_seeds, blocks = next(iter(dataloader))
        blocks = [blk.to(self.device) for blk in blocks]
        assert dst_seeds.shape == unique_negative.shape
        assert dst_seeds.tolist() == unique_negative.tolist()
        unique_negative_random = self._logits_batch(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
        negative_random = unique_negative_random[slice_map.to(self.device)]
        labels_neg = torch.full((batch_size * k,), self.negative_label, dtype=self.label_dtype)

        # self.target_embedder.set_embed(unique_negative.detach().cpu().numpy(), unique_negative_random.detach().cpu().numpy())

        nodes = torch.cat([node_embeddings_batch, node_embeddings_neg_batch], dim=0)
        embs = torch.cat([dst_embeddings, negative_random], dim=0)
        labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)
        return nodes, embs, labels

    def compute_acc_loss(self, node_embs_, element_embs_, labels):
        logits = self.link_predictor(node_embs_, element_embs_)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1))

        acc = _compute_accuracy(logits.argmax(dim=1), labels)

        return acc, loss


class TargetLinkMapper(GraphLinkSampler):
    def __init__(self, elements, nodes):
        super(TargetLinkMapper, self).__init__(elements, nodes, compact_dst=False, dst_to_global=True, emb_size=1)

    def init(self, compact_dst):
        if compact_dst:
            elem2id, self.inverse_dst_map = compact_property(self.elements['dst'], return_order=True)
            self.elements['emb_id'] = self.elements['dst'].apply(lambda x: elem2id[x])
        else:
            self.elements['emb_id'] = self.elements['dst']

        self.link_type2id, self.inverse_link_type_map = compact_property(self.elements['type'], return_order=True, index_from_one=True)
        self.elements["link_type"] = list(map(lambda x: self.link_type2id[x], self.elements["type"].tolist()))

        self.element_lookup = {}
        for id_, emb_id, link_type in self.elements[["id", "emb_id", "link_type"]].values:
            if id_ in self.element_lookup:
                self.element_lookup[id_].append((emb_id, link_type))
            else:
                self.element_lookup[id_] = [(emb_id, link_type)]

        self.init_neg_sample()
        self.num_classes = len(self.inverse_link_type_map)

    def __getitem__(self, ids):
        node_ids, labels = zip(*(rnd.choice(self.element_lookup[id]) for id in ids))
        return np.array(node_ids, dtype=np.int32), torch.LongTensor(np.array(labels, dtype=np.int32))

    def sample_negative(self, size, ids=None, strategy="w2v"):  # TODO switch to w2v?
        if strategy == "w2v":
            negative = ElementEmbedderBase.sample_negative(self, size)
        else:
            negative = Scorer.sample_closest_negative(self, ids, k=size // len(ids))
            assert len(negative) == size
        return negative
