from collections import OrderedDict

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective


class GraphLinkObjective(AbstractObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(GraphLinkObjective, self).__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )

    def verify_parameters(self):
        pass

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.create_graph_link_sampler(data_loading_func, nodes)

    def create_link_predictor(self):
        if self.link_predictor_type == "nn":
            self.create_nn_link_predictor()
        elif self.link_predictor_type == "inner_prod":
            self.create_inner_prod_link_predictor()
        else:
            raise NotImplementedError()

    def forward(self, input_nodes, seeds, blocks, train_embeddings=True, neg_sampling_strategy=None):
        masked = None
        graph_emb = self._graph_embeddings(input_nodes, blocks, train_embeddings, masked=masked)
        node_embs_, element_embs_, labels = self.prepare_for_prediction(
            graph_emb, seeds, self.get_targets_from_nodes, negative_factor=1,
            neg_sampling_strategy=neg_sampling_strategy,
            train_embeddings=train_embeddings, update_embeddings_for_queries=True
        )
        # node_embs_, element_embs_, labels = self._logits_nodes(
        #     graph_emb, self.target_embedder, self.link_predictor,
        #     self._create_loader, seeds, train_embeddings=train_embeddings, neg_sampling_strategy=neg_sampling_strategy,
        #     update_embeddings_for_queries=True
        # )

        acc, loss = self.compute_acc_loss(node_embs_, element_embs_, labels)

        return loss, acc

    def evaluate(self, data_split, neg_sampling_factor=1):
        loss, acc, ndcg = self._evaluate_nodes(
            self.target_embedder, self.link_predictor, self._create_loader, data_split=data_split,
            neg_sampling_factor=neg_sampling_factor
        )
        # ndcg = None
        if data_split == "val":
            self.check_early_stopping(acc)
        return loss, acc, ndcg

    def parameters(self, recurse: bool = True):
        return self.link_predictor.parameters()

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.link_predictor.state_dict().items():
            state_dict[f"link_predictor.{k}"] = v
        return state_dict

    def custom_load_state_dict(self, state_dicts):
        self.link_predictor.load_state_dict(
            self.get_prefix("link_predictor", state_dicts)
        )

    # def _logits_nodes(self, node_embeddings,
    #                   elem_embedder, link_predictor, create_dataloader,
    #                   src_seeds, negative_factor=1, train_embeddings=True, neg_sampling_strategy=None):
    #     k = negative_factor
    #     indices = self.seeds_to_global(src_seeds).tolist()
    #     batch_size = len(indices)
    #
    #     node_embeddings_batch = node_embeddings
    #     next_call_indices = elem_embedder[indices]  # this assumes indices is torch tensor
    #
    #     # dst targets are not unique
    #     unique_dst, slice_map = self._handle_non_unique(next_call_indices)
    #     assert unique_dst[slice_map].tolist() == next_call_indices.tolist()
    #
    #     dataloader = create_dataloader(unique_dst)
    #     input_nodes, dst_seeds, blocks = next(iter(dataloader))
    #     blocks = [blk.to(self.device) for blk in blocks]
    #     assert dst_seeds.shape == unique_dst.shape
    #     assert dst_seeds.tolist() == unique_dst.tolist()
    #     unique_dst_embeddings = self._logits_batch(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
    #     next_call_embeddings = unique_dst_embeddings[slice_map.to(self.device)]
    #     labels_pos = torch.full((batch_size,), self.positive_label, dtype=self.label_dtype)
    #
    #     node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
    #     # negative_indices = torch.tensor(elem_embedder.sample_negative(
    #     #     batch_size * k), dtype=torch.long)  # embeddings are sampled from 3/4 unigram distribution
    #     negative_indices = torch.tensor(self.sample_negative(
    #         batch_size * k, ids=indices, neg_sampling_strategy=neg_sampling_strategy
    #     ), dtype=torch.long)
    #     unique_negative, slice_map = self._handle_non_unique(negative_indices)
    #     assert unique_negative[slice_map].tolist() == negative_indices.tolist()
    #
    #     self.target_embedder.set_embed(unique_dst.detach().cpu().numpy(), unique_dst_embeddings.detach().cpu().numpy())
    #
    #     dataloader = create_dataloader(unique_negative)
    #     input_nodes, dst_seeds, blocks = next(iter(dataloader))
    #     blocks = [blk.to(self.device) for blk in blocks]
    #     assert dst_seeds.shape == unique_negative.shape
    #     assert dst_seeds.tolist() == unique_negative.tolist()
    #     unique_negative_random = self._logits_batch(input_nodes, blocks, train_embeddings)  # use_types, ntypes)
    #     negative_random = unique_negative_random[slice_map.to(self.device)]
    #     labels_neg = torch.full((batch_size * k,), self.negative_label, dtype=self.label_dtype)
    #
    #     self.target_embedder.set_embed(unique_negative.detach().cpu().numpy(), unique_negative_random.detach().cpu().numpy())
    #
    #     nodes = torch.cat([node_embeddings_batch, node_embeddings_neg_batch], dim=0)
    #     embs = torch.cat([next_call_embeddings, negative_random], dim=0)
    #     labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)
    #     return nodes, embs, labels


# class GraphLinkTypeObjective(GraphLinkObjective):
#     def __init__(
#             self, name, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_ndcg=False, dilate_ndcg=1
#     ):
#         self.set_num_classes(data_loading_func)
#
#         super(GraphLinkObjective, self).__init__(
#             name, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
#         )
#
#     def set_num_classes(self, data_loading_func):
#         pass
#
#     def create_nn_link_type_predictor(self):
#         self.link_predictor = LinkClassifier(2 * self.graph_model.emb_size, self.num_classes).to(self.device)
#         self.positive_label = 1
#         self.negative_label = 0
#         self.label_dtype = torch.long
#
#     def create_link_predictor(self):
#         if self.link_predictor_type == "nn":
#             self.create_nn_link_predictor()
#         elif self.link_predictor_type == "inner_prod":
#             self.create_inner_prod_link_predictor()
#         else:
#             raise NotImplementedError()

