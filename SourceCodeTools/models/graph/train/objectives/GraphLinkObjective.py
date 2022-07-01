from collections import OrderedDict

import torch

from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective


class GraphLinkObjective(AbstractObjective):
    def __init__(self, **kwargs):
        super(GraphLinkObjective, self).__init__(**kwargs)

        self.target_embedding_fn = self.get_targets_from_nodes
        self.negative_factor = 1
        self.update_embeddings_for_queries = True

    def _verify_parameters(self):
        pass

    def _warmup_if_needed(self, partition, update_ns_callback):
        pass

    def _create_target_embedder(self, data_loading_func, tokenizer_path):
        # raise NotImplementedError()
        pass
        # self.create_graph_link_sampler(data_loading_func, nodes)

    # def _prepare_for_prediction(
    #         self, node_embeddings, positive_embeddings, negative_embeddings, *args, **kwargs
    # ):
    #     # TODO breaks cache in
    #     #  SourceCodeTools.models.graph.train.objectives.GraphLinkClassificationObjective.TargetLinkMapper.get_labels
    #     labels_pos = self._create_positive_labels(torch.zeros(positive_embeddings.size(0)))
    #     labels_neg = self._create_negative_labels(torch.zeros(negative_embeddings.size(0)))
    #
    #     src_embs = torch.cat([node_embeddings, node_embeddings], dim=0)
    #     dst_embs = torch.cat([positive_embeddings, negative_embeddings], dim=0)
    #     labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)
    #     return src_embs, dst_embs, labels

    def forward(
            self, input_nodes, input_mask, blocks, positive_indices, negative_indices,
            update_ns_callback=None, graph=None
    ):
        seeds = blocks[-1].dstnodes["node_"].data["_ID"]
        all_nodes = torch.cat([seeds, positive_indices, negative_indices])
        all_embeddigs, _ = self.get_targets_from_nodes(all_nodes, graph=graph)

        graph_embeddings = all_embeddigs[:len(seeds), :]
        positive_embeddings = all_embeddigs[len(seeds): len(seeds) + len(positive_indices), :]
        negative_embeddings = all_embeddigs[len(seeds) + len(positive_indices):, :]

        update_ns_callback(all_nodes[len(seeds):].cpu().detach().numpy(), all_embeddigs[len(seeds):].cpu().detach().numpy())

        # node_embs_, element_embs_, labels = self._prepare_for_prediction(
        #     graph_embeddings, positive_embeddings, negative_embeddings
        # )

        pos_labels = self._create_positive_labels(positive_indices).to(self.device)
        neg_labels = self._create_negative_labels(negative_embeddings).to(self.device)

        pos_logits, pos_acc, pos_loss = self._compute_acc_loss(graph_embeddings, positive_embeddings, pos_labels)
        neg_logits, neg_acc, neg_loss = self._compute_acc_loss(graph_embeddings, negative_embeddings, neg_labels)

        pos_size = positive_indices.size(0)
        neg_size = negative_embeddings.size(0)
        loss = (pos_loss * pos_size + neg_loss * neg_size) / (pos_size + neg_size)
        acc = (pos_acc * pos_size + neg_acc * neg_size) / (pos_size + neg_size)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        # logits, acc, loss  = self._compute_acc_loss(node_embs_, element_embs_, labels)

        return graph_embeddings, logits, labels, loss, acc

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


# class GraphLinkTypeObjective(GraphLinkObjective):
#     def __init__(
#             self, name, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         self.set_num_classes(data_loading_func)
#
#         super(GraphLinkObjective, self).__init__(
#             name, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
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

