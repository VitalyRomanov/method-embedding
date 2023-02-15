from collections import OrderedDict

import torch

from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective, ObjectiveOutput


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
        pass

    def get_inner_prod_link_scorer_class(self):
        from SourceCodeTools.models.graph.LinkPredictor import DirectedCosineLinkScorer
        return DirectedCosineLinkScorer

    def forward(
            self, input_nodes, input_mask, blocks, positive_indices, negative_indices,
            update_ns_callback=None, subgraph=None, **kwargs
    ):
        gnn_output = self._graph_embeddings(input_nodes, blocks, mask=input_mask)
        unique_embeddings = gnn_output.output

        all_embeddings = unique_embeddings[kwargs["slice_map"]]

        graph_embeddings = all_embeddings[kwargs["src_nodes_mask"]]
        positive_embeddings = all_embeddings[kwargs["positive_nodes_mask"]]
        negative_embeddings = all_embeddings[kwargs["negative_nodes_mask"]]

        non_src_nodes_mask = ~kwargs["src_nodes_mask"]
        non_src_ids = kwargs["compute_embeddings_for"][non_src_nodes_mask]
        non_src_embeddings = all_embeddings[non_src_nodes_mask].cpu().detach().numpy()
        update_ns_callback(non_src_ids, non_src_embeddings)

        labels_pos = self._create_positive_labels(positive_indices).to(self.device)
        labels_neg = self._create_negative_labels(negative_embeddings).to(self.device)

        pos_neg_scores, avg_scores, loss = self._compute_scores_loss(
            graph_embeddings, positive_embeddings, negative_embeddings, labels_pos, labels_neg
        )

        return ObjectiveOutput(
            gnn_output=gnn_output,
            logits=pos_neg_scores,
            labels=(labels_pos, labels_neg),
            loss=loss,
            scores=avg_scores,
            prediction=(
                torch.softmax(pos_neg_scores[0], dim=-1),
                torch.softmax(pos_neg_scores[1], dim=-1) if pos_neg_scores[1] is not None else None
            )
        )

    def parameters(self, recurse: bool = True):
        return self.link_scorer.parameters()

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.link_scorer.state_dict().items():
            state_dict[f"link_scorer.{k}"] = v
        return state_dict

    def custom_load_state_dict(self, state_dicts):
        self.link_scorer.load_state_dict(
            self.get_prefix("link_predictor", state_dicts)
        )


# class GraphLinkTypeObjective(GraphLinkObjective):
#     def __init__(
#             self, name, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_scorer_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         self.set_num_classes(data_loading_func)
#
#         super(GraphLinkObjective, self).__init__(
#             name, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_scorer_type=link_scorer_type,
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
#         if self.link_scorer_type == "nn":
#             self.create_nn_link_scorer()
#         elif self.link_scorer_type == "inner_prod":
#             self.create_inner_prod_link_predictor()
#         else:
#             raise NotImplementedError()

