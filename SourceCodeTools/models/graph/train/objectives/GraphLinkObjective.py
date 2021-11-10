from collections import OrderedDict

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective


class GraphLinkObjective(AbstractObjective):
    def __init__(
            self, name, graph_model, node_embedder, dataset, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, nn_index="brute", ns_groups=None
    ):
        super(GraphLinkObjective, self).__init__(
            name, graph_model, node_embedder, dataset, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, nn_index=nn_index,
            ns_groups=ns_groups
        )
        self.target_embedding_fn = self.get_targets_from_nodes
        self.negative_factor = 1
        self.update_embeddings_for_queries = True

    def verify_parameters(self):
        pass

    def create_target_embedder(self, data_loading_func, nodes, edges, tokenizer_path):
        self.create_graph_link_sampler(data_loading_func, nodes, edges)

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
#             self, name, graph_model, node_embedder, dataset, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         self.set_num_classes(data_loading_func)
#
#         super(GraphLinkObjective, self).__init__(
#             name, graph_model, node_embedder, dataset, data_loading_func, device,
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

