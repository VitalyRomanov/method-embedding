from SourceCodeTools.code.data.dataset.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.GraphLinkClassificationObjective import \
    GraphLinkClassificationObjective
from SourceCodeTools.models.graph.train.objectives.GraphLinkObjective import GraphLinkObjective
from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import NodeNameClassifier, NodeClassifierObjective
from SourceCodeTools.models.graph.train.objectives.SubwordEmbedderObjective import SubwordEmbedderObjective
from SourceCodeTools.models.graph.train.objectives.TextPredictionObjective import GraphTextPrediction, GraphTextGeneration


# class TokenNamePrediction(SubwordEmbedderObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         super(TokenNamePrediction, self).__init__(
#             "TokenNamePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
#         )
#
#
# class NodeNamePrediction(SubwordEmbedderObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1, nn_index="brute"
#     ):
#         super(NodeNamePrediction, self).__init__(
#             "NodeNamePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, nn_index=nn_index
#         )
#
#
# class TypeAnnPrediction(NodeClassifierObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type=None, masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         super(TypeAnnPrediction, self).__init__(
#             "TypeAnnPrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             # tokenizer_path=tokenizer_path,
#             target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, dilate_scores=dilate_scores
#         )
#
#
# class VariableNameUsePrediction(SubwordEmbedderObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         super(VariableNameUsePrediction, self).__init__(
#             "VariableNamePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
#         )
#
#
# class NextCallPrediction(GraphLinkObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         super(NextCallPrediction, self).__init__(
#             "NextCallPrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
#         )
#
#
# class GlobalLinkPrediction(GraphLinkObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         super(GlobalLinkPrediction, self).__init__(
#             "GlobalLinkPrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
#         )
#
#
# # class LinkTypePrediction(GraphLinkTypeObjective):
# #     def __init__(
# #             self, graph_model, node_embedder, nodes, data_loading_func, device,
# #             sampling_neighbourhood_size, batch_size,
# #             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
# #             measure_scores=False, dilate_scores=1
# #     ):
# #         super(GraphLinkTypeObjective, self).__init__(
# #             "LinkTypePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
# #             sampling_neighbourhood_size, batch_size,
# #             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
# #             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
# #         )
#
#
# class EdgePrediction(GraphLinkObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1, nn_index="brute", ns_groups=None
#     ):
#         super(EdgePrediction, self).__init__(
#             "EdgePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, nn_index=nn_index,
#             ns_groups=ns_groups
#         )
#         # super(EdgePrediction, self).__init__(
#         #     "LinkTypePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
#         #     sampling_neighbourhood_size, batch_size,
#         #     tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#         #     masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
#         # )
#
# class EdgePrediction2(SubwordEmbedderObjective):
#     def __init__(
#             self, graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
#             measure_scores=False, dilate_scores=1
#     ):
#         super(EdgePrediction2, self).__init__(
#             "EdgePrediction2", graph_model, node_embedder, nodes, data_loading_func, device,
#             sampling_neighbourhood_size, batch_size,
#             tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
#             masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
#         )
#
#     def _create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
#         from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedder
#         self.target_embedder = ElementEmbedder(
#             elements=data_loading_func(), nodes=nodes, emb_size=self.target_emb_size,
#         ).to(self.device)