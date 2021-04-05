from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.GraphLinkObjective import GraphLinkObjective, GraphLinkTypeObjective
from SourceCodeTools.models.graph.train.objectives.SubwordEmbedderObjective import SubwordEmbedderObjective
from SourceCodeTools.models.graph.train.objectives.TextPredictionObjective import GraphTextPrediction, GraphTextGeneration


class TokenNamePrediction(SubwordEmbedderObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(TokenNamePrediction, self).__init__(
            "TokenNamePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )


class NodeNamePrediction(SubwordEmbedderObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(NodeNamePrediction, self).__init__(
            "NodeNamePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )


class VariableNameUsePrediction(SubwordEmbedderObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(VariableNameUsePrediction, self).__init__(
            "VariableNamePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )


class NextCallPrediction(GraphLinkObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(NextCallPrediction, self).__init__(
            "NextCallPrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )


class GlobalLinkPrediction(GraphLinkObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(GlobalLinkPrediction, self).__init__(
            "GlobalLinkPrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )


class LinkTypePrediction(GraphLinkTypeObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(GraphLinkTypeObjective, self).__init__(
            "LinkTypePrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )
