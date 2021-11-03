from collections import OrderedDict
from itertools import chain

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective


class SubwordEmbedderObjective(AbstractObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, nn_index="brute"
    ):
        super(SubwordEmbedderObjective, self).__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores, nn_index=nn_index
        )
        self.target_embedding_fn = self.get_targets_from_embedder
        self.negative_factor = 1
        self.update_embeddings_for_queries = False

    def verify_parameters(self):
        if self.link_predictor_type == "inner_prod":
            assert self.target_emb_size == self.graph_model.emb_size, "Graph embedding and target embedder dimensionality should match for `inner_prod` type of link predictor."

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.create_subword_embedder(data_loading_func, nodes, tokenizer_path)

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