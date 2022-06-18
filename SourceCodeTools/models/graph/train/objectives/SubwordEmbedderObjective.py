import logging
from collections import OrderedDict
from itertools import chain

from SourceCodeTools.models.graph.TargetEmbedder import TargetEmbedderWithBpeSubwords
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective


class SubwordEmbedderObjective(AbstractObjective):
    def __init__(self, **kwargs):
        super(SubwordEmbedderObjective, self).__init__(**kwargs)

        self.target_embedding_fn = self.get_targets_from_embedder
        self.negative_factor = 1
        self.update_embeddings_for_queries = True

    # def _verify_parameters(self):
    #     if self.link_predictor_type == "inner_prod":
    #         assert self.target_emb_size == self.graph_model.emb_size, "Graph embedding and target embedder dimensionality should match for `inner_prod` type of link predictor."

    def _verify_parameters(self):
        if self.link_predictor_type == "inner_prod":
            if self.graph_model.emb_size != self.target_emb_size:
                self.target_emb_size = self.graph_model.emb_size
                logging.warning(f"Graph embedding and target embedding sizes do not match. "
                                f"Fixing...set to {self.graph_model.emb_size}")
        pass

    def _create_target_embedder(self, target_emb_size, tokenizer_path):
        self.target_embedder = TargetEmbedderWithBpeSubwords(
            self.dataloader.label_encoder.get_original_targets(), emb_size=target_emb_size, num_buckets=200000,
            max_len=20, tokenizer_path=tokenizer_path
        )

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