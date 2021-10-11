from collections import OrderedDict
from itertools import chain

from SourceCodeTools.code.data.sourcetrail.SubwordMasker import SubwordMasker
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective


class SubwordEmbedderObjective(AbstractObjective):
    def __init__(
            self, name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super(SubwordEmbedderObjective, self).__init__(
            name, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )
        self.target_embedding_fn = self.get_targets_from_embedder
        self.negative_factor = 1
        self.update_embeddings_for_queries = True

    def verify_parameters(self):
        if self.link_predictor_type == "inner_prod":
            assert self.target_emb_size == self.graph_model.emb_size, "Graph embedding and target embedder dimensionality should match for `inner_prod` type of link predictor."

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.create_subword_embedder(data_loading_func, nodes, tokenizer_path)

    def create_link_predictor(self):
        if self.link_predictor_type == "nn":
            self.create_nn_link_predictor()
        elif self.link_predictor_type == "inner_prod":
            self.create_inner_prod_link_predictor()
        else:
            raise NotImplementedError()

    # def forward(self, input_nodes, seeds, blocks, train_embeddings=True, neg_sampling_strategy=None):
    #     masked = self.masker.get_mask(self.seeds_to_python(seeds)) if self.masker is not None else None
    #     graph_emb = self._graph_embeddings(input_nodes, blocks, train_embeddings, masked=masked)
    #     node_embs_, element_embs_, labels = self.prepare_for_prediction(
    #         graph_emb, seeds, self.get_targets_from_embedder, negative_factor=1,
    #         neg_sampling_strategy=None, train_embeddings=train_embeddings,
    #         update_embeddings_for_queries=False
    #     )
    #     # node_embs_, element_embs_, labels = self._logits_embedder(
    #     #     graph_emb, self.target_embedder, self.link_predictor, seeds, neg_sampling_strategy=neg_sampling_strategy
    #     # )
    #     acc, loss = self.compute_acc_loss(node_embs_, element_embs_, labels)
    #
    #     return loss, acc

    # def train(self):
    #     pass

    def evaluate(self, data_split, neg_sampling_factor=1):
        loss, acc, ndcg = self._evaluate_embedder(
            self.target_embedder, self.link_predictor, data_split=data_split,
            neg_sampling_factor=neg_sampling_factor
        )
        if data_split == "val":
            self.check_early_stopping(acc)
        return loss, acc, ndcg

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