import logging
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from os.path import join

import dgl
import torch
from tqdm import tqdm

from SourceCodeTools.models.graph.LinkPredictor import UndirectedCosineLinkScorer, BilinearLinkClassifier, \
    UndirectedL2LinkScorer, ScorerObjectiveOptimizationDirection

import torch.nn as nn

from SourceCodeTools.models.graph.TargetEmbedder import TargetEmbedder
from SourceCodeTools.models.graph.train.Scorer import Scorer


class ZeroEdges(Exception):
    def __init__(self, *args):
        super(ZeroEdges, self).__init__(*args)


class EarlyStoppingTracker:
    def __init__(self, early_stopping_tolerance):
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_counter = 0
        self.early_stopping_value = 0.
        self.early_stopping_trigger = False

    def should_stop(self, metric):
        if metric <= self.early_stopping_value:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_tolerance:
                return True
        else:
            self.early_stopping_counter = 0
            self.early_stopping_value = metric
        return False

    def reset(self):
        self.early_stopping_counter = 0
        self.early_stopping_value = 0.


class ScoringMethods(Enum):
    nn = 0
    l2 = 1
    inner_prod = 3


class AbstractObjective(nn.Module):
    def __init__(
            self, *, name, graph_model, node_embedder, dataset, label_load_fn, device,
            sampling_neighbourhood_size, batch_size, labels_for, number_of_hops, preload_for="package",
            masker_fn=None, label_loader_class=None, label_loader_params=None, dataloader_class=None,
            tokenizer_path=None, target_emb_size=None, link_scorer_type="inner_prod",
            measure_scores=False, dilate_scores=1, early_stopping=False, early_stopping_tolerance=20, nn_index="brute",
            model_base_path=None, force_w2v=False, neg_sampling_factor=1, use_ns_groups=False, embedding_table_size=300000
    ):
        super(AbstractObjective, self).__init__()

        self.name = name
        self.graph_model = graph_model
        self.sampling_neighbourhood_size = sampling_neighbourhood_size
        self.batch_size = batch_size
        self.target_emb_size = target_emb_size
        self.node_embedder = node_embedder
        self.device = device
        self.dataloader_class = dataloader_class
        self.link_scorer_type = ScoringMethods[link_scorer_type]
        self.measure_scores = measure_scores
        self.dilate_scores = dilate_scores
        self.nn_index = nn_index
        self.early_stopping_tracker = EarlyStoppingTracker(early_stopping_tolerance) if early_stopping else None
        self.early_stopping_trigger = False
        self.base_path = model_base_path
        self.force_w2v = force_w2v
        self.use_ns_groups = use_ns_groups
        self.embedding_table_size = embedding_table_size
        self.neg_sampling_factor = neg_sampling_factor

        self._verify_parameters()

        self._create_dataloader(
            dataset, labels_for, number_of_hops, batch_size, preload_for=preload_for,
            labels=label_load_fn(), masker_fn=masker_fn, label_loader_class=label_loader_class,
            label_loader_params=label_loader_params
        )
        self._create_target_embedder(target_emb_size, tokenizer_path)

        self._create_link_scorer()
        self._create_global_scorers()

        self.target_embedding_fn = self.get_targets_from_embedder
        self.negative_factor = 1
        self.update_embeddings_for_queries = True

        self.train_proximity_ns_warmup_complete = False
        self.val_proximity_ns_warmup_complete = False
        self.test_proximity_ns_warmup_complete = False

    def _verify_parameters(self):
        pass

    def _create_dataloader(
            self, dataset, labels_for, number_of_hops, batch_size, preload_for="package", labels=None,
            masker_fn=None, label_loader_class=None, label_loader_params=None
    ):
        self.dataloader = self.dataloader_class(
            dataset, labels_for, number_of_hops, batch_size, preload_for=preload_for, labels=labels,
            masker_fn=masker_fn, label_loader_class=label_loader_class, label_loader_params=label_loader_params,
            negative_sampling_strategy="w2v" if self.force_w2v else "closest", neg_sampling_factor=self.neg_sampling_factor,
            base_path=self.base_path, objective_name=self.name, device=self.device, embedding_table_size=self.embedding_table_size
        )
        self._create_loaders()

    def _create_target_embedder(self, target_emb_size, tokenizer_path):
        self.target_embedder = TargetEmbedder(
            # self.dataloader.label_encoder.get_original_targets()
            self.dataloader.label_encoder, emb_size=target_emb_size, num_buckets=200000,
            max_len=20, tokenizer_path=tokenizer_path
        ).to(self.device)

    def _create_global_scorers(self):
        for partition in ["train", "test", "val"]:
            setattr(
                self,
                f"{partition}_scorer",
                Scorer(getattr(self.dataloader, f"{partition}_loader"))
            )

    @property
    def positive_label(self):
        return self.link_scorer.positive_label

    @property
    def negative_label(self):
        return self.link_scorer.negative_label

    @property
    def label_dtype(self):
        return self.link_scorer.label_dtype

    def _adapt_scores_to_minimization_objective(self, scores):
        if self.link_scorer.optimization_direction == ScorerObjectiveOptimizationDirection.maximize:
            return 1. - scores
        else:
            return scores

    def get_nn_link_scorer_class(self):
        return BilinearLinkClassifier

    def get_inner_prod_link_scorer_class(self):
        return UndirectedCosineLinkScorer

    def get_l2_link_scorer_class(self):
        return UndirectedL2LinkScorer

    def _create_link_scorer_class(self, scorer_class, **kwargs):
        self.link_scorer = scorer_class(
            input_dim1=self.target_emb_size,
            input_dim2=self.graph_model.emb_size,
            num_classes=2, **kwargs
        ).to(self.device)

    def _create_nn_link_scorer(self):
        self._create_link_scorer_class(self.get_nn_link_scorer_class())
        self._loss_op = nn.CrossEntropyLoss()

        def compute_loss(positive_scores, positive_labels, negative_scores, negative_labels):
            scores = torch.cat([positive_scores, negative_scores], dim=0)
            labels = torch.cat([positive_labels, negative_labels], dim=0)

            scores = self._adapt_scores_to_minimization_objective(scores)

            return self._loss_op(scores, labels)

        def compute_average_score(scores, labels=None):
            assert len(scores.shape) > 1 and scores.shape[1] > 1
            sm = nn.Softmax(dim=-1)
            return sm(scores)[torch.full(scores.shape, False).scatter_(1, labels.reshape(-1, 1), True)].mean().item()
            # return compute_accuracy(scores.argmax(dim=-1), labels)

        self._compute_loss = compute_loss
        self._compute_average_score = compute_average_score

    def _create_inner_prod_link_scorer(self):
        self._create_link_scorer_class(self.get_inner_prod_link_scorer_class())

        margin_value = 0.8

        self._loss_op = nn.HingeEmbeddingLoss(margin=margin_value)

        def compute_loss(positive_scores, positive_labels, negative_scores, negative_labels):
            scores = torch.cat([positive_scores, negative_scores], dim=0)
            labels = torch.cat([positive_labels, negative_labels], dim=0)

            scores = self._adapt_scores_to_minimization_objective(scores)

            return self._loss_op(scores, labels)

        # self._loss_op = nn.TripletMarginWithDistanceLoss(
        #     distance_function=lambda anchor, candidate: candidate,
        #     margin=margin_value
        # )
        #
        # def compute_loss(positive_scores, positive_labels, negative_scores, negative_labels):
        #     return self._loss_op(None, positive_scores, negative_scores)

        def compute_average_score(scores, labels=None):
            return scores.mean().item()

        self._compute_loss = compute_loss
        self._compute_average_score = compute_average_score

    def _create_l2_link_scorer(self):
        self._create_link_scorer_class(self.get_l2_link_scorer_class())

        margin_value = 0.2
        self._loss_op = nn.HingeEmbeddingLoss(margin=margin_value)
        # self._loss_op = nn.MSELoss()
        # self._loss_op = nn.TripletMarginWithDistanceLoss(
        #     distance_function=lambda anchor, candidate: candidate,
        #     margin=margin_value
        # )

        def compute_loss(positive_scores, positive_labels, negative_scores, negative_labels):
            scores = torch.cat([positive_scores, negative_scores], dim=0)
            labels = torch.cat([positive_labels, negative_labels], dim=0)

            scores = self._adapt_scores_to_minimization_objective(scores)

            return self._loss_op(scores, labels)

        def compute_average_score(scores, labels=None):
            return scores.mean().item()

        self._compute_loss = compute_loss
        self._compute_average_score = compute_average_score

    def _create_link_scorer(self):
        if self.link_scorer_type == ScoringMethods.nn:
            self._create_nn_link_scorer()
        elif self.link_scorer_type == ScoringMethods.inner_prod:
            self._create_inner_prod_link_scorer()
        elif self.link_scorer_type == ScoringMethods.l2:
            self._create_l2_link_scorer()
        else:
            raise NotImplementedError()

        self._loss_op.to(self.device)

    def _create_loaders(self):
        self.train_loader = self.dataloader.partition_iterator("train")
        self.val_loader = self.dataloader.partition_iterator("val")
        self.test_loader = self.dataloader.partition_iterator("test")

        # def get_num_batches_estimate(ids):
        #     return len(ids) // self.batch_size + 1

        self.num_train_batches = self.dataloader.train_num_batches
        self.num_test_batches = self.dataloader.test_num_batches
        self.num_val_batches = self.dataloader.val_num_batches

    @staticmethod
    def _idx_len(idx):
        if isinstance(idx, dict):
            length = 0
            for key in idx:
                length += len(idx[key])
        else:
            length = len(idx)
        return length

    @staticmethod
    def _handle_non_unique(non_unique_ids):
        id_list = non_unique_ids.tolist()
        unique_ids = list(set(id_list))
        new_position = dict(zip(unique_ids, range(len(unique_ids))))
        slice_map = torch.tensor(list(map(lambda x: new_position[x], id_list)), dtype=torch.long)
        return torch.tensor(unique_ids, dtype=torch.long), slice_map

    def _create_loader(self, graph, indices):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.graph_model.num_layers)
        return dgl.dataloading.NodeDataLoader(
            graph, {"node_": indices}, sampler, batch_size=len(indices), num_workers=0)

    def _extract_embed(self, input_nodes, mask=None):
        # emb = {}
        # for node_type, nid in input_nodes.items():
        #     emb[node_type] = self.node_embedder(nid, mask[node_type]).to(self.device)
        # return emb
        return self.node_embedder(input_nodes, mask).to(self.device)

    def _compute_scores_loss(self, node_embs, positive_embs, negative_embs, positive_labels, negative_labels):

        positive_scores = self.link_scorer(node_embs, positive_embs, labels=positive_labels)

        negative_scores = self.link_scorer(
            node_embs.tile(negative_embs.size(0) // node_embs.size(0), 1),
            negative_embs, labels=negative_labels
        )

        loss = self._compute_loss(positive_scores, positive_labels, negative_scores, negative_labels)

        with torch.no_grad():
            scores = {
                f"positive_score/{self.link_scorer_type.name}": self._compute_average_score(positive_scores, positive_labels),
                f"negative_score/{self.link_scorer_type.name}": self._compute_average_score(negative_scores, negative_labels)
            }

        return (positive_scores, negative_scores), scores, loss

    @staticmethod
    def _wrap_into_dict(input):
        if isinstance(input, torch.Tensor):
            return {"node_": input}
        else:
            return input

    def _graph_embeddings(self, input_nodes, blocks, mask=None):
        emb = self._extract_embed(input_nodes, mask=mask)
        graph_emb = self.graph_model(self._wrap_into_dict(emb), blocks)
        return graph_emb["node_"]

    def _create_positive_labels(self, ids, positive_targets=None):
        return torch.full((len(ids),), self.positive_label, dtype=self.label_dtype).to(self.device)

    def _create_negative_labels(self, ids, negative_targets=None):
        return torch.full((len(ids),), self.negative_label, dtype=self.label_dtype).to(self.device)

    def _prepare_for_prediction(
            self, node_embeddings, positive_indices, negative_indices, target_embedding_fn, update_ns_callback
    ):
        positive_dst, negative_dst = target_embedding_fn(
            positive_indices, negative_indices, update_ns_callback
        )

        # TODO breaks cache in
        #  SourceCodeTools.models.graph.train.objectives.GraphLinkClassificationObjective.TargetLinkMapper.get_labels
        labels_pos = self._create_positive_labels(positive_indices)
        labels_neg = self._create_negative_labels(negative_indices)

        return node_embeddings, positive_dst, negative_dst, labels_pos, labels_neg

        # num_positive = positive_indices.size(0)
        # num_negative = negative_indices.size(0)
        # tile_factor = 1 + num_negative // num_positive
        # src_embs = torch.tile(node_embeddings, (tile_factor, 1)) # src_embs = torch.cat([node_embeddings, node_embeddings], dim=0)

        # dst_embs = torch.cat([positive_dst, negative_dst], dim=0)
        # labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)
        # return src_embs, dst_embs, labels

    @staticmethod
    def _sum_scores(s):
        n = len(s)
        if n == 0:
            n += 1
        return sum(s) / n

    def _update_num_batches_for_split(self, data_split, batch_num):
        if batch_num > getattr(self, f"num_{data_split}_batches"):
            setattr(self, f"num_{data_split}_batches", batch_num + 1)

    # def _create_scorer(self, target_loader):
    #     scorer = Scorer(target_loader)
    #     return scorer

    def _warmup_if_needed(self, partition, update_ns_callback):
        warmup_flag_name = f"{partition}_proximity_ns_warmup_complete"
        warmup_complete = getattr(self, warmup_flag_name)
        if self.update_embeddings_for_queries and warmup_complete is False and self.target_embedder is not None:
            self._warm_up_proximity_ns(update_ns_callback)
            setattr(self, warmup_flag_name, True)

    def _do_score_measurement(self, batch, graph_emb, longterm_metrics, scorer, **kwargs):
        scores_ = scorer.score_candidates(
            batch["indices"], graph_emb, self.link_scorer, at=[1, 3, 5, 10],
            type=self.link_scorer_type.name, device=self.device
        )
        for key, val in scores_.items():
            longterm_metrics[key].append(val)

    def make_step(
            self, batch_ind, batch, partition, longterm_metrics, scorer
    ):
        self._update_num_batches_for_split(partition, batch_ind)

        labels_loader = batch["labels_loader"]
        # blocks = batch["blocks"]
        # input_nodes = batch["input_nodes"]
        # input_mask = batch["input_mask"]
        # positive_indices = batch["positive_indices"]
        # negative_indices = batch["negative_indices"]
        update_ns_callback = labels_loader.set_embed
        # graph = batch["subgraph"]

        self._warmup_if_needed(partition, update_ns_callback)

        if batch_ind % 10 == 0:
            labels_loader.update_index()

        # do_break = False
        # for block in blocks:
        #     if block.num_edges() == 0:
        #         do_break = True
        # if do_break:
        #     return None, None

        # try:
        graph_emb, logits, labels, loss, scores_ = self(
            # input_nodes, input_mask, blocks, positive_indices, negative_indices,
            # update_ns_callback=update_ns_callback  #, graph=graph  # do not pass graph, possibly increases cache size
            update_ns_callback=update_ns_callback, **batch
        )

        # loss = loss / len(self.objectives)  # assumes the same batch size for all objectives
        # for groups in self.optimizer.param_groups:
        #     for param in groups["params"]:
        #         torch.nn.utils.clip_grad_norm_(param, max_norm=1.)
        # loss.backward()  # create_graph = True

        if partition == "train":
            scores = {"Loss": loss.item()} #, "Accuracy": acc}
            scores.update(scores_)
            longterm_metrics[f"Loss/{partition}_avg/{self.name}"].append(loss.item())
            for key, val in scores_.items():
                longterm_metrics[f"{key}/{partition}_avg/{self.name}"].append(val)
        else:
            scores = {}
            if self.measure_scores:
                if batch_ind % self.dilate_scores == 0:
                    self._do_score_measurement(batch, graph_emb, longterm_metrics, scorer, y_true=labels, logits=logits)
            longterm_metrics["Loss"].append(loss.item())
            for key, val in scores_.items():
                longterm_metrics[f"{key}"].append(val)
            # longterm_metrics["Accuracy"].append(acc)

        return loss, scores

    def _evaluate_objective(self, data_split):
        longterm_scores = defaultdict(list)
        scorer = getattr(self, f"{data_split}_scorer")

        for batch_ind, batch in enumerate(tqdm(
                self.get_iterator(data_split), total=getattr(self, f"num_{data_split}_batches")
        )):
            try:
                _, _ = self.make_step(batch_ind, batch, data_split, longterm_scores, scorer)
            except RuntimeError:
                with open(join(self.base_path, "batch_errors.log"), "a") as error_log:
                    error_log.write(
                        f"Runtime error at training step {batch_ind}\n"
                    )
                    torch.cuda.empty_cache()

        return longterm_scores

    def _check_early_stopping(self, metric):
        """
        Checks the metric value and raises Early Stopping when the metric stops increasing.
            Assumes that the metric grows. Uses accuracy as a metric by default. Check implementation of child classes
        :param metric: metric value
        :return: Nothing
        """
        if self.early_stopping_tracker is not None:
            self.early_stopping_trigger = self.early_stopping_tracker.should_stop(metric)

    def _warm_up_proximity_ns(self, update_ns_callback):
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield torch.LongTensor(lst[i:i + n])

        all_keys = self.target_embedder.keys()
        batches = chunks(all_keys, self.batch_size)
        # for batch in tqdm(
        #         batches,
        #         total=len(all_keys) // self.trainer_params["batch_size"] + 1,
        #         desc="Precompute Target Embeddings", leave=True
        # ):
        logging.info('Warm up proximity negative sampler')
        for batch in batches:
            _ = self.target_embedding_fn(batch, update_ns_callback=update_ns_callback)  # scorer embedding updated inside

        # self.proximity_ns_warmup_complete = True

    def get_targets_from_nodes(
            self, positive_indices, negative_indices=None, update_ns_callback=None, graph=None
    ):
        # negative_indices = torch.tensor(negative_indices, dtype=torch.long) if negative_indices is not None else None
        # TODO
        # try to do this in forward() computing graph embeddings three times is too expensive

        def get_embeddings_for_targets(dst):
            unique_dst, slice_map = self._handle_non_unique(dst)
            assert unique_dst[slice_map].tolist() == dst.tolist()

            dataloader = self._create_loader(graph, unique_dst)
            input_nodes, dst_seeds, blocks = next(iter(dataloader))
            blocks = [blk.to(self.device) for blk in blocks]
            assert dst_seeds.shape == unique_dst.shape
            assert dst_seeds.tolist() == unique_dst.tolist()
            input_ = blocks[0].srcnodes["node_"].data["embedding_id"].to(self.device)
            assert -1 not in input_.cpu().tolist()
            unique_dst_embeddings = self._graph_embeddings(input_, blocks)  # use_types, ntypes)
            dst_embeddings = unique_dst_embeddings[slice_map.to(self.device)]

            if self.update_embeddings_for_queries and update_ns_callback is not None:
                update_ns_callback(unique_dst.detach().cpu().numpy(), unique_dst_embeddings.detach().cpu().numpy())

            return dst_embeddings

        positive_dst = get_embeddings_for_targets(positive_indices)
        negative_dst = get_embeddings_for_targets(negative_indices) if negative_indices is not None else None
        return positive_dst, negative_dst

    def get_targets_from_embedder(
            self, positive_indices, negative_indices=None, update_ns_callback=None, *args, **kwargs
    ):

        def get_embeddings_for_targets(dst):
            unique_dst, slice_map = self._handle_non_unique(dst)
            assert unique_dst[slice_map].tolist() == dst.tolist()
            unique_dst_embeddings = self.target_embedder(unique_dst.to(self.device))
            dst_embeddings = unique_dst_embeddings[slice_map.to(self.device)]

            if self.update_embeddings_for_queries:
                update_ns_callback(unique_dst.detach().cpu().numpy(), unique_dst_embeddings.detach().cpu().numpy())

            return dst_embeddings

        # positive_dst = self.target_embedder(positive_indices.to(self.device))
        # negative_dst = self.target_embedder(negative_indices.to(self.device)) if negative_indices is not None else None
        #
        positive_dst = get_embeddings_for_targets(positive_indices)
        negative_dst = get_embeddings_for_targets(negative_indices) if negative_indices is not None else None

        return positive_dst, negative_dst

    def forward(
            self, input_nodes, input_mask, blocks, positive_indices, negative_indices,
            update_ns_callback=None, **kwargs
    ):
        graph_emb = self._graph_embeddings(input_nodes, blocks, mask=input_mask)
        _, positive_emb, negative_emb, labels_pos, labels_neg = self._prepare_for_prediction(
            graph_emb, positive_indices, negative_indices, self.target_embedding_fn, update_ns_callback  # , subgraph
        )

        pos_neg_scores, avg_scores, loss  = self._compute_scores_loss(
            graph_emb, positive_emb, negative_emb, labels_pos, labels_neg
        )

        return graph_emb, pos_neg_scores, (labels_pos, labels_neg), loss, avg_scores

    def evaluate(self, data_split, *, neg_sampling_strategy=None, early_stopping=False, early_stopping_tolerance=20):
        scores = self._evaluate_objective(data_split)
        # if data_split == "val":
        #     self._check_early_stopping(scores["Accuracy"])
        return scores

    @abstractmethod
    def parameters(self, recurse: bool = True):
        raise NotImplementedError()

    @abstractmethod
    def custom_state_dict(self):
        raise NotImplementedError()

    @abstractmethod
    def custom_load_state_dict(self, state_dicts):
        raise NotImplementedError()

    @staticmethod
    def get_prefix(prefix, state_dict):
        return {key.replace(f"{prefix}.", ""): val for key, val in state_dict.items() if key.startswith(prefix)}

    # def reset_iterator(self, data_split):
    #     iter_name = f"{data_split}_loader_iter"
    #     setattr(self, iter_name, iter(getattr(self, f"{data_split}_loader")))

    def get_iterator(self, data_split):
        return iter(getattr(self, f"{data_split}_loader"))

    # def loader_next(self, data_split):
    #     iter_name = f"{data_split}_loader_iter"
    #     if not hasattr(self, iter_name):
    #         setattr(self, iter_name, iter(getattr(self, f"{data_split}_loader")))
    #     return next(getattr(self, iter_name))
