import os
from collections import defaultdict
from copy import copy
from pathlib import Path
from pprint import pprint
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from time import time
from os.path import join
import logging

from tqdm import tqdm

from SourceCodeTools.code.data.dataset.DataLoader import SGNodesDataLoader, SGEdgesDataLoader, SGSubgraphDataLoader
from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.models.Embedder import Embedder
from SourceCodeTools.models.graph.TargetLoader import TargetLoader, GraphLinkTargetLoader
from SourceCodeTools.models.graph.train.objectives import GraphTextPrediction, GraphTextGeneration, \
    NodeNameClassifier, NodeClassifierObjective, SubwordEmbedderObjective, GraphLinkObjective
from SourceCodeTools.models.graph.NodeEmbedder import SimplestNodeEmbedder
from SourceCodeTools.models.graph.train.objectives.GraphLinkClassificationObjective import TransRObjective
from SourceCodeTools.models.graph.train.objectives.SubgraphClassifierObjective import SubgraphClassifierObjective, \
    SubgraphEmbeddingObjective, SubgraphClassifierObjectiveWithUnetPool, SubgraphClassifierObjectiveWithAttentionPooling


class EarlyStopping(Exception):
    def __init__(self, *args, **kwargs):
        super(EarlyStopping, self).__init__(*args, **kwargs)


class SamplingMultitaskTrainer:

    def __init__(
            self, dataset=None, model_name=None, model_params=None, trainer_params=None, restore=None, device=None,
            pretrained_embeddings_path=None, tokenizer_path=None  #, load_external_dataset=None
    ):
        self._verify_arguments(model_params, trainer_params)

        self.model_params = model_params
        self.trainer_params = trainer_params
        self.device = device
        self.epoch = 0
        self.restore_epoch = 0
        self.batch = 0
        self.dtype = torch.float32
        self.dataset = dataset
        # if load_external_dataset is not None:
        #     logging.info("Loading external dataset")
        #     external_args, external_dataset = load_external_dataset()
        #     self.graph_model.g = external_dataset.g
        #     dataset = external_dataset

        self.create_node_embedder(
            dataset, tokenizer_path, n_dims=model_params["h_dim"],
            pretrained_path=pretrained_embeddings_path, n_buckets=trainer_params["embedding_table_size"]
        )

        self.graph_model = model_name(trainer_params["ntypes"], trainer_params["etypes"], **model_params).to(device)

        self.create_objectives(dataset, tokenizer_path)

        if restore:
            self.restore_from_checkpoint(self.model_base_path)

        # if load_external_dataset is not None:
        #     self.trainer_params["model_base_path"] = external_args.external_model_base

        self._create_optimizer()

        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=1.0)
        # self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=10, cooldown=20)
        self.summary_writer = SummaryWriter(self.model_base_path)

    def _verify_arguments(self, model_params, trainer_params):
        if len(trainer_params["objectives"]) > 1 and trainer_params["early_stopping"] is True:
            print("Early stopping disabled when several objectives are used")
            trainer_params["early_stopping"] = False
        model_params["activation"] = resolve_activation_function(model_params["activation"])
        if model_params["h_dim"] is None:
            print(f"Model parameter `h_dim` is not provided, setting it to: {model_params['node_emb_size']}")
            model_params["h_dim"] = model_params['node_emb_size']

    def create_objectives(self, dataset, tokenizer_path):
        objective_list = self.trainer_params["objectives"]

        self.objectives = nn.ModuleList()
        if "token_pred" in objective_list:
            self.create_token_pred_objective(dataset, tokenizer_path)
        if "node_name_pred" in objective_list:
            self.create_node_name_objective(dataset, tokenizer_path)
        if "var_use_pred" in objective_list:
            self.create_var_use_objective(dataset, tokenizer_path)
        # if "next_call_pred" in objective_list:
        #     self.create_api_call_objective(dataset, tokenizer_path)
        # if "global_link_pred" in objective_list:
        #     self.create_global_link_objective(dataset, tokenizer_path)
        if "edge_pred" in objective_list:
            self.create_edge_objective(dataset, tokenizer_path)
        # if "transr" in objective_list:
        #     self.create_transr_objective(dataset, tokenizer_path)
        # if "doc_pred" in objective_list:
        #     self.create_text_prediction_objective(dataset, tokenizer_path)
        # if "doc_gen" in objective_list:
        #     self.create_text_generation_objective(dataset, tokenizer_path)
        if "node_clf" in objective_list:
            self.create_node_classifier_objective(dataset, tokenizer_path)
        if "type_ann_pred" in objective_list:
            self.create_type_ann_objective(dataset, tokenizer_path)
        # if "subgraph_name_clf" in objective_list:
        #     self.create_subgraph_name_objective(dataset, tokenizer_path)
        if "subgraph_clf" in objective_list:
            self.create_subgraph_classifier_objective(dataset, tokenizer_path)

        if len(self.objectives) == 0:
            raise Exception("No valid objectives provided:", objective_list)

    def _create_node_level_objective(
            self, *, objective_name, objective_class, dataset, labels_fn, tokenizer_path,
            masker_fn=None, preload_for="package", label_loader_class=None, label_loader_params=None,
            dataloader_class=SGNodesDataLoader
    ):
        if label_loader_class is None:
            label_loader_class = TargetLoader

        label_loader_params_ = {"emb_size": self.elem_emb_size, "tokenizer_path": tokenizer_path, "use_ns_groups": self.trainer_params["use_ns_groups"]}
        if label_loader_params is not None:
            label_loader_params_.update(label_loader_params)

        return objective_class(
            name=objective_name, graph_model=self.graph_model, node_embedder=self.node_embedder, dataset=dataset,
            label_load_fn=labels_fn, device=self.device, sampling_neighbourhood_size=self.sampling_neighbourhood_size,
            batch_size=self.batch_size, labels_for="nodes", number_of_hops=self.model_params["n_layers"],
            preload_for=preload_for, masker_fn=masker_fn, label_loader_class=label_loader_class,
            label_loader_params=label_loader_params_, dataloader_class=dataloader_class,
            tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size,
            link_scorer_type=self.trainer_params["metric"],
            measure_scores=self.trainer_params["measure_scores"], dilate_scores=self.trainer_params["dilate_scores"],
            early_stopping=False, early_stopping_tolerance=20, nn_index=self.trainer_params["nn_index"],
            model_base_path=self.model_base_path, force_w2v=self.trainer_params["force_w2v_ns"],
            neg_sampling_factor=self.neg_sampling_factor,
            embedding_table_size=self.trainer_params["embedding_table_size"]
        )

    def _create_subgraph_objective(
            self, *, objective_name, objective_class, dataset, labels_fn, tokenizer_path,
            masker_fn=None, preload_for="file", label_loader_class=None, label_loader_params=None,
            dataloader_class=SGSubgraphDataLoader
    ):
        if label_loader_class is None:
            label_loader_class = TargetLoader

        label_loader_params_ = {"emb_size": self.elem_emb_size, "tokenizer_path": tokenizer_path, "use_ns_groups": self.trainer_params["use_ns_groups"]}
        if label_loader_params is not None:
            label_loader_params_.update(label_loader_params)

        return objective_class(
            name=objective_name, graph_model=self.graph_model, node_embedder=self.node_embedder, dataset=dataset,
            label_load_fn=labels_fn, device=self.device, sampling_neighbourhood_size=self.sampling_neighbourhood_size,
            batch_size=self.batch_size, labels_for="subgraphs", number_of_hops=self.model_params["n_layers"],
            preload_for=preload_for, masker_fn=masker_fn, label_loader_class=label_loader_class,
            label_loader_params=label_loader_params_, dataloader_class=dataloader_class,
            tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size,
            link_scorer_type=self.trainer_params["metric"],
            measure_scores=self.trainer_params["measure_scores"], dilate_scores=self.trainer_params["dilate_scores"],
            early_stopping=False, early_stopping_tolerance=20, nn_index=self.trainer_params["nn_index"],
            model_base_path=self.model_base_path, force_w2v=self.trainer_params["force_w2v_ns"],
            neg_sampling_factor=self.neg_sampling_factor,
            embedding_table_size=self.trainer_params["embedding_table_size"]
        )

    def create_token_pred_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="TokenNamePrediction",
                objective_class=SubwordEmbedderObjective,
                dataset=dataset,
                labels_fn=dataset.load_token_prediction,
                tokenizer_path=tokenizer_path,
                masker_fn=dataset.create_subword_masker,
                preload_for="package"
            )
        )

    def create_node_name_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="NodeNamePrediction",
                objective_class=SubwordEmbedderObjective,
                dataset=dataset,
                labels_fn=dataset.load_node_names,
                tokenizer_path=tokenizer_path,
                masker_fn=dataset.create_node_name_masker,
                preload_for="package"
            )
        )

    def create_subgraph_name_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_subgraph_objective(
                objective_name="SubgraphNameEmbeddingObjective",
                objective_class=SubgraphEmbeddingObjective,
                dataset=dataset,
                tokenizer_path=tokenizer_path,
                labels_fn=dataset.load_subgraph_function_names,
            )
        )

    def create_subgraph_classifier_objective(self, dataset, tokenizer_path):
        from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import ClassifierTargetMapper

        def load_labels():
            filecontent_path = Path(dataset.data_path).joinpath("common_filecontent.json.bz2")
            filecontent = unpersist(filecontent_path)
            return filecontent[["id", "label"]].rename({"id": "src", "label": "dst"}, axis=1)

        self.objectives.append(
            self._create_subgraph_objective(
                objective_name="SubgraphClassifierObjective",
                objective_class=SubgraphClassifierObjective,  # SubgraphClassifierObjectiveWithAttentionPooling,  # SubgraphClassifierObjectiveWithUnetPool,  # SubgraphClassifierObjective,
                dataset=dataset,
                tokenizer_path=tokenizer_path,
                labels_fn=load_labels,
                label_loader_class=ClassifierTargetMapper,
                label_loader_params={"emb_size": None, "tokenizer_path": None, "use_ns_groups": False}
            )
        )

    def create_type_ann_objective(self, dataset, tokenizer_path):
        from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import ClassifierTargetMapper
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="TypeAnnPrediction",
                objective_class=NodeClassifierObjective,
                label_loader_class=ClassifierTargetMapper,
                dataset=dataset,
                labels_fn=dataset.load_type_prediction,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="package",
            )
        )

    def create_node_name_classifier_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="NodeNameClassifier",
                objective_class=NodeClassifierObjective,
                dataset=dataset,
                labels_fn=dataset.load_node_names,
                tokenizer_path=tokenizer_path,
                masker_fn=dataset.create_node_name_masker,
                preload_for="package"
            )
        )

    def create_var_use_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="VariableNameUsePrediction",
                objective_class=SubwordEmbedderObjective,
                dataset=dataset,
                labels_fn=dataset.load_var_use,
                tokenizer_path=tokenizer_path,
                masker_fn=dataset.create_variable_name_masker,
                preload_for="package"
            )
        )

    def create_api_call_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="NextCallPrediction",
                objective_class=GraphLinkObjective,
                dataset=dataset,
                labels_fn=dataset.load_api_call,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="package"
            )
        )

    def create_global_link_objective(self, dataset, tokenizer_path):
        assert dataset.no_global_edges is True, "No edges should be in the graph for GlobalLinkPrediction objective"

        self.objectives.append(
            self._create_node_level_objective(
                objective_name="GlobalLinkPrediction",
                objective_class=GraphLinkObjective,
                dataset=dataset,
                labels_fn=dataset.load_global_edges_prediction,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="package"  # "file", "function"
            )
        )

    def create_edge_objective(self, dataset, tokenizer_path):
        assert self.trainer_params["use_ns_groups"] is True, "Parameter `use_ns_groups` should be set to True " \
                                                             "for edge link prediction"
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="EdgePrediction",
                objective_class=GraphLinkObjective,
                dataset=dataset,
                labels_fn=dataset.load_edge_prediction,
                label_loader_class=GraphLinkTargetLoader,
                label_loader_params={"compact_dst": False},
                dataloader_class=SGEdgesDataLoader,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="package" # "file", "mention"
            )
        )

    def create_transr_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="TransRObjective",
                objective_class=TransRObjective,
                dataset=dataset,
                labels_fn=dataset.load_edge_prediction,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="package"  # "file", "function"
            )
        )

    def create_text_prediction_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="GraphTextPrediction",
                objective_class=GraphTextPrediction,
                dataset=dataset,
                labels_fn=dataset.load_docstring,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="package"  # "file", "function"
            )
        )

    def create_text_generation_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            self._create_node_level_objective(
                objective_name="GraphTextGeneration",
                objective_class=GraphTextGeneration,
                dataset=dataset,
                labels_fn=dataset.load_docstring,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="package"  # "file", "function"
            )
        )

    def create_node_classifier_objective(self, dataset, tokenizer_path):
        from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import ClassifierTargetMapper

        self.objectives.append(
            self._create_node_level_objective(
                objective_name="NodeTypeClassifier",
                objective_class=NodeClassifierObjective,
                label_loader_class=ClassifierTargetMapper,
                dataset=dataset,
                labels_fn=dataset.load_node_classes,
                tokenizer_path=tokenizer_path,
                masker_fn=dataset.create_node_clf_masker,
                preload_for="package"
            )
        )

    def create_node_embedder(self, dataset, tokenizer_path, n_dims=None, pretrained_path=None, n_buckets=500000):
        if pretrained_path is not None:
            logging.info("Loading pre-trained node embeddings is no longer supported. Initializing new embedding table.")

        logging.info(f"Node embedding size is {n_dims}")

        self.node_embedder = SimplestNodeEmbedder(
            emb_size=n_dims,
            dtype=self.dtype,
            n_buckets=n_buckets
        ).to(self.device)

    @property
    def lr(self):
        return self.trainer_params['learning_rate']

    @property
    def batch_size(self):
        return self.trainer_params['batch_size']

    @property
    def sampling_neighbourhood_size(self):
        return self.trainer_params['sampling_neighbourhood_size']

    @property
    def neg_sampling_factor(self):
        return self.trainer_params['neg_sampling_factor']

    @property
    def epochs(self):
        return self.trainer_params['epochs']

    @property
    def elem_emb_size(self):
        return self.trainer_params['elem_emb_size']

    @property
    def node_name_file(self):
        return self.trainer_params['node_name_file']

    @property
    def var_use_file(self):
        return self.trainer_params['var_use_file']

    @property
    def call_seq_file(self):
        return self.trainer_params['call_seq_file']

    @property
    def model_base_path(self):
        return self.trainer_params['model_base_path']

    @property
    def finetune(self):
        if self.trainer_params['pretraining_phase'] == -1:
            return False
        return self.epoch >= self.trainer_params['pretraining_phase']

    @property
    def subgraph_id_column(self):
        return self.trainer_params["subgraph_id_column"]

    @property
    def do_save(self):
        return self.trainer_params['save_checkpoints']

    @staticmethod
    def add_to_summary(summary, partition, objective_name, scores, postfix):
        summary.update({
            f"{key}/{partition}/{objective_name}_{postfix}": val for key, val in scores.items()
        })

    def write_summary(self, scores, batch_step):
        # main_name = os.path.basename(self.model_base_path)
        for var, val in scores.items():
            # self.summary_writer.add_scalar(f"{main_name}/{var}", val, batch_step)
            self.summary_writer.add_scalar(var, val, batch_step)
        # self.summary_writer.add_scalars(main_name, scores, batch_step)

    def write_hyperparams(self, scores, epoch):
        params = copy(self.model_params)
        params["epoch"] = epoch
        main_name = os.path.basename(self.model_base_path)
        params = {k: v for k, v in params.items() if type(v) in {int, float, str, bool, torch.Tensor}}

        main_name = os.path.basename(self.model_base_path)
        scores = {f"h_metric/{k}": v for k, v in scores.items()}
        self.summary_writer.add_hparams(params, scores, run_name=f"h_metric/{epoch}")

    def _create_optimizer(self):
        parameters = nn.ParameterList(self.graph_model.parameters())
        nodeembedder_params = list(self.node_embedder.parameters())
        # parameters.extend(self.node_embedder.parameters())
        [parameters.extend(objective.parameters()) for objective in self.objectives]
        # AdaHessian  TODO could not run
        # optimizer = Yogi(parameters, lr=self.lr)
        self.optimizer = torch.optim.AdamW(
            [{"params": parameters}], lr=self.lr, weight_decay=0.5
        )
        self.sparse_optimizer = torch.optim.SparseAdam(
            [{"params": nodeembedder_params}], lr=self.lr
        )

    # def _warm_up_proximity_ns(self, objective, update_ns_callback):
    #     if objective.update_embeddings_for_queries:
    #         def chunks(lst, n):
    #             for i in range(0, len(lst), n):
    #                 yield torch.LongTensor(lst[i:i + n])
    #
    #         all_keys = objective.target_embedder.keys()
    #         batches = chunks(all_keys, self.trainer_params["batch_size"])
    #         for batch in tqdm(
    #                 batches,
    #                 total=len(all_keys) // self.trainer_params["batch_size"] + 1,
    #                 desc="Precompute Target Embeddings", leave=True
    #         ):
    #             _ = objective.target_embedding_fn(batch, update_ns_callback)  # scorer embedding updated inside
    #
    #         self.proximity_ns_warmup_complete = True

    def _get_grad_norms(self):
        total_norm = 0.
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _reduce_metrics(self, metrics):
        return {key: sum(val) / len(val) for key, val in metrics.items()}

    def train_step_for_objective(self, step, objective, objective_iterator, longterm_metrics):
        batch = next(objective_iterator)

        loss, scores = objective.make_step(
            step, batch, "train", longterm_metrics, scorer=None
        )

        if scores is None:
            return None

        loss.backward()
        return scores

    def train_all(self):
        """
        Training procedure for the model with node classifier
        :return:
        """
        # best_val_loss = float("inf")
        # write_best_model = False

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            start = time()
            num_batches = min([objective.num_train_batches for objective in self.objectives])
            longterm_metrics = defaultdict(list)

            objective_iterators = [objective.get_iterator("train") for objective in self.objectives]

            longlongloop = range(num_batches * 4)  # make sure there are more steps than the actual number of steps

            for step in tqdm(longlongloop, total=num_batches, desc=f"Epoch {self.epoch}"):

                batch_summary = {}
                self.optimizer.zero_grad()
                self.sparse_optimizer.zero_grad()

                try:
                    for objective, objective_iterator in zip(self.objectives, objective_iterators):
                        scores = self.train_step_for_objective(step, objective, objective_iterator, longterm_metrics)
                        self.add_to_summary(
                            summary=batch_summary, partition="train", objective_name=objective.name,
                            scores=scores, postfix=""
                        )
                    # loaders = [next(objective_iterator) for objective_iterator in objective_iterators]
                except StopIteration:
                    break

                for groups in self.optimizer.param_groups:
                    for param in groups["params"]:
                        torch.nn.utils.clip_grad_norm_(param, max_norm=1.)

                # for ind, (objective, batch) in enumerate(zip(self.objectives, loaders)):
                #     loss, scores = objective.make_step(
                #         self.batch, batch, "train", longterm_metrics, scorer=None
                #     )
                #
                #     if scores is None:
                #         continue
                #
                #     loss.backward()
                #     for groups in self.optimizer.param_groups:
                #         for param in groups["params"]:
                #             torch.nn.utils.clip_grad_norm_(param, max_norm=1.)
                #
                #     self.add_to_summary(
                #         summary=batch_summary, partition="train", objective_name=objective.name,
                #         scores=scores, postfix=""
                #     )

                self.add_to_summary(
                    summary=batch_summary, partition="train", objective_name="",
                    scores={"grad_norm": self._get_grad_norms()}, postfix=""
                )

                self.optimizer.step()
                self.sparse_optimizer.step()

                self.write_summary(batch_summary, self.batch)

                self.batch += 1

            epoch_summary = self._reduce_metrics(longterm_metrics)

            epoch_summary.update(self._do_evaluation())
            self.write_summary(epoch_summary, self.batch)

            end = time()

            if self.do_save:
                self.save_checkpoint(self.model_base_path, write_best_model=False)  # write_best_model)

            # if objective.early_stopping_trigger is True:
            #     raise EarlyStopping()
            # objective.train()

            # self.write_hyperparams({k.replace("vs_batch", "vs_epoch"): v for k, v in summary_dict.items()}, self.epoch)

            print(f"Epoch: {self.epoch}, Time: {int(end - start)} s", end="\n")
            pprint(epoch_summary)

            self.lr_scheduler.step()

    def parameters(self):
        for par in self.graph_model.parameters():
            yield par

        for par in self.node_embedder.parameters():
            yield par

        for objective in self.objectives:
            for par in objective.parameters():
                yield par

    def save_checkpoint(self, checkpoint_path=None, checkpoint_name=None, write_best_model=False, **kwargs):

        model_path = join(checkpoint_path, f"saved_state.pt")

        param_dict = {
            'graph_model': self.graph_model.state_dict(),
            'node_embedder': self.node_embedder.state_dict(),
            "epoch": self.epoch,
            "batch": self.batch
        }

        for objective in self.objectives:
            param_dict[objective.name] = objective.custom_state_dict()

        if len(kwargs) > 0:
            param_dict.update(kwargs)

        torch.save(param_dict, model_path)
        if self.trainer_params["save_each_epoch"]:
            torch.save(param_dict, join(checkpoint_path, f"saved_state_{self.epoch}.pt"))

        if write_best_model:
            torch.save(param_dict,  join(checkpoint_path, f"best_model.pt"))

    def restore_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(join(checkpoint_path, "saved_state.pt"), map_location=torch.device('cpu'))
        self.graph_model.load_state_dict(checkpoint['graph_model'])
        self.node_embedder.load_state_dict(checkpoint['node_embedder'])
        for objective in self.objectives:
            objective.custom_load_state_dict(checkpoint[objective.name])
        self.epoch = checkpoint['epoch']
        self.restore_epoch = checkpoint['epoch']
        self.batch = checkpoint['batch']
        logging.info(f"Restored from epoch {checkpoint['epoch']}")
        # TODO needs test

    def _do_evaluation(self, evaluate_train_set=False):
        summary_dict = {}

        for objective in self.objectives:
            objective.eval()

        for objective in self.objectives:
            # objective.reset_iterator("train")
            # objective.reset_iterator("val")
            # objective.reset_iterator("test")
            # objective.early_stopping = False
            # self._warm_up_proximity_ns(objective)
            # objective.target_embedder.update_index()
            objective.update_embeddings_for_queries = False

        with torch.set_grad_enabled(False):

            for objective in self.objectives:
                if evaluate_train_set:
                    train_scores = objective.evaluate("train")
                    self.add_to_summary(summary_dict, "train_avg_eval", objective.name, self._reduce_metrics(train_scores),
                                        postfix="")  # "final")
                val_scores = objective.evaluate("val")
                test_scores = objective.evaluate("test")

                self.add_to_summary(summary_dict, "val_avg", objective.name, self._reduce_metrics(val_scores),
                                    postfix="")  # "final")
                self.add_to_summary(summary_dict, "test_avg", objective.name, self._reduce_metrics(test_scores),
                                    postfix="")  # "final")

        for objective in self.objectives:
            objective.train()

        return summary_dict

    def final_evaluation(self):
        summary_dict = self._do_evaluation()

        scores_str = ", ".join([f"{k}: {v}" for k, v in summary_dict.items()])

        print(f"Final eval: {scores_str}")

        return summary_dict

    def eval(self):
        self.graph_model.eval()
        self.node_embedder.eval()
        for objective in self.objectives:
            objective.eval()

    def train(self):
        self.graph_model.train()
        self.node_embedder.train()
        for objective in self.objectives:
            objective.train()

    def to(self, device):
        self.graph_model.to(device)
        self.node_embedder.to(device)
        for objective in self.objectives:
            objective.to(device)

    def inference(self):
        from SourceCodeTools.code.data.dataset.DataLoader import SGNodesDataLoader
        self.dataset.inference_mode()
        batch_size = 512  # self.trainer_params["batch_size"]
        dataloader = SGNodesDataLoader(
            dataset=self.dataset, labels_for="nodes", number_of_hops=self.model_params["n_layers"],
            batch_size=batch_size, preload_for="package", labels=None,  # self.dataset.inference_labels,
            masker_fn=None, label_loader_class=TargetLoader, label_loader_params={}, device="cpu",
            negative_sampling_strategy="w2v", embedding_table_size=self.trainer_params["embedding_table_size"]
        )

        id_maps = dict()
        embeddings = []

        for batch in tqdm(
                dataloader.get_dataloader("any"),
                total=dataloader.train_num_batches,
                desc="Computing final embeddings"
        ):
            with torch.no_grad():
                graph_emb = self.graph_model(
                    {"node_": self.node_embedder(batch["input_nodes"])},
                    batch["blocks"]
                )["node_"].to("cpu").numpy()

            for node_id, emb in zip(batch["indices"], graph_emb):
                if node_id not in id_maps:
                    id_maps[node_id] = len(id_maps)

                ind_to_set = id_maps[node_id]

                if ind_to_set == len(embeddings):
                    embeddings.append(emb)
                elif ind_to_set < len(embeddings):
                    embeddings[ind_to_set] = emb
                else:
                    raise ValueError()

        embedder = Embedder(id_maps, np.vstack(embeddings))

        return embedder

    # def get_embeddings(self):
    #     # self.graph_model.g.nodes["function"].data.keys()
    #     nodes = self.graph_model.g.nodes
    #     node_embs = {
    #         ntype: self.node_embedder(node_type=ntype, node_ids=nodes[ntype].data['typed_id'], train_embeddings=False)
    #         for ntype in self.graph_model.g.ntypes
    #     }
    #
    #     logging.info("Computing all embeddings")
    #     h = self.graph_model.inference(batch_size=2048, device='cpu', num_workers=0, x=node_embs)
    #
    #     original_id = []
    #     global_id = []
    #     embeddings = []
    #     for ntype in self.graph_model.g.ntypes:
    #         embeddings.append(h[ntype])
    #         original_id.extend(nodes[ntype].data['original_id'].tolist())
    #         global_id.extend(nodes[ntype].data['global_graph_id'].tolist())
    #
    #     embeddings = torch.cat(embeddings, dim=0).detach().numpy()
    #
    #     return [Embedder(dict(zip(original_id, global_id)), embeddings)]


def select_device(args):
    device = 'cpu'
    use_cuda = args["gpu"] >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args["gpu"])
        device = 'cuda:%d' % args["gpu"]
    return device


def resolve_activation_function(function_name):
    known_functions = {
        "tanh": torch.tanh
    }
    if function_name in known_functions:
        return known_functions[function_name]
    return eval(f"nn.functional.{function_name}")


def training_procedure(
        dataset, model_name, model_params, trainer_params, model_base_path,
        tokenizer_path=None, trainer=None, load_external_dataset=None
) -> Tuple[SamplingMultitaskTrainer, dict, Embedder]:
    model_params = copy(model_params)
    trainer_params = copy(trainer_params)

    if trainer is None:
        trainer = SamplingMultitaskTrainer

    device = select_device(trainer_params)

    trainer_params['model_base_path'] = model_base_path

    trainer = trainer(
        dataset=dataset,
        model_name=model_name,
        model_params=model_params,
        trainer_params=trainer_params,
        restore=trainer_params["restore_state"],
        device=device,
        pretrained_embeddings_path=trainer_params["pretrained"],
        tokenizer_path=tokenizer_path,
        # load_external_dataset=load_external_dataset
    )

    # try:
    trainer.train_all()
    # except KeyboardInterrupt:
    #     logging.info("Training interrupted")
    # except EarlyStopping:
    #     logging.info("Early stopping triggered")
    # except Exception as e:
    #     print("There was an exception", e)

    trainer.eval()
    scores = trainer.final_evaluation()
    trainer.to('cpu')
    embedder = trainer.inference()

    return trainer, scores, embedder


def evaluation_procedure(
        dataset, model_name, model_params, trainer_params, model_base_path,
        tokenizer_path=None, trainer=None, load_external_dataset=None
):
    model_params = copy(model_params)

    if trainer is None:
        trainer = SamplingMultitaskTrainer

    device = select_device(trainer_params)

    trainer_params['model_base_path'] = model_base_path

    trainer = trainer(
        dataset=dataset,
        model_name=model_name,
        model_params=model_params,
        trainer_params=trainer_params,
        restore=trainer_params["restore_state"],
        device=device,
        pretrained_embeddings_path=trainer_params["pretrained"],
        tokenizer_path=tokenizer_path,
        # load_external_dataset=load_external_dataset
    )

    trainer.eval()
    scores = trainer.final_evaluation()
    return scores
