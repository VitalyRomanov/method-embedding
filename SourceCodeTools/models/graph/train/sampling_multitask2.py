import os
from collections import defaultdict
from copy import copy
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

from SourceCodeTools.models.Embedder import Embedder
from SourceCodeTools.models.graph.train.objectives import VariableNameUsePrediction, TokenNamePrediction, \
    NextCallPrediction, NodeNamePrediction, GlobalLinkPrediction, GraphTextPrediction, GraphTextGeneration, \
    NodeNameClassifier, EdgePrediction, TypeAnnPrediction, EdgePrediction2, NodeClassifierObjective
from SourceCodeTools.models.graph.NodeEmbedder import NodeEmbedder
from SourceCodeTools.models.graph.train.objectives.GraphLinkClassificationObjective import TransRObjective


class EarlyStopping(Exception):
    def __init__(self, *args, **kwargs):
        super(EarlyStopping, self).__init__(*args, **kwargs)


def add_to_summary(summary, partition, objective_name, scores, postfix):
    summary.update({
        f"{key}/{partition}/{objective_name}_{postfix}": val for key, val in scores.items()
    })


class SamplingMultitaskTrainer:

    def __init__(self,
                 dataset=None, model_name=None, model_params=None,
                 trainer_params=None, restore=None, device=None,
                 pretrained_embeddings_path=None,
                 tokenizer_path=None
                 ):

        self.graph_model = model_name(dataset.g, **model_params).to(device)
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.device = device
        self.epoch = 0
        self.restore_epoch = 0
        self.batch = 0
        self.dtype = torch.float32
        self.create_node_embedder(
            dataset, tokenizer_path, n_dims=model_params["h_dim"],
            pretrained_path=pretrained_embeddings_path, n_buckets=trainer_params["embedding_table_size"]
        )
        if hasattr(dataset, "heldout"):
            self.holdout = dataset.heldout
            self.edges = dataset.edges

        self.summary_writer = SummaryWriter(self.model_base_path)

        self.create_objectives(dataset, tokenizer_path)

        if restore:
            self.restore_from_checkpoint(self.model_base_path)

        self._create_optimizer()

        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=1.0)
        # self.lr_scheduler = ReduceLROnPlateau(self.optimizer, patience=10, cooldown=20)

    def create_objectives(self, dataset, tokenizer_path):
        self.objectives = nn.ModuleList()
        if "token_pred" in self.trainer_params["objectives"]:
            self.create_token_pred_objective(dataset, tokenizer_path)
        if "node_name_pred" in self.trainer_params["objectives"]:
            self.create_node_name_objective(dataset, tokenizer_path)
        if "var_use_pred" in self.trainer_params["objectives"]:
            self.create_var_use_objective(dataset, tokenizer_path)
        if "next_call_pred" in self.trainer_params["objectives"]:
            self.create_api_call_objective(dataset, tokenizer_path)
        if "global_link_pred" in self.trainer_params["objectives"]:
            self.create_global_link_objective(dataset, tokenizer_path)
        if "edge_pred" in self.trainer_params["objectives"]:
            self.create_edge_objective(dataset, tokenizer_path)
        if "transr" in self.trainer_params["objectives"]:
            self.create_transr_objective(dataset, tokenizer_path)
        if "doc_pred" in self.trainer_params["objectives"]:
            self.create_text_prediction_objective(dataset, tokenizer_path)
        if "doc_gen" in self.trainer_params["objectives"]:
            self.create_text_generation_objective(dataset, tokenizer_path)
        if "node_clf" in self.trainer_params["objectives"]:
            self.create_node_classifier_objective(dataset, tokenizer_path)
        if "type_ann_pred" in self.trainer_params["objectives"]:
            self.create_type_ann_objective(dataset, tokenizer_path)

    def create_token_pred_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            TokenNamePrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_token_prediction, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type="inner_prod",
                masker=dataset.create_subword_masker(), measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"]
            )
        )

    def create_node_name_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            # GraphTextGeneration(
            #     self.graph_model, self.node_embedder, dataset,
            #     dataset.load_node_names, self.device,
            #     self.sampling_neighbourhood_size, self.batch_size,
            #     tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size,
            # )
            NodeNamePrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_node_names, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type="inner_prod",
                masker=dataset.create_node_name_masker(tokenizer_path),
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"], nn_index=self.trainer_params["nn_index"]
            )
        )

    def create_type_ann_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            TypeAnnPrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_type_prediction, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type="inner_prod",
                masker=None,
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"]
            )
        )

    def create_node_name_classifier_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            NodeNameClassifier(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_node_names, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size,
                masker=dataset.create_node_name_masker(tokenizer_path),
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"]
            )
        )

    def create_var_use_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            VariableNameUsePrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_var_use, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type="inner_prod",
                masker=dataset.create_variable_name_masker(tokenizer_path),
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"]
            )
        )

    def create_api_call_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            NextCallPrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_api_call, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type="inner_prod",
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"]
            )
        )

    def create_global_link_objective(self, dataset, tokenizer_path):
        assert dataset.no_global_edges is True

        self.objectives.append(
            GlobalLinkPrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_global_edges_prediction, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type="nn",
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"]
            )
        )

    def create_edge_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            EdgePrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_edge_prediction, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type=self.trainer_params["metric"],
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"], nn_index=self.trainer_params["nn_index"],
                # ns_groups=dataset.get_negative_sample_groups()
            )
        )

    def create_transr_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            TransRObjective(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_edge_prediction, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size,
                link_predictor_type=self.trainer_params["metric"],
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"],
                # ns_groups=dataset.get_negative_sample_groups()
            )
        )

    def create_text_prediction_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            GraphTextPrediction(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_docstring, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size, link_predictor_type="inner_prod",
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"]
            )
        )

    def create_text_generation_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            GraphTextGeneration(
                self.graph_model, self.node_embedder, dataset,
                dataset.load_docstring, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size,
            )
        )

    def create_node_classifier_objective(self, dataset, tokenizer_path):
        self.objectives.append(
            NodeClassifierObjective(
                "NodeTypeClassifier",
                self.graph_model, self.node_embedder, dataset,
                dataset.load_node_classes, self.device,
                self.sampling_neighbourhood_size, self.batch_size,
                tokenizer_path=tokenizer_path, target_emb_size=self.elem_emb_size,
                masker=dataset.create_node_clf_masker(),
                measure_scores=self.trainer_params["measure_scores"],
                dilate_scores=self.trainer_params["dilate_scores"],
                early_stopping=self.trainer_params["early_stopping"],
                early_stopping_tolerance=self.trainer_params["early_stopping_tolerance"]
            )
        )

    def create_node_embedder(self, dataset, tokenizer_path, n_dims=None, pretrained_path=None, n_buckets=500000):
        from SourceCodeTools.nlp.embed.fasttext import load_w2v_map

        if pretrained_path is not None:
            pretrained = load_w2v_map(pretrained_path)
        else:
            pretrained = None

        if pretrained_path is None and n_dims is None:
            raise ValueError(f"Specify embedding dimensionality or provide pretrained embeddings")
        elif pretrained_path is not None and n_dims is not None:
            assert n_dims == pretrained.n_dims, f"Requested embedding size and pretrained embedding " \
                                                f"size should match: {n_dims} != {pretrained.n_dims}"
        elif pretrained_path is not None and n_dims is None:
            n_dims = pretrained.n_dims

        if pretrained is not None:
            logging.info(f"Loading pretrained embeddings...")
        logging.info(f"Input embedding size is {n_dims}")

        self.node_embedder = NodeEmbedder(
            nodes=dataset.nodes,
            emb_size=n_dims,
            # tokenizer_path=tokenizer_path,
            dtype=self.dtype,
            pretrained=dataset.buckets_from_pretrained_embeddings(pretrained_path, n_buckets)
            if pretrained_path is not None else None,
            n_buckets=n_buckets
        )

    @property
    def lr(self):
        return self.trainer_params['lr']

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
    def do_save(self):
        return self.trainer_params['save_checkpoints']

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

    def compute_embeddings_for_scorer(self, objective):
        if hasattr(objective.target_embedder, "scorer_all_keys") and objective.update_embeddings_for_queries:
            def chunks(lst, n):
                for i in range(0, len(lst), n):
                    yield torch.LongTensor(lst[i:i + n])

            batches = chunks(objective.target_embedder.scorer_all_keys, self.trainer_params["batch_size"])
            for batch in tqdm(
                    batches,
                    total=len(objective.target_embedder.scorer_all_keys) // self.trainer_params["batch_size"] + 1,
                    desc="Precompute Target Embeddings", leave=True
            ):
                _ = objective.target_embedding_fn(batch)  # scorer embedding updated inside

    def train_all(self):
        """
        Training procedure for the model with node classifier
        :return:
        """

        summary_dict = {}
        best_val_loss = float("inf")
        write_best_model = False

        for objective in self.objectives:
            with torch.set_grad_enabled(False):
                self.compute_embeddings_for_scorer(objective)
                objective.target_embedder.prepare_index()  # need this to update sampler for the next epoch

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            start = time()

            summary_dict = {}
            num_batches = min([objective.num_train_batches for objective in self.objectives])

            train_losses = defaultdict(list)
            train_accs = defaultdict(list)

            # def append_metric(destination, name, metric):
            #     if name not in destination:
            #         destination[name] = []
            #     destination[name].append(metric)

            for step in tqdm(range(num_batches), total=num_batches, desc=f"Epoch {self.epoch}"):

                loss_accum = 0

                summary = {}

                try:
                    loaders = [objective.loader_next("train") for objective in self.objectives]
                except StopIteration:
                    break

                self.optimizer.zero_grad()
                self.sparse_optimizer.zero_grad()
                for ind, (objective, (input_nodes, seeds, blocks)) in enumerate(zip(self.objectives, loaders)):
                    blocks = [blk.to(self.device) for blk in blocks]

                    objective.target_embedder.prepare_index()

                    # do_break = False
                    # for block in blocks:
                    #     if block.num_edges() == 0:
                    #         do_break = True
                    # if do_break:
                    #     break

                    # try:
                    loss, acc = objective(
                        input_nodes, seeds, blocks, train_embeddings=self.finetune,
                        neg_sampling_strategy="w2v" if self.trainer_params["force_w2v_ns"] else None
                    )

                    loss = loss / len(self.objectives)  # assumes the same batch size for all objectives
                    loss_accum += loss.item()
                    # for groups in self.optimizer.param_groups:
                    #     for param in groups["params"]:
                    #         torch.nn.utils.clip_grad_norm_(param, max_norm=1.)
                    loss.backward()  # create_graph = True
                    
                    summary = {}
                    add_to_summary(
                        summary=summary, partition="train", objective_name=objective.name,
                        scores={"Loss": loss.item(), "Accuracy": acc}, postfix=""
                    )

                    train_losses[f"Loss/train_avg/{objective.name}"].append(loss.item())
                    train_accs[f"Accuracy/train_avg/{objective.name}"].append(acc)

                    # except ZeroEdges as e:
                    #     logging.warning(f"Zero edges in loader in step {step}")
                    # except Exception as e:
                    #     raise e

                self.optimizer.step()
                self.sparse_optimizer.step()
                step += 1

                self.write_summary(summary, self.batch)
                summary_dict.update(summary)

                self.batch += 1
                # summary = {
                #     f"Loss/train": loss_accum,
                # }
                summary = {key: sum(val) / len(val) for key, val in train_losses.items()}
                summary.update({key: sum(val) / len(val) for key, val in train_accs.items()})
                self.write_summary(summary, self.batch)
                summary_dict.update(summary)

            for objective in self.objectives:
                objective.reset_iterator("train")

            for objective in self.objectives:
                objective.eval()

                with torch.set_grad_enabled(False):
                    objective.target_embedder.prepare_index()  # need this to update sampler for the next epoch

                    val_scores = objective.evaluate("val")
                    test_scores = objective.evaluate("test")
                    
                summary = {}
                add_to_summary(summary, "val", objective.name, val_scores, postfix="")
                add_to_summary(summary, "test", objective.name, test_scores, postfix="")

                self.write_summary(summary, self.batch)
                summary_dict.update(summary)

                val_losses = [item for key, item in summary_dict.items() if key.startswith("Loss/val")]
                avg_val_loss = sum(val_losses) / len(val_losses)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    write_best_model = True

                if self.do_save:
                    self.save_checkpoint(self.model_base_path, write_best_model=write_best_model)
                write_best_model = False

                if objective.early_stopping_trigger is True:
                    raise EarlyStopping()

                objective.train()

            # self.write_hyperparams({k.replace("vs_batch", "vs_epoch"): v for k, v in summary_dict.items()}, self.epoch)

            end = time()

            print(f"Epoch: {self.epoch}, Time: {int(end - start)} s", end="\t")
            print(summary_dict)

            self.lr_scheduler.step()

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

    def final_evaluation(self):

        summary_dict = {}

        for objective in self.objectives:
            objective.reset_iterator("train")
            objective.reset_iterator("val")
            objective.reset_iterator("test")
            # objective.early_stopping = False
            self.compute_embeddings_for_scorer(objective)
            objective.target_embedder.prepare_index()
            objective.update_embeddings_for_queries = False

        with torch.set_grad_enabled(False):

            for objective in self.objectives:

                # train_scores = objective.evaluate("train")
                val_scores = objective.evaluate("val")
                test_scores = objective.evaluate("test")
                if hasattr(self, "holdout"):
                    objective.set_edges(self.edges)
                    holdout_scores = objective.evaluate_holdout(self.holdout)
                
                summary = {}
                # add_to_summary(summary, "train", objective.name, train_scores, postfix="final")
                add_to_summary(summary, "val", objective.name, val_scores, postfix="final")
                add_to_summary(summary, "test", objective.name, test_scores, postfix="final")
                if hasattr(self, "holdout"):
                    add_to_summary(summary, "holdout", objective.name, holdout_scores, postfix="final")
                
                summary_dict.update(summary)

        # self.write_hyperparams(summary_dict, self.epoch)

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

    def get_embeddings(self):
        # self.graph_model.g.nodes["function"].data.keys()
        nodes = self.graph_model.g.nodes
        node_embs = {
            ntype: self.node_embedder(node_type=ntype, node_ids=nodes[ntype].data['typed_id'], train_embeddings=False)
            for ntype in self.graph_model.g.ntypes
        }

        logging.info("Computing all embeddings")
        h = self.graph_model.inference(batch_size=2048, device='cpu', num_workers=0, x=node_embs)

        original_id = []
        global_id = []
        embeddings = []
        for ntype in self.graph_model.g.ntypes:
            embeddings.append(h[ntype])
            original_id.extend(nodes[ntype].data['original_id'].tolist())
            global_id.extend(nodes[ntype].data['global_graph_id'].tolist())

        embeddings = torch.cat(embeddings, dim=0).detach().numpy()

        return [Embedder(dict(zip(original_id, global_id)), embeddings)]


def select_device(args):
    device = 'cpu'
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = 'cuda:%d' % args.gpu
    return device


def training_procedure(
        dataset, model_name, model_params, args, model_base_path, trainer=None
) -> Tuple[SamplingMultitaskTrainer, dict]:

    if trainer is None:
        trainer = SamplingMultitaskTrainer

    device = select_device(args)

    model_params['num_classes'] = args.node_emb_size
    model_params['use_gcn_checkpoint'] = args.use_gcn_checkpoint
    model_params['use_att_checkpoint'] = args.use_att_checkpoint
    model_params['use_gru_checkpoint'] = args.use_gru_checkpoint

    if len(args.objectives.split(",")) > 1 and args.early_stopping is True:
        print("Early stopping disabled when several objectives are used")
        args.early_stopping = False

    trainer_params = {
        'lr': model_params.pop('lr'),
        'batch_size': args.batch_size,
        'sampling_neighbourhood_size': args.num_per_neigh,
        'neg_sampling_factor': args.neg_sampling_factor,
        'epochs': args.epochs,
        'elem_emb_size': args.elem_emb_size,
        'model_base_path': model_base_path,
        'pretraining_phase': args.pretraining_phase,
        'use_layer_scheduling': args.use_layer_scheduling,
        'schedule_layers_every': args.schedule_layers_every,
        'embedding_table_size': args.embedding_table_size,
        'save_checkpoints': args.save_checkpoints,
        'measure_scores': args.measure_scores,
        'dilate_scores': args.dilate_scores,
        "objectives": args.objectives.split(","),
        "early_stopping": args.early_stopping,
        "early_stopping_tolerance": args.early_stopping_tolerance,
        "force_w2v_ns": args.force_w2v_ns,
        "metric": args.metric,
        "nn_index": args.nn_index,
        "save_each_epoch": args.save_each_epoch
    }

    trainer = trainer(
        dataset=dataset,
        model_name=model_name,
        model_params=model_params,
        trainer_params=trainer_params,
        restore=args.restore_state,
        device=device,
        pretrained_embeddings_path=args.pretrained,
        tokenizer_path=args.tokenizer
    )

    try:
        trainer.train_all()
    except KeyboardInterrupt:
        logging.info("Training interrupted")
    except EarlyStopping:
        logging.info("Early stopping triggered")
    except Exception as e:
        raise e

    trainer.eval()
    scores = trainer.final_evaluation()

    trainer.to('cpu')

    return trainer, scores


def evaluation_procedure(
        dataset, model_name, model_params, args, model_base_path, trainer=None
):

    if trainer is None:
        trainer = SamplingMultitaskTrainer

    device = select_device(args)

    model_params['num_classes'] = args.node_emb_size
    model_params['use_gcn_checkpoint'] = args.use_gcn_checkpoint
    model_params['use_att_checkpoint'] = args.use_att_checkpoint
    model_params['use_gru_checkpoint'] = args.use_gru_checkpoint

    model_params["activation"] = eval(f"nn.functional.{model_params['activation']}")

    trainer_params = {
        'lr': model_params.pop("lr"),
        'batch_size': args.batch_size,
        'sampling_neighbourhood_size': args.num_per_neigh,
        'neg_sampling_factor': args.neg_sampling_factor,
        'epochs': args.epochs,
        'elem_emb_size': args.elem_emb_size,
        'model_base_path': model_base_path,
        'pretraining_phase': args.pretraining_phase,
        'use_layer_scheduling': args.use_layer_scheduling,
        'schedule_layers_every': args.schedule_layers_every,
        'embedding_table_size': args.embedding_table_size,
        'save_checkpoints': args.save_checkpoints,
        'measure_scores': args.measure_scores,
        'dilate_scores': args.dilate_scores
    }

    trainer = trainer( #SamplingMultitaskTrainer(
        dataset=dataset,
        model_name=model_name,
        model_params=model_params,
        trainer_params=trainer_params,
        restore=args.restore_state,
        device=device,
        pretrained_embeddings_path=args.pretrained,
        tokenizer_path=args.tokenizer
    )

    trainer.eval()
    scores = trainer.final_evaluation()
