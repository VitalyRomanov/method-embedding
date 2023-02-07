import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from os.path import join
import logging

from tqdm import tqdm

from SourceCodeTools.code.data.dataset.DataLoader import SGTrueEdgesDataLoader
from SourceCodeTools.models.graph.TargetLoader import TargetLoader
from SourceCodeTools.models.graph.train.objectives.GraphLinkClassificationObjective import RelationalDistMult
from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import ClassifierTargetMapper
from SourceCodeTools.models.graph.train.sampling_multitask2 import SamplingMultitaskTrainer


class RelationalNodeEmbedder(nn.Module):
    def __init__(self, num_unique_embeddings, node_type_params, shared_node_params, nodes, global_node_params=None, dtype=None):
        super(RelationalNodeEmbedder, self).__init__()

        self._id2desc = dict(zip(nodes["id"], zip(nodes["type"], nodes["name"])))
        self.nodes = nodes

        self.emb_size = node_type_params[1].shape[1]
        self.dtype = dtype
        if dtype is None:
            self.dtype = torch.float32
        self.n_buckets = num_unique_embeddings

        self.node_type_params = torch.from_numpy(node_type_params[1])
        self.node_type_order = dict(
            zip(
                zip(["type_node"] * len(node_type_params[0]), node_type_params[0]),
                range(len(global_node_params[0]))
            )
        )
        self.shared_node_params = nn.Parameter(torch.from_numpy(shared_node_params[1]))
        self.shared_node_order = dict(zip(shared_node_params[0], range(len(shared_node_params[0]))))
        if global_node_params is not None:
            self.global_node_params = nn.Parameter(torch.from_numpy(global_node_params[1]))
            self.global_node_order = dict(zip(global_node_params[0], range(len(global_node_params[0]))))
        else:
            self.global_node_params = torch.zeros(1, self.emb_size, dtype=self.dtype)
            self.global_node_order = {"emply": "empty"}
        self.params = nn.Parameter(torch.Tensor(self.n_buckets, self.emb_size))

        self.node_type_params.requires_grad = False
        self.shared_node_params.requires_grad = False
        self.global_node_params.requires_grad = False
        nn.init.xavier_uniform_(self.params)

    def _adapt_id(self, id_):
        desc = self._id2desc[id_]
        if desc in self.shared_node_order:
            return self.shared_node_order[desc] + len(self.node_type_order)
        elif desc in self.global_node_order:
            return self.global_node_order[desc] + len(self.node_type_order) + len(self.shared_node_order)
        elif desc in self.node_type_order:
            return self.node_type_order[desc]
        else:
            return int(hashlib.md5(f"{desc[0].strip()}_{desc[1].strip()}".encode('utf-8')).hexdigest()[:16], 16) % \
                   self.n_buckets + len(self.node_type_order) + len(self.shared_node_order) + len(self.global_node_order)

    def forward(self, input_nodes, mask=None):

        _input_nodes = torch.LongTensor(list(map(self._adapt_id, input_nodes.tolist())))

        all_params = torch.cat([
            self.node_type_params,
            self.shared_node_params,
            self.global_node_params,
            self.params
        ])

        return nn.functional.embedding(_input_nodes, all_params)


class RelationalFinetuneTrainer(SamplingMultitaskTrainer):
    def __init__(
            self, *args, **kwargs
    ):
        self.read_all_relational_params(Path(kwargs["dataset"].data_path))
        super().__init__(*args, **kwargs)

    def read_all_relational_params(self, checkpoint_path):
        self.edge_type_params = self.read_relational_params(checkpoint_path, "edge_types")
        self.node_type_params = self.read_relational_params(checkpoint_path, "node_types")
        self.shared_node_params = self.read_relational_params(checkpoint_path, "shared_nodes")
        self.global_node_params = self.read_relational_params(checkpoint_path, "global_nodes")

    def create_objectives(self, dataset, tokenizer_path):
        objective_list = self.trainer_params["objectives"]

        def load_labels():
            return dataset._graph_storage._edges[["id", "type"]].rename({"id": "src", "type": "dst"}, axis=1)

        self.objectives = nn.ModuleList()
        # if "DistMult" in objective_list:
        self.objectives.append(
            self._create_edge_level_objective(
                objective_name="RelationalDistMult",
                objective_class=RelationalDistMult,
                dataset=dataset,
                labels_fn=load_labels,
                label_loader_class=ClassifierTargetMapper,
                # label_loader_params={"compact_dst": False, "emb_size": None},
                dataloader_class=SGTrueEdgesDataLoader,
                tokenizer_path=tokenizer_path,
                masker_fn=None,
                preload_for="file",  # "file", "function"
                edge_type_params=self.edge_type_params
            )
        )

        if len(self.objectives) == 0:
            raise Exception("No valid objectives provided:", objective_list)

    def read_relational_params(self, path, name):
        order = []
        with open(path.joinpath(f"{name}_order"), "r") as order_source:
            for line in order_source:
                parts = line.strip().split("\t")
                if len(parts) == 1:
                    content = parts[0]
                elif len(parts) == 2:
                    content = tuple(parts)
                else:
                    raise ValueError()
                order.append(content)
        order = dict(zip(order, range(len(order))))

        params = np.load(path.joinpath(f"{name}_params.npy"))
        return order, params

    def restore_from_checkpoint(self, checkpoint_path):
        # checkpoint = torch.load(join(checkpoint_path, "saved_state.pt"), map_location=torch.device('cpu'))
        # self.graph_model.load_state_dict(checkpoint['graph_model'])
        # self.node_embedder.load_state_dict(checkpoint['node_embedder'])
        # for objective in self.objectives:
        #     objective.custom_load_state_dict(checkpoint[objective.name])
        # self.epoch = checkpoint['epoch']
        # self.restore_epoch = checkpoint['epoch']
        # self.batch = checkpoint['batch']
        # logging.info(f"Restored from epoch {checkpoint['epoch']}")
        ...

    def save_checkpoint(self, checkpoint_path=None, checkpoint_name=None, write_best_model=False, **kwargs):

        if checkpoint_name is None:
            checkpoint_name = f"saved_state.pt"

        model_path = join(checkpoint_path, checkpoint_name)

        param_dict = {
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

    def parameters(self):
        for par in self.node_embedder.parameters():
            yield par

        for objective in self.objectives:
            for par in objective.parameters():
                yield par

    def create_node_embedder(self, dataset, tokenizer_path, n_dims=None, pretrained_path=None, n_buckets=500000):
        if pretrained_path is not None:
            logging.info("Loading pre-trained node embeddings is no longer supported. Initializing new embedding table.")

        logging.info(f"Node embedding size is {n_dims}")

        self.node_embedder = RelationalNodeEmbedder(
            num_unique_embeddings=dataset.get_num_nodes(),
            node_type_params=self.node_type_params,
            shared_node_params=self.shared_node_params,
            global_node_params=self.global_node_params,
            nodes=dataset._graph_storage.get_nodes()
        ).to(self.device)

    def create_graph_model(self, model_name, model_params, trainer_params, device):
        self.graph_model = None

    def _create_optimizer(self):
        parameters = list(self.node_embedder.parameters())
        # parameters.extend(self.node_embedder.parameters())
        [parameters.extend(objective.parameters()) for objective in self.objectives]
        # AdaHessian  TODO could not run
        # optimizer = Yogi(parameters, lr=self.lr)
        self.optimizer = torch.optim.AdamW(
            [{"params": parameters}], lr=self.lr, weight_decay=0.5
        )

    def eval(self):
        self.node_embedder.eval()
        for objective in self.objectives:
            objective.eval()

    def to(self, device):
        self.node_embedder.to(device)
        for objective in self.objectives:
            objective.to(device)

    def inference(self):
        from SourceCodeTools.code.data.dataset.DataLoader import SGNodesDataLoader
        self.dataset.inference_mode()
        batch_size = 512  # self.trainer_params["batch_size"]

        # if self.trainer_params["inference_ids_path"] is not None:
        #     labels = pd.read_csv(self.trainer_params["inference_ids_path"])
        #     labels["dst"] = 0
        #     labels.rename({"id": "src"}, axis=1, inplace=True)
        # else:
        #     labels = None

        dataloader = SGNodesDataLoader(
            dataset=self.dataset, labels_for="nodes", number_of_hops=self.model_params["n_layers"],
            batch_size=batch_size, preload_for="package", labels=None,  # self.dataset.inference_labels,
            masker_fn=None, label_loader_class=TargetLoader, label_loader_params={}, device="cpu",
            negative_sampling_strategy="w2v", embedding_table_size=self.trainer_params["embedding_table_size"]
        )

        embedding_store = {}  #KVStore(join(self.model_base_path, "embeddings.kv"))

        for batch in tqdm(
                dataloader.get_dataloader("any"),
                total=dataloader.train_num_batches,
                desc="Computing final embeddings"
        ):
            with torch.no_grad():
                original_ids = batch["blocks"][-1].dstdata["original_id"]
                graph_emb = self.node_embedder(batch["blocks"][-1].dstdata["original_id"])

            for node_id, emb in zip(original_ids.tolist(), graph_emb):
                embedding_store[node_id] = np.ravel(emb)

        # embedding_store.save()
        #
        return embedding_store