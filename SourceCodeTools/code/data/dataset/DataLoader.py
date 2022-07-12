import tempfile
from copy import copy

import dgl
import numpy as np
import torch
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
import diskcache as dc

from SourceCodeTools.code.data.dataset.Dataset import SGPartitionStrategies
from SourceCodeTools.code.data.dataset.NewProcessNodeDataLoader import NewProcessNodeDataLoader
from SourceCodeTools.code.data.dataset.partition_strategies import SGLabelSpec
from SourceCodeTools.models.graph.TargetLoader import LabelDenseEncoder, TargetEmbeddingProximity


class SGNodesDataLoader:
    def __init__(
            self, dataset, labels_for, number_of_hops, batch_size, preload_for="package", labels=None,
            masker_fn=None, label_loader_class=None, label_loader_params=None, device="cpu",
            negative_sampling_strategy="w2v", neg_sampling_factor=1, base_path=None, objective_name=None,
            embedding_table_size=300000
    ):
        preload_for = SGPartitionStrategies[preload_for]
        labels_for = SGLabelSpec[labels_for]

        if labels_for == SGLabelSpec.subgraphs:
            assert preload_for == SGPartitionStrategies.file, "Subgraphs objectives are currently " \
                                                              "partitioned only in files"

        self.dataset = dataset
        self.labels_for = labels_for
        self.number_of_hops = number_of_hops
        self.batch_size = batch_size
        self.preload_for = preload_for
        self.masker_fn = masker_fn
        self.device = device
        self.n_buckets = embedding_table_size
        self.neg_sampling_factor = neg_sampling_factor
        self.negative_sampling_strategy = negative_sampling_strategy
        assert negative_sampling_strategy in {"w2v", "closest"}
        self._masker_cache_path = tempfile.TemporaryDirectory(suffix="MaskerCache")
        self._masker_cache = dc.Cache(self._masker_cache_path.name)

        self.train_num_batches = 0
        self.val_num_batches = 0
        self.test_num_batches = 0

        if labels is not None:
            self.label_encoder = LabelDenseEncoder(labels)
            if "emb_size" in label_loader_params and label_loader_params["emb_size"] is not None:  # if present, need to create an index for distance queries
                self.target_embedding_proximity = TargetEmbeddingProximity(
                    self.label_encoder.encoded_labels(), label_loader_params["emb_size"]
                )
            else:
                self.target_embedding_proximity = None

        for partition_label in ["train", "val", "test", "any"]:  # can memory consumption be improved?
            self._create_label_loader_for_partition(
                labels, partition_label, label_loader_class, label_loader_params, base_path, objective_name
            )

        self._active_loader = None

    def _create_label_loader_for_partition(
            self, labels, partition_label, label_loader_class, label_loader_params, base_path, objective_name
    ):
        if labels is not None:
            partition_labels = self.dataset.get_labels_for_partition(
                labels, partition_label, self.labels_for, group_by=self.preload_for
            )
            setattr(self, f"{partition_label}_num_batches", len(partition_labels) // self.batch_size + 1)
            if partition_label == "train":
                label_loader_params = copy(label_loader_params)
                label_loader_params["logger_path"] = base_path
                label_loader_params["logger_name"] = objective_name
            label_loader = label_loader_class(
                partition_labels, self.label_encoder, target_embedding_proximity=self.target_embedding_proximity,
                **label_loader_params
            )
        else:
            label_loader = None
            partition_set = getattr(self.dataset.partition, f"_{partition_label}_set")
            setattr(self, f"{partition_label}_num_batches", len(partition_set) // self.batch_size + 1)
        setattr(self, f"{partition_label}_loader", label_loader)

    def subgraph_iterator(self):
        return self.dataset.iterate_packages()

    # def get_original_targets(self):
    #     return self.node_label_loader.get_original_targets()

    def _iterate_subgraphs(
            self, *, node_data=None, edge_data=None, subgraph_data=None, masker_fn=None, node_label_loader=None,
            edge_label_loader=None, subgraph_label_loader=None, grouping_strategy=None
    ):
        assert grouping_strategy is not None

        labels_loader = None
        if node_label_loader is not None:
            node_data["has_label"] = node_label_loader.has_label_mask()
            groups = node_label_loader.get_groups()
            labels_loader = node_label_loader
        elif edge_label_loader is not None:
            edge_data["has_label"] = edge_label_loader.has_label_mask()
            groups = edge_label_loader.get_groups()
            labels_loader = edge_label_loader
        elif subgraph_label_loader is not None:
            subgraph_data["has_label"] = subgraph_label_loader.has_label_mask()
            groups = subgraph_label_loader.get_groups()
            labels_loader = subgraph_label_loader
        else:
            groups = None

        for subgraph in self.dataset.iterate_subgraphs(
                grouping_strategy, groups, node_data, edge_data, subgraph_data, self.n_buckets
        ):

            if masker_fn is not None:
                cache_key = self.dataset._get_df_hash(subgraph["nodes"]) + self.dataset._get_df_hash(subgraph["edges"])
                if cache_key not in self._masker_cache:
                    # masker = masker_fn(nodes, edges)
                    self._masker_cache[cache_key] = masker_fn(subgraph["nodes"], subgraph["edges"])
                masker = self._masker_cache[cache_key]
            else:
                masker = None

            subgraph["masker"] = masker
            subgraph["labels_loader"] = labels_loader
            # subgraph["node_label_loader"] = node_label_loader
            # subgraph["edge_label_loader"] = edge_label_loader
            # subgraph["subgraph_label_loader"] = subgraph_label_loader

            yield subgraph
            # yield group, subgraph, masker, node_label_loader, edge_label_loader, edges_bloom_filter

    @staticmethod
    def get_nodes_from_partition(graph, partition, labels_for):
        nodes = {}

        for node_type in graph.ntypes:
            node_data = graph.nodes[node_type].data
            partition_mask = node_data[partition]
            with torch.no_grad():
                if partition_mask.any().item() is True:
                    nodes_for_batching_mask = node_data["current_type_mask"] & partition_mask  # for current subgraph
                    # labels for subgraphs should be handled differently
                    if "has_label" in node_data and labels_for != SGLabelSpec.subgraphs:
                        nodes_for_batching_mask &= node_data["has_label"]
                    nodes[node_type] = graph.nodes(node_type)[nodes_for_batching_mask]
        return nodes

    @staticmethod
    def seeds_to_python(seeds):
        if isinstance(seeds, dict):
            python_seeds = {}
            for key, val in seeds.items():
                if len(val) > 0:
                    python_seeds[key] = val.tolist()
        else:
            python_seeds = seeds.tolist()
        return python_seeds

    def _num_nodes_total(self, nodes):
        if isinstance(nodes, dict):
            total = 0
            for n in nodes.values():
                total += len(n)
        else:
            total = len(nodes)
        return total

    @property
    def _use_external_process_loader(self):
        from sys import platform
        return False  # platform == "linux" or platform == "linux2"

    def _create_batch_iterator(self, subgraph, nodes_for_batching, sampler, batch_size):
        if self._use_external_process_loader:
            if self._active_loader is not None:
                assert isinstance(self._active_loader, NewProcessNodeDataLoader)
                self._active_loader.terminate()
            self._active_loader = NewProcessNodeDataLoader(subgraph, nodes_for_batching, sampler, batch_size=batch_size)
        else:
            self._active_loader = NodeDataLoader(
                subgraph, nodes_for_batching, sampler, batch_size=batch_size, shuffle=True, num_workers=0
            )
        return self._active_loader

    def _finalize_batch_iterator(self, loader):
        if self._use_external_process_loader:
            self._active_loader.terminate()
        else:
            del loader

        self._active_loader = None

    def create_batches(self, subgraph_generator, number_of_hops, batch_size, partition, labels_for):
        for subgraph_ in subgraph_generator:
            group = subgraph_["group"]
            subgraph = subgraph_["subgraph"]
            masker = subgraph_["masker"]
            labels_loader = subgraph_["labels_loader"]

            # TODO shuffle subgraphs

            sampler = MultiLayerFullNeighborSampler(number_of_hops)
            nodes_for_batching = self.get_nodes_from_partition(subgraph, partition, labels_for)
            if self._num_nodes_total(nodes_for_batching) == 0:
                continue

            # loader = NodeDataLoader(
            #     subgraph, nodes_for_batching, sampler, batch_size=batch_size, shuffle=True, num_workers=0
            # )
            loader = self._create_batch_iterator(subgraph, nodes_for_batching, sampler, batch_size)  # loader = NewProcessNodeDataLoader(subgraph, nodes_for_batching, sampler, batch_size=batch_size)

            nodes_in_graph = set(subgraph.nodes("node_")[subgraph.nodes["node_"].data["current_type_mask"]].cpu().numpy())
            # nodes_in_graph = set(nodes_for_batching["node_"].numpy())

            for ind, (input_nodes, seeds, blocks) in enumerate(loader):  # 2-3gb consumed here
                if masker is not None:
                    input_mask = masker.get_mask(mask_for=seeds, input_nodes=input_nodes).to(self.device)
                else:
                    input_mask = None

                # indices = self.seeds_to_python(seeds)  # dgl returns torch tensor
                indices = blocks[-1].dstnodes["node_"].data["original_id"].tolist()

                if isinstance(indices, dict):
                    raise NotImplementedError("Using node types is currently not supported. Set use_node_types=False")

                if labels_loader is not None:
                    positive_indices = torch.LongTensor(labels_loader.sample_positive(indices)).to(self.device)
                    negative_indices = torch.LongTensor(labels_loader.sample_negative(
                        indices, k=self.neg_sampling_factor, strategy=self.negative_sampling_strategy,
                        current_group=group
                    )).to(self.device)
                else:
                    positive_indices = None
                    negative_indices = None

                input_nodes = blocks[0].srcnodes["node_"].data["embedding_id"]

                assert len(set(seeds.numpy()) - nodes_in_graph) == 0

                assert -1 not in input_nodes.tolist()

                batch = {
                    # "seeds": seeds,
                    "group": group,
                    "input_nodes": input_nodes.to(self.device),
                    "input_mask": input_mask,
                    "indices": indices,
                    "blocks": [block.to(self.device) for block in blocks],
                    "positive_indices": positive_indices,
                    "negative_indices": negative_indices,
                    "labels_loader": labels_loader,
                    # "edge_labels_loader": edge_labels_loader,
                    # "update_node_negative_sampler_callback": labels_loader.set_embed,
                    # "update_edge_negative_sampler_callback": None,
                }

                yield batch

                for key in list(batch.keys()):
                    if key in ["blocks"]:
                        for ind in range(len(batch[key])):
                            del batch[key][0]
                    del batch[key]
            self._finalize_batch_iterator(loader)

    def get_dataloader(
            self, partition_label
    ):
        label_loader = getattr(self, f"{partition_label}_loader")

        data = {}
        kwargs = {}

        if self.labels_for == SGLabelSpec.nodes:
            kwargs["node_label_loader"] = label_loader
            kwargs["node_data"] = data
            kwargs["edge_data"] = {}
        elif self.labels_for == SGLabelSpec.edges:
            kwargs["edge_label_loader"] = label_loader
            kwargs["node_data"] = {}
            kwargs["edge_data"] = data
        elif self.labels_for == SGLabelSpec.subgraphs:
            kwargs["subgraph_label_loader"] = label_loader
            kwargs["node_data"] = {}
            kwargs["edge_data"] = {}
            kwargs["subgraph_data"] = data
        else:
            raise ValueError()

        for partition_key in ["train_mask", "test_mask", "val_mask", "any_mask"]:
            data[partition_key] = self.dataset.partition.create_exclusive(partition_key)
        # get name for the partition mask
        current_partition_key = self.dataset.get_proper_partition_column_name(partition_label)

        return self.create_batches(
            self._iterate_subgraphs(
                masker_fn=self.masker_fn, grouping_strategy=self.preload_for, **kwargs
            ),
            self.number_of_hops, self.batch_size, current_partition_key, self.labels_for
        )

    def partition_iterator(self, partition_label):
        class SGDLIter:
            iter_fn = self.get_dataloader

            def __init__(self):
                self.partition_label = partition_label

            @classmethod
            def __iter__(cls):
                return cls.iter_fn(partition_label)

        return SGDLIter()


class SGEdgesDataLoader(SGNodesDataLoader):

    def iterate_nodes_for_batches(self, nodes):
        if isinstance(nodes, dict):
            nodes = nodes["node_"]
        total_nodes = len(nodes)
        for i in range(0, total_nodes, self.batch_size):
            yield nodes[i: min(i + self.batch_size, total_nodes)]

    @staticmethod
    def _handle_non_unique(non_unique_ids):
        id_list = non_unique_ids.tolist()
        unique_ids = sorted(list(set(id_list)))
        new_position = dict(zip(unique_ids, range(len(unique_ids))))
        slice_map = torch.tensor(list(map(lambda x: new_position[x], id_list)), dtype=torch.long)
        return torch.tensor(unique_ids, dtype=torch.long), slice_map

    def create_batches(self, subgraph_generator, number_of_hops, batch_size, partition, labels_for):

        sampler = MultiLayerFullNeighborSampler(number_of_hops)

        # for group, subgraph, masker, labels_loader, edge_labels_loader, edges_bloom_filter in subgraph_generator:
        for subgraph_ in subgraph_generator:
            group = subgraph_["group"]
            subgraph = subgraph_["subgraph"]
            masker = subgraph_["masker"]
            labels_loader = subgraph_["labels_loader"]
            edges_bloom_filter = subgraph_["edges_bloom_filter"]

            nodes_in_graph = set(subgraph.nodes("node_")[subgraph.nodes["node_"].data["current_type_mask"]].cpu().numpy())
            nodes_for_batching = self.get_nodes_from_partition(subgraph, partition, labels_for)
            if self._num_nodes_total(nodes_for_batching) == 0:
                continue

            graph_id_to_original_id = dict(zip(
                subgraph.nodes("node_").numpy(),
                subgraph.nodes["node_"].data["original_id"].numpy(),
            ))
            original_id_to_graph_id = dict(zip(
                subgraph.nodes["node_"].data["original_id"].numpy(),
                subgraph.nodes("node_").numpy()
            ))


            for nodes_in_batch_g in self.iterate_nodes_for_batches(self.seeds_to_python(nodes_for_batching)):
                nodes_in_batch = np.array(list(map(graph_id_to_original_id.get, nodes_in_batch_g)))

                if labels_loader is not None:
                    positive_indices = labels_loader.sample_positive(nodes_in_batch)
                    negative_indices = labels_loader.sample_negative(
                        nodes_in_batch, k=self.neg_sampling_factor, strategy=self.negative_sampling_strategy,
                        current_group=group, bloom_filter=edges_bloom_filter
                    )
                    positive_indices_g = torch.LongTensor(list(map(original_id_to_graph_id.get, positive_indices)))
                    negative_indices_g = torch.LongTensor(list(map(original_id_to_graph_id.get, negative_indices)))
                else:
                    positive_indices = None
                    negative_indices = None
                    positive_indices_g = None
                    negative_indices_g = None

                nodes_in_total = len(nodes_in_batch) + len(positive_indices) + len(negative_indices)
                empty_mask = [False] * nodes_in_total

                src_nodes_mask = torch.BoolTensor(empty_mask)
                positive_nodes_mask = torch.BoolTensor(empty_mask)
                negative_nodes_mask = torch.BoolTensor(empty_mask)

                positive_start = len(nodes_in_batch)
                negative_start = len(nodes_in_batch) + len(positive_indices)

                src_nodes_mask[:positive_start] = True
                positive_nodes_mask[positive_start: negative_start] = True
                negative_nodes_mask[negative_start:] = True

                all_nodes = torch.cat([torch.LongTensor(nodes_in_batch_g), positive_indices_g, negative_indices_g])

                unique_nodes, slice_map = self._handle_non_unique(all_nodes)
                assert unique_nodes[slice_map].tolist() == all_nodes.tolist()

                loader = NodeDataLoader(
                    subgraph, {"node_": unique_nodes}, sampler, batch_size=len(unique_nodes), shuffle=True, num_workers=0
                )

                input_nodes, seeds, blocks = next(iter(loader))

                if masker is not None:
                    input_mask = masker.get_mask(mask_for=unique_nodes, input_nodes=input_nodes).to(self.device)
                else:
                    input_mask = None

                # indices = self.seeds_to_python(seeds)  # dgl returns torch tensor
                #
                # if isinstance(indices, dict):
                #     raise NotImplementedError("Using node types is currently not supported. Set use_node_types=False")

                input_nodes = blocks[0].srcnodes["node_"].data["embedding_id"]

                assert len(set(seeds.numpy()) - nodes_in_graph) == 0
                assert -1 not in input_nodes.tolist()

                batch = {
                    # "seeds": seeds,  # list of unique nodes
                    "indices": nodes_in_batch,
                    "slice_map": slice_map,  # needed to restore original_nodes
                    "compute_embeddings_for": np.concatenate([nodes_in_batch, positive_indices, negative_indices]),  # all_nodes,
                    "group": group,
                    "input_nodes": input_nodes.to(self.device),
                    "input_mask": input_mask,
                    "blocks": [block.to(self.device) for block in blocks],
                    "positive_indices": positive_indices,  # .to(self.device),
                    "negative_indices": negative_indices,  # .to(self.device),
                    "labels_loader": labels_loader,
                    "src_nodes_mask": src_nodes_mask,
                    "positive_nodes_mask": positive_nodes_mask,
                    "negative_nodes_mask": negative_nodes_mask
                }

                yield batch


class SGSubgraphDataLoader(SGNodesDataLoader):

    def _prepare_batch(self, subgraph_ids, subgraphs, labels_loader, number_of_hops):
        if labels_loader is not None:
            positive_labels = torch.LongTensor(labels_loader.sample_positive(subgraph_ids)).to(self.device)
        else:
            positive_labels = None

        subgraph_nodes = []
        for subg in subgraphs:
            subg.nodes["node_"].data["original_id"] = subg.nodes("node_")
            subgraph_nodes.append(set(subg.nodes("node_")[subg.nodes["node_"].data["current_type_mask"]].tolist()))

        batched_subgraphs = dgl.batch(subgraphs)
        input_nodes = batched_subgraphs.srcnodes["node_"].data["embedding_id"]

        # sampler = MultiLayerFullNeighborSampler(number_of_hops)
        # all_nodes = batched_subgraphs.nodes("node_")[batched_subgraphs.nodes["node_"].data["current_type_mask"]]
        # loader = NewProcessNodeDataLoader(
        #     batched_subgraphs, {"node_": all_nodes}, sampler, batch_size=len(all_nodes)
        # )
        #
        # input_nodes, seeds, blocks = next(iter(loader))
        # input_nodes = blocks[0].srcnodes["node_"].data["embedding_id"]
        #
        # assert -1 not in input_nodes[blocks[0].srcnodes["node_"].data["current_type_mask"]].tolist()
        #
        # original_nodes = blocks[-1].dstnodes["node_"].data["original_id"].tolist()
        # subgraph_masks = [
        #     torch.BoolTensor([node_id in subg_nodes for node_id in original_nodes]).to(self.device)
        #     for subg_nodes in subgraph_nodes
        # ]

        batch = {
            # input_nodes, input_mask, blocks, positive_indices, negative_indices
            "indices": subgraph_ids,
            "input_nodes": input_nodes.to(self.device),
            "input_mask": None,
            "blocks": batched_subgraphs,
            "positive_indices": positive_labels,
            "negative_indices": None,
            # "subgraph_masks": subgraph_masks,
            # "blocks": [block.to(self.device) for block in blocks],
            # "labels": positive_labels,
            "labels_loader": labels_loader
        }

        return batch


    def create_batches(self, subgraph_generator, number_of_hops, batch_size, partition, labels_for):



        batch_subgrap_id = []
        batch_subgraphs = []

        labels_loader = None

        def in_partition(subgraph_id):
            in_p = subgraph_data[partition][subgraph_id]
            if "has_label" in subgraph_data:
                in_p &= subgraph_id in subgraph_data["has_label"] and subgraph_data["has_label"][subgraph_id]
            return in_p

        for subgraph_ in subgraph_generator:
            group = subgraph_["group"][0]
            subgraph = subgraph_["subgraph"]
            masker = subgraph_["masker"]
            labels_loader = subgraph_["label_loader"]
            subgraph_data = subgraph_["subgraph_data"]

            if not in_partition(group):
                continue

            batch_subgrap_id.append(group)
            batch_subgraphs.append(subgraph)

            if len(batch_subgrap_id) == batch_size:
                yield self._prepare_batch(batch_subgrap_id, batch_subgraphs, labels_loader, number_of_hops)
                batch_subgrap_id.clear()
                batch_subgraphs.clear()

        if len(batch_subgrap_id) > 0:
            yield self._prepare_batch(batch_subgrap_id, batch_subgraphs, labels_loader, number_of_hops)
            batch_subgrap_id.clear()
            batch_subgraphs.clear()