import logging
import random
import tempfile
from abc import abstractmethod, ABC
from collections import defaultdict
from dgl.multiprocessing import Process, Queue
from copy import copy
from enum import Enum
from os.path import join
from queue import Empty
from time import sleep
from typing import Union, Set

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.dataloading import MultiLayerFullNeighborSampler  #, NodeDataLoader, EdgeDataLoader
import diskcache as dc
from tqdm import tqdm

from SourceCodeTools.code.data.GraphStorage import OnDiskGraphStorageWithFastIterationNoPandas
from SourceCodeTools.code.data.dataset.Dataset import SGPartitionStrategies
from SourceCodeTools.code.data.dataset.deprecated.Dataset2 import SourceGraphDatasetNoPandas
from SourceCodeTools.code.data.dataset.NewProcessNodeDataLoader import NewProcessNodeDataLoader
from SourceCodeTools.code.data.dataset.partition_strategies import SGLabelSpec
from SourceCodeTools.models.graph.TargetLoader import LabelDenseEncoder, TargetEmbeddingProximity


class SGAbstractDataLoader(ABC):
    partitions = ["train", "val", "test", "any"]
    train_loader = None
    val_loader = None
    test_loader = None
    any_loader = None

    train_num_batches = None
    val_num_batches = None
    test_num_batches = None
    any_num_batches = None
    workers = None
    num_workers=1

    def __init__(
            self, dataset_cls, dataset_spec, labels_for, number_of_hops, batch_size, preload_for="package", labels=None,
            masker_fn=None, label_loader_class=None, label_loader_params=None, device="cpu",
            negative_sampling_strategy="w2v", neg_sampling_factor=1, base_path=None, objective_name=None,
            embedding_table_size=300000, worker_id=None
    ):
        self._initialize(
            dataset_cls, dataset_spec, labels_for, number_of_hops, batch_size, preload_for=preload_for, labels=labels,
            masker_fn=masker_fn, label_loader_class=label_loader_class, label_loader_params=label_loader_params,
            device=device, negative_sampling_strategy=negative_sampling_strategy,
            neg_sampling_factor=neg_sampling_factor, base_path=base_path, objective_name=objective_name,
            embedding_table_size=embedding_table_size
        )

    def _initialize(
            self, dataset_cls, dataset_spec, labels_for, number_of_hops, batch_size, preload_for="package", labels=None,
            masker_fn=None, label_loader_class=None, label_loader_params=None, device="cpu",
            negative_sampling_strategy="w2v", neg_sampling_factor=1, base_path=None, objective_name=None,
            embedding_table_size=300000, worker_id=None
    ):
        preload_for = SGPartitionStrategies[preload_for]
        labels_for = SGLabelSpec[labels_for]

        if labels_for == SGLabelSpec.subgraphs:
            assert preload_for == SGPartitionStrategies.file, "Subgraphs objectives are currently " \
                                                              "partitioned only in files"
        self.dataset = dataset_cls
        self.dataset_spec = dataset_spec
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
        self._worker_id = worker_id

        self.train_num_batches = 0
        self.val_num_batches = 0
        self.test_num_batches = 0

        if labels is not None:
            logging.info("Encoding labels")
            self.label_encoder = LabelDenseEncoder(labels)
            if "emb_size" in label_loader_params and label_loader_params["emb_size"] is not None:  # if present, need to create an index for distance queries
                logging.info("Creating Proximity Target Embedder")
                self.target_embedding_proximity = TargetEmbeddingProximity(
                    self.label_encoder.encoded_labels(), label_loader_params["emb_size"]
                )
            else:
                self.target_embedding_proximity = None

        for partition_label in self.partitions:  # can memory consumption be improved?
            self._create_label_loader_for_partition(
                labels, partition_label, label_loader_class, label_loader_params, base_path, objective_name
            )

        self._active_loader = None

    def _set_num_batches(self, partition, value):
        if partition == "train":
            self.train_num_batches = value
        elif partition == "val":
            self.val_num_batches = value
        elif partition == "test":
            self.test_num_batches = value
        elif partition == "any":
            self.any_num_batches = value

    def _set_label_loader(self, partition, value):
        if partition == "train":
            self.train_loader = value
        elif partition == "val":
            self.val_loader = value
        elif partition == "test":
            self.test_loader = value
        elif partition == "any":
            self.any_loader = value

    def get_num_batches(self, partition):
        if partition == "train":
            return self.train_num_batches
        elif partition == "val":
            return self.val_num_batches
        elif partition == "test":
            return self.test_num_batches
        elif partition == "any":
            return self.any_num_batches

    @abstractmethod
    def get_label_loader(self, partition):
        if partition == "train":
            return self.train_loader
        elif partition == "val":
            return self.val_loader
        elif partition == "test":
            return self.test_loader
        elif partition == "any":
            return self.any_loader

    def _create_label_loader_for_partition(
            self, labels, partition_label, label_loader_class, label_loader_params, base_path, objective_name
    ):
        if labels is not None:
            partition_labels = self.dataset.get_labels_for_partition(
                self.dataset_spec, labels, partition_label, self.labels_for, group_by=self.preload_for
            )
            self._set_num_batches(partition_label, len(partition_labels) // self.batch_size + 1)  # setattr(self, f"{partition_label}_num_batches", len(partition_labels) // self.batch_size + 1)
            if partition_label == "train":
                label_loader_params = copy(label_loader_params)
                label_loader_params["logger_path"] = base_path
                label_loader_params["logger_name"] = objective_name
            logging.info("Creating label loader")
            label_loader = label_loader_class(
                partition_labels, self.label_encoder, target_embedding_proximity=self.target_embedding_proximity,
                **label_loader_params
            )
        else:
            label_loader = None
            self._set_num_batches(partition_label, self.dataset.get_partition_size(self.dataset_spec, partition_label) // self.batch_size + 1)  # setattr(self, f"{partition_label}_num_batches", self.dataset.get_partition_size(partition_label) // self.batch_size + 1)
        self._set_label_loader(partition_label, label_loader)  # setattr(self, f"{partition_label}_loader", label_loader)

    @staticmethod
    def _get_df_hash(table):
        return str(pd.util.hash_pandas_object(table).sum())

    @classmethod
    def dataloading_pipeline(
            cls, dataset_spec, grouping_strategy, groups, node_data, edge_data, subgraph_data, n_buckets,
            partition, labels_for, number_of_hops, batch_size, neg_sampling_factor, masker_fn, labels_loader
    ):
        from SourceCodeTools.code.data.dataset.deprecated.Dataset2 import SourceGraphDatasetNoPandas

        subgraph_iterator = SourceGraphDatasetNoPandas.iterate_subgraphs(
            dataset_spec, grouping_strategy, groups, node_data, edge_data, subgraph_data, n_buckets
        )

        def iterate_subgraphs(subgraph_iterator):
            for subgraph in subgraph_iterator:

                if masker_fn is not None:
                    # cache_key = self._get_df_hash(subgraph["nodes"]) + self._get_df_hash(subgraph["edges"])
                    # if cache_key not in self._masker_cache:
                    #     # masker = masker_fn(nodes, edges)
                    #     self._masker_cache[cache_key] = masker_fn(subgraph["nodes"], subgraph["edges"])
                    # masker = self._masker_cache[cache_key]
                    masker = masker_fn(subgraph["nodes"], subgraph["edges"])
                else:
                    masker = None

                subgraph["masker"] = masker
                subgraph["labels_loader"] = labels_loader
                # subgraph["node_label_loader"] = node_label_loader
                # subgraph["edge_label_loader"] = edge_label_loader
                # subgraph["subgraph_label_loader"] = subgraph_label_loader

                yield subgraph

        def sample_frontier(subgraph_generator):
            for subgraph_ in subgraph_generator:
                group = subgraph_["group"]
                subgraph = subgraph_["subgraph"]
                masker = subgraph_["masker"]
                labels_loader = subgraph_["labels_loader"]
                edges_bloom_filter = subgraph_["edges_bloom_filter"]

                for_batching = cls._get_ids_from_partition(subgraph, partition, labels_for)
                _num_for_batching_total = cls._num_for_batching_total(for_batching)

                for sg, ind in cls.batches_from_graph(
                    subgraph, sampler=None, ids_for_batch=for_batching, masker=masker, labels_loader=labels_loader,
                    # metagraph=metagraph, ntypes=ntypes, etypes=etypes,
                    batch_size=batch_size,
                    number_of_hops=number_of_hops,
                    # device=device,
                    neg_sampling_factor=neg_sampling_factor,
                ):
                    yield {
                        "indices": ind,
                        "group": sg,
                        "subgraph": subgraph_["subgraph"],
                        "masker": subgraph_["masker"],
                        "labels_loader": subgraph_["labels_loader"],
                        "edges_bloom_filter": subgraph_["edges_bloom_filter"],
                    }

        yield from sample_frontier(iterate_subgraphs(subgraph_iterator))

        # def sample_frontier(subgraph_generator):
        #     for subgraph_ in subgraph_generator:
        #         group = subgraph_["group"]
        #         subgraph = subgraph_["subgraph"]
        #         masker = subgraph_["masker"]
        #         labels_loader = subgraph_["labels_loader"]
        #         edges_bloom_filter = subgraph_["edges_bloom_filter"]
        #
        #         for_batching = cls._get_ids_from_partition(subgraph, partition, labels_for)
        #         _num_for_batching_total = cls._num_for_batching_total(for_batching)
        #
        #         yield from cls.batches_from_graph(
        #             subgraph, sampler=None, ids_for_batch=for_batching, masker=masker, labels_loader=labels_loader,
        #             # metagraph=metagraph, ntypes=ntypes, etypes=etypes,
        #             batch_size=batch_size,
        #             number_of_hops=number_of_hops,
        #             # device=device,
        #             neg_sampling_factor=neg_sampling_factor,
        #         )
        #
        # yield from sample_frontier(iterate_subgraphs(subgraph_iterator))

    def _iterate_subgraphs(
            self, *, node_data=None, edge_data=None, subgraph_data=None, masker_fn=None, node_label_loader=None,
            edge_label_loader=None, subgraph_label_loader=None, grouping_strategy=None, current_partition_key=None,
            labels_for=None
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

        num_per_worker = len(groups) // self.num_workers
        self.workers = []
        for i in range(0, len(groups), num_per_worker):
            worker_group = groups[i: i+num_per_worker]
            if len(worker_group) > 0:
                self.workers.append(MPIterator(
                    iter_fn=self.__class__.dataloading_pipeline, dataset_spec=self.dataset_spec,
                    grouping_strategy=grouping_strategy, groups=worker_group, node_data=node_data, edge_data=edge_data,
                    subgraph_data=subgraph_data, n_buckets=self.n_buckets, partition=current_partition_key,
                    labels_for=labels_for, number_of_hops=self.number_of_hops, batch_size=self.batch_size,
                    neg_sampling_factor=self.neg_sampling_factor, masker_fn=masker_fn, labels_loader=labels_loader
                ))

        ignore_workers = []
        workers_iterators = [iter(worker) for worker in self.workers]

        terminate = False
        while terminate is False:
            for ind, iterator in enumerate(workers_iterators):
                if ind in ignore_workers:
                    continue
                try:
                    yield next(iterator)
                except StopIteration:
                    ignore_workers.append(ind)

                if len(ignore_workers) == len(self.workers):
                    terminate = True

        # yield from self.dataloading_pipeline(
        #     self.dataset_spec, grouping_strategy, groups, node_data, edge_data, subgraph_data, self.n_buckets,
        #     current_partition_key, labels_for, self.number_of_hops, self.batch_size, self.neg_sampling_factor, masker_fn, labels_loader
        # )

        # for subgraph in self.dataset.iterate_subgraphs(
        #        self.dataset_spec, grouping_strategy, groups, node_data, edge_data, subgraph_data, self.n_buckets
        # ):
        #
        #     if masker_fn is not None:
        #         # cache_key = self._get_df_hash(subgraph["nodes"]) + self._get_df_hash(subgraph["edges"])
        #         # if cache_key not in self._masker_cache:
        #         #     # masker = masker_fn(nodes, edges)
        #         #     self._masker_cache[cache_key] = masker_fn(subgraph["nodes"], subgraph["edges"])
        #         # masker = self._masker_cache[cache_key]
        #         masker = masker_fn(subgraph["nodes"], subgraph["edges"])
        #     else:
        #         masker = None
        #
        #     subgraph["masker"] = masker
        #     subgraph["labels_loader"] = labels_loader
        #     # subgraph["node_label_loader"] = node_label_loader
        #     # subgraph["edge_label_loader"] = edge_label_loader
        #     # subgraph["subgraph_label_loader"] = subgraph_label_loader
        #
        #     yield subgraph
        #     # yield group, subgraph, masker, node_label_loader, edge_label_loader, edges_bloom_filter

    @staticmethod
    def _get_ids_from_partition_for_nodes(graph, partition, labels_for):
        nodes = {}

        for node_type in graph.ntypes:
            node_data = graph.nodes[node_type].data
            partition_mask = node_data[partition]
            with torch.no_grad():
                if partition_mask.any().item() is True:
                    # nodes_for_batching_mask = node_data["current_type_mask"] & partition_mask  # for current subgraph
                    nodes_for_batching_mask = partition_mask  # for current subgraph
                    # labels for subgraphs should be handled differently
                    if "has_label" in node_data and labels_for != SGLabelSpec.subgraphs:
                        nodes_for_batching_mask &= node_data["has_label"]
                    nodes[node_type] = graph.nodes(node_type)[nodes_for_batching_mask]
        return nodes

    @classmethod
    def _get_ids_from_partition(cls, graph, partition, labels_for):
        if labels_for == SGLabelSpec.edges:
            edges = {}
            # original_ids = {}

            for edge_type in graph.canonical_etypes:
                edge_data = graph.edges[edge_type].data
                if partition not in edge_data:
                    continue
                edges_for_batching_mask = edge_data[partition]
                with torch.no_grad():
                    if edges_for_batching_mask.any().item() is True:
                        # labels for subgraphs should be handled differently
                        if "has_label" in edge_data:
                            edges_for_batching_mask &= edge_data["has_label"]
                        edges[edge_type] = graph.edges(etype=edge_type, form="eid")[edges_for_batching_mask]
                        # original_ids[edge_type] = graph.edges[edge_type].data["original_id"][edges_for_batching_mask]
            return edges  # , original_ids
        else:
            return cls._get_ids_from_partition_for_nodes(graph, partition, labels_for)

    @staticmethod
    def _seeds_to_python(seeds):
        if isinstance(seeds, dict):
            python_seeds = {}
            for key, val in seeds.items():
                if len(val) > 0:
                    python_seeds[key] = val.tolist()
        else:
            python_seeds = seeds.tolist()
        return python_seeds

    @staticmethod
    def _num_for_batching_total(ids):
        if isinstance(ids, dict):
            total = 0
            for n in ids.values():
                total += len(n)
        else:
            total = len(ids)
        return total

    @property
    def _use_external_process_loader(self):
        # from sys import platform
        return False  # platform == "linux" or platform == "linux2"

    @classmethod
    def _create_node_batch_iterator(cls, subgraph, nodes_for_batching, sampler, batch_size):
        # if self._use_external_process_loader:
        #     if self._active_loader is not None:
        #         assert isinstance(self._active_loader, NewProcessNodeDataLoader)
        #         self._active_loader.terminate()
        #     self._active_loader = NewProcessNodeDataLoader(subgraph, nodes_for_batching, sampler, batch_size=batch_size)
        # else:
        #     self._active_loader = NodeDataLoader(
        #         subgraph, nodes_for_batching, sampler, batch_size=batch_size, shuffle=False, num_workers=0
        #     )
        # return self._active_loader
        return NodeDataLoader(
                subgraph, nodes_for_batching, sampler, batch_size=batch_size, shuffle=False, num_workers=0
            )

    def _finalize_batch_iterator(self, loader):
        pass
        # if self._use_external_process_loader:
        #     self._active_loader.terminate()
        # else:
        #     del loader
        #
        # self._active_loader = None

    @classmethod
    def batches_from_graph(
            self, graph, sampler, ids_for_batch, masker, labels_loader, batch_size, number_of_hops,
            neg_sampling_factor, device="cpu", **kwargs
    ):
        raise NotImplementedError

    @classmethod
    def create_batches(
            cls, dataset_spec, subgraph_generator, number_of_hops, batch_size, partition, labels_for,
            neg_sampling_factor, device="cpu"
    ):
        sampler = MultiLayerFullNeighborSampler(number_of_hops)
        pool = []
        in_pool = 0

        for subgraph_ in subgraph_generator:
            group = subgraph_["group"]
            subgraph = subgraph_["subgraph"]
            masker = subgraph_["masker"]
            labels_loader = subgraph_["labels_loader"]
            edges_bloom_filter = subgraph_["edges_bloom_filter"]

            # logging.info("Preparing new DGL graph for batching")

            for_batching = cls._get_ids_from_partition(subgraph, partition, labels_for)
            _num_for_batching_total = cls._num_for_batching_total(for_batching)

            if _num_for_batching_total == 0:
                continue
            else:
                pool.append(subgraph)
                in_pool += _num_for_batching_total

            if in_pool < batch_size:
                continue

            graphs = [s for s in pool]
            graph = dgl.batch(graphs)
            for_batching = cls._get_ids_from_partition(graph, partition, labels_for)
            in_pool = 0
            pool.clear()

            yield from cls.batches_from_graph(
                graph=graph,
                sampler=sampler,
                ids_for_batch=for_batching,
                masker=masker,
                labels_loader=labels_loader,
                batch_size=batch_size,
                neg_sampling_factor=neg_sampling_factor,
                number_of_hops=number_of_hops
            )

        if len(pool) > 0:
            graphs = [s for s in pool]
            graph = dgl.batch(graphs)
            for_batching = cls._get_ids_from_partition(graph, partition, labels_for)
            in_pool = 0
            pool.clear()

            yield from cls.batches_from_graph(
                graph=graph,
                sampler=sampler,
                ids_for_batch=for_batching,
                masker=masker,
                labels_loader=labels_loader,
                batch_size=batch_size,
                neg_sampling_factor=neg_sampling_factor,
                number_of_hops=number_of_hops
            )

    @abstractmethod
    def get_label_encoder(self):
        return self.label_encoder

    def get_dataloader(
            self, partition_label
    ):
        label_loader = self.get_label_loader(partition_label)  # getattr(self, f"{partition_label}_loader")

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

        for partition in ["train", "test", "val", "any"]:
            data[f"{partition}_mask"] = self.dataset.get_partition_slice(self.dataset_spec, partition)
        # get name for the partition mask
        current_partition_key = self.dataset.get_proper_partition_column_name(partition_label)

        return self.create_batches(
            self.dataset_spec,
            self._iterate_subgraphs(
                masker_fn=self.masker_fn, grouping_strategy=self.preload_for,
                current_partition_key=current_partition_key, labels_for=self.labels_for,
                **kwargs
            ),
            self.number_of_hops, self.batch_size, current_partition_key, self.labels_for,
            self.neg_sampling_factor
        )

    @abstractmethod
    def get_train_num_batches(self):
        return self.train_num_batches

    @abstractmethod
    def get_test_num_batches(self):
        return self.test_num_batches

    @abstractmethod
    def get_val_num_batches(self):
        return self.val_num_batches

    @abstractmethod
    def get_any_num_batches(self):
        return self.any_num_batches

    @abstractmethod
    def get_num_classes(self):
        return self.train_loader.num_classes

    @abstractmethod
    def partition_iterator(self, partition_label):
        class SGDLIter:
            iter_fn = self.get_dataloader

            def __init__(self):
                self.partition_label = partition_label

            @classmethod
            def __iter__(cls):
                return cls.iter_fn(partition_label)

        return SGDLIter()


class SGNodesDataLoader(SGAbstractDataLoader):
    def __init__(
            self, *args, **kwargs
    ):
        super(SGNodesDataLoader, self).__init__(*args, **kwargs)

    @classmethod
    def positive_negative(self, labels_loader, target_nodes, neg_sampling_factor, device="cpu"):
        if labels_loader is not None:
            positive_indices = torch.LongTensor(labels_loader.sample_positive(target_nodes)).to(device)
            negative_indices = torch.LongTensor(labels_loader.sample_negative(
                target_nodes, k=neg_sampling_factor, strategy="w2v",
                current_group=None
            )).to(device)
        else:
            positive_indices = None
            negative_indices = None

        return positive_indices, negative_indices

    def get_label_loader(self, partition):
        return super().get_label_loader(partition)

    def get_label_encoder(self):
        return super().get_label_encoder()

    def get_train_num_batches(self):
        return super().get_train_num_batches()

    def get_test_num_batches(self):
        return super().get_test_num_batches()

    def get_val_num_batches(self):
        return super().get_val_num_batches()

    def get_any_num_batches(self):
        return super().get_any_num_batches()

    def partition_iterator(self, partition_label):
        return super().partition_iterator(partition_label)

    def get_num_classes(self):
        return super().get_num_classes()

    @classmethod
    def batches_from_graph(
            cls, graph, sampler, ids_for_batch, masker, labels_loader, batch_size, number_of_hops,
            neg_sampling_factor, device="cpu", **kwargs
    ):
        subgraph = graph
        # loader = NewProcessNodeDataLoader(subgraph, nodes_for_batching, sampler, batch_size=batch_size)
        loader = cls._create_node_batch_iterator(subgraph, ids_for_batch, sampler, batch_size)

        # nodes_in_graph = set(subgraph.nodes("node_")[subgraph.nodes["node_"].data["current_type_mask"]].cpu().numpy())
        # nodes_in_graph = set(nodes_for_batching["node_"].numpy())

        for ind, (input_nodes, seeds, blocks) in enumerate(loader):  # 2-3gb consumed here
            if masker is not None:
                input_mask = masker.get_mask(
                    mask_for=blocks[-1].dstdata["original_id"], input_nodes=blocks[0].srcdata["original_id"]
                ).to(device)
            else:
                input_mask = None

            # indices = self.seeds_to_python(seeds)  # dgl returns torch tensor
            indices = blocks[-1].dstnodes["node_"].data["original_id"].tolist()
            input_nodes = blocks[0].srcnodes["node_"].data["embedding_id"]

            if isinstance(indices, dict):
                raise NotImplementedError("Using node types is currently not supported. Set use_node_types=False")

            positive_targets, negative_targets = cls.positive_negative(labels_loader, indices, neg_sampling_factor)

            # assert len(set(seeds.numpy()) - nodes_in_graph) == 0

            assert -1 not in input_nodes.tolist()

            batch = {
                "input_nodes": input_nodes.to(device),
                "input_mask": input_mask,
                "indices": indices,
                "blocks": [block.to(device) for block in blocks],
                "positive_indices": positive_targets,
                "negative_indices": negative_targets,
                # "labels_loader": labels_loader,
            }

            yield batch

            # for key in list(batch.keys()):
            #     if key in ["blocks"]:
            #         for ind in range(len(batch[key])):
            #             del batch[key][0]
            #     del batch[key]

        # self._finalize_batch_iterator(loader)


class SGNodesEfficientDataLoader(SGNodesDataLoader):
    # def batches_from_graph(
    #         self, graph, sampler, ids_for_batch, masker, labels_loader, **kwargs
    # ):
    #     subgraph = graph
    #     # loader = NewProcessNodeDataLoader(subgraph, nodes_for_batching, sampler, batch_size=batch_size)
    #     loader = self._create_node_batch_iterator(subgraph, ids_for_batch, sampler, self.batch_size)
    #     nxg = dgl.to_homogeneous(graph.reverse()).to_networkx()
    #
    #     edges_ = set()
    #     seeds = ids_for_batch["node_"].tolist()
    #     def extend(graph, seeds, edges, level, max_levels):
    #         if level < max_levels:
    #             for seed in seeds:
    #                 for next_ in graph[seed]:
    #                     edges.add((next_, seed))
    #                 extend(graph, graph[seed], edges, level+1, max_levels)
    #
    #     extend(nxg, seeds, edges_, 0, 5)
    #
    #     # nodes_in_graph = set(subgraph.nodes("node_")[subgraph.nodes["node_"].data["current_type_mask"]].cpu().numpy())
    #     # nodes_in_graph = set(nodes_for_batching["node_"].numpy())
    #
    #     for ind, (input_nodes, seeds, blocks) in enumerate(loader):  # 2-3gb consumed here
    #         if masker is not None:
    #             input_mask = masker.get_mask(
    #                 mask_for=blocks[-1].dstdata["original_id"], input_nodes=blocks[0].srcdata["original_id"]
    #             ).to(self.device)
    #         else:
    #             input_mask = None
    #
    #         # indices = self.seeds_to_python(seeds)  # dgl returns torch tensor
    #         indices = blocks[-1].dstnodes["node_"].data["original_id"].tolist()
    #         input_nodes = blocks[0].srcnodes["node_"].data["embedding_id"]
    #
    #         if isinstance(indices, dict):
    #             raise NotImplementedError("Using node types is currently not supported. Set use_node_types=False")
    #
    #         positive_targets, negative_targets = self.positive_negative(labels_loader, indices)
    #
    #         # assert len(set(seeds.numpy()) - nodes_in_graph) == 0
    #
    #         assert -1 not in input_nodes.tolist()
    #
    #         batch = {
    #             "input_nodes": input_nodes.to(self.device),
    #             "input_mask": input_mask,
    #             "indices": indices,
    #             "blocks": [block.to(self.device) for block in blocks],
    #             "positive_indices": positive_targets,
    #             "negative_indices": negative_targets,
    #             # "labels_loader": labels_loader,
    #         }
    #
    #         yield batch
    #
    #         # for key in list(batch.keys()):
    #         #     if key in ["blocks"]:
    #         #         for ind in range(len(batch[key])):
    #         #             del batch[key][0]
    #         #     del batch[key]
    #
    #     self._finalize_batch_iterator(loader)

    @staticmethod
    def _get_ids_from_partition_for_nodes(graph, partition, labels_for):
        nodes = []

        for node in graph.nodes:
            node_data = graph.nodes[node]
            in_partition = node_data[partition]
            if "has_label" in node_data:
                in_partition &= node_data["has_label"]
            if in_partition:
                nodes.append(node)

        return nodes

    @classmethod
    def sample_frontier(cls, graph, seeds, subgraph_nodes, level, max_levels):
        if level < max_levels:
            for seed in seeds:
                for next_ in graph[seed]:
                    subgraph_nodes.add(next_)
                cls.sample_frontier(graph, graph[seed], subgraph_nodes, level + 1, max_levels)

    @classmethod
    def positive_negative(cls, labels_loader, target_nodes, neg_sampling_factor, device="cpu"):
        if labels_loader is not None:
            positive_indices = labels_loader.sample_positive(target_nodes)
            negative_indices = labels_loader.sample_negative(
                target_nodes, k=neg_sampling_factor, strategy="w2v",
                current_group=None
            )
        else:
            positive_indices = None
            negative_indices = None

        return positive_indices, negative_indices

    @classmethod
    def batches_from_graph(
            cls, graph: nx.DiGraph, sampler, ids_for_batch, masker, labels_loader, batch_size, number_of_hops,
            neg_sampling_factor, device="cpu", **kwargs
    ):
        # metagraph = kwargs.pop("metagraph")
        # ntypes = kwargs.pop("ntypes")
        # etypes = kwargs.pop("etypes")
        reversed = graph.reverse(False)
        pool = []
        target_ids = set()

        for ind in range(0, len(ids_for_batch), batch_size):
            seeds = ids_for_batch[ind: ind + batch_size]
            subgraph_nodes = set(seeds)

            cls.sample_frontier(reversed, seeds, subgraph_nodes, 0, number_of_hops)
            s_ = graph.subgraph(subgraph_nodes).copy()

            node_fields = list(graph.nodes[0].keys())
            edge_fields = list(graph.get_edge_data(0, 1).keys())

            # subgraph = dgl.from_networkx(s_, node_attrs=node_fields, edge_attrs=edge_fields)
            # # if len(etypes) > 1:
            # subgraph = dgl.to_heterogeneous(
            #     subgraph, ntypes, etypes, ntype_field="type", etype_field="type", metagraph=metagraph
            # )
            #
            # target_mask = torch.BoolTensor([seed in seeds for seed in subgraph.ndata["nx_id"].numpy()])
            # input_nodes = subgraph.ndata["embedding_id"]
            # indices = subgraph.ndata["original_id"][target_mask]
            indices = [data["original_id"] for seed, data in s_.nodes(data=True) if data["nx_id"] in seeds]
            yield s_, indices
            # # target_ids.update(indices.tolist())
            # # pool.append(subgraph)
            #
            # # if len(target_ids) >= self.batch_size:
            # #     bgraph = dgl.batch(pool)
            # #     target_mask = torch.BoolTensor([seed in target_ids for seed in subgraph.ndata["original_id"].numpy()])
            # #     input_nodes = subgraph.ndata["embedding_id"]
            # #     indices = subgraph.ndata["original_id"][target_mask]
            # #     pool.clear()
            # #     target_ids.clear()
            #
            # if masker is not None:
            #     input_mask = masker.get_mask(
            #         mask_for=indices, input_nodes=subgraph.srcdata["original_id"]
            #     ).to(device)
            # else:
            #     input_mask = None
            #
            # positive_targets, negative_targets = cls.positive_negative(labels_loader, indices.tolist(), neg_sampling_factor, device)
            positive_targets, negative_targets = cls.positive_negative(labels_loader, indices, neg_sampling_factor)
            #
            # blocks = [subgraph] * number_of_hops
            #
            # batch = {
            #     "input_nodes": input_nodes.to(device),
            #     "input_mask": input_mask,
            #     "indices": indices,
            #     "blocks": [block.to(device) for block in blocks],
            #     "positive_indices": positive_targets,
            #     "negative_indices": negative_targets,
            #     "target_mask": target_mask.to(device),
            #     # "labels_loader": labels_loader,
            # }
            #
            # yield batch

    @classmethod
    def create_batches(
            cls, dataset_spec, subgraph_generator, number_of_hops, batch_size, partition, labels_for,
            neg_sampling_factor, device="cpu"
    ):
        metagraph = {}

        from SourceCodeTools.code.data.dataset.deprecated.Dataset2 import SourceGraphDatasetNoPandas
        ntypes, etypes = SourceGraphDatasetNoPandas.get_graph_types(dataset_spec)

        for etype in etypes:
            signature = ("node_", etype, "node_")
            metagraph[signature] = []

        metagraph = dgl.heterograph(metagraph).metagraph()

        for subgraph_ in subgraph_generator:
            group = subgraph_["group"]
            subgraph = subgraph_["subgraph"]
            masker = subgraph_["masker"]
            labels_loader = subgraph_["labels_loader"]
            edges_bloom_filter = subgraph_["edges_bloom_filter"]

            # logging.info("Preparing new DGL graph for batching")

            for_batching = cls._get_ids_from_partition(subgraph, partition, labels_for)
            _num_for_batching_total = cls._num_for_batching_total(for_batching)

            yield from cls.batches_from_graph(
                subgraph, sampler=None, ids_for_batch=for_batching, masker=masker, labels_loader=labels_loader,
                metagraph=metagraph, ntypes=ntypes, etypes=etypes, batch_size=batch_size, number_of_hops=number_of_hops,
                device=device, neg_sampling_factor=neg_sampling_factor,
            )

            # if _num_for_batching_total == 0:
            #     continue
            # else:
            #     pool.append(subgraph)
            #     in_pool += _num_for_batching_total
            #
            # if in_pool < self.batch_size:
            #     continue
            #
            # graphs = [s for s in pool]
            # graph = dgl.batch(graphs)
            # for_batching = self._get_ids_from_partition(graph, partition, labels_for)
            # in_pool = 0
            # pool.clear()
            #
            # yield from self.batches_from_graph(
            #     graph=graph,
            #     sampler=sampler,
            #     ids_for_batch=for_batching,
            #     masker=masker,
            #     labels_loader=labels_loader,
            # )


class Message:
    def __init__(self, descriptor, content):
        self.descriptor = descriptor
        self.content = content


class MPIteratorWorker:
    class InboxTypes(Enum):
        iterate = 0

    class OutboxTypes(Enum):
        worker_started = 0
        next = 1
        stop_iteration = 2

    inbox_queue: Queue
    outbox_queue: Queue
    iteration_queue: Queue

    def __init__(self, config, inbox_queue, outbox_queue, iteration_queue):
        self.iter_fn = config.pop("iter_fn")
        self.iter_fn_kwargs = config

        self.inbox_queue = inbox_queue
        self.outbox_queue = outbox_queue
        self.iteration_queue = iteration_queue
        self._send_init_confirmation()

    def _send_init_confirmation(self):
        self.send_out(Message(
            descriptor=MPIteratorWorker.OutboxTypes.worker_started,
            content=None
        ))

    def send_out(self, message, queue=None, keep_trying=True) -> True:
        """
        :param message:
        :param queue:
        :param keep_trying: Block until can put the item in the queue
        :return: Return True if could put item in the queue, and False otherwise
        """
        if queue is None:
            queue = self.outbox_queue

        if keep_trying:
            while queue.full():
                sleep(0.2)
        else:
            if queue.full():
                return False

        queue.put(message)
        return True

    def check_for_new_messages(self):
        interrupt_iteration = False
        try:
            message = self.inbox_queue.get(timeout=0.0)
            if message.descriptor == self.InboxTypes.iterate:
                while not self.outbox_queue.empty():
                    self.outbox_queue.get()
                self.inbox_queue.put(message)
                interrupt_iteration = True
            else:
                self._handle_message(message)
        except Empty:
            pass
        return interrupt_iteration

    def _iterate(self):
        iterator = self.iter_fn(**self.iter_fn_kwargs)

        for value in iterator:
            sent = False
            while not sent:
                interrupt_iteration = self.check_for_new_messages()
                if interrupt_iteration:
                    return
                sent = self.send_out(Message(
                    descriptor=self.OutboxTypes.next,
                    content=value
                ), queue=self.iteration_queue, keep_trying=False)
        self.send_out(Message(
            descriptor=self.OutboxTypes.stop_iteration,
            content=None
        ), queue=self.iteration_queue)

    def _handle_message(self, message):
        if message.descriptor == self.InboxTypes.iterate:
            self._iterate()

        else:
            raise ValueError(f"Unrecognized message descriptor: {message.descriptor.name}")

    def handle_incoming(self):
        message = self.inbox_queue.get()
        response = self._handle_message(message)


def start_worker(config, inbox_queue, outbox_queue, iteration_queue, *args, **kwargs):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")
    worker = MPIteratorWorker(config, inbox_queue, outbox_queue, iteration_queue)

    while True:
        try:
            worker.handle_incoming()
        except Exception as e:
            outbox_queue.put(e)
            raise e


class MPIterator:
    def __init__(self, **config):
        self.inbox_queue = Queue(maxsize=30)
        self.outbox_queue = Queue()
        self.iteration_queue = Queue(maxsize=30)

        self.worker_proc = Process(
            target=start_worker, args=(
                config,
                self.outbox_queue,
                self.inbox_queue,
                self.iteration_queue
            )
        )
        self.worker_proc.start()
        self.receive_init_confirmation()
        self._stop_iteration = False

    def receive_init_confirmation(self):
        self.receive_expected(MPIteratorWorker.OutboxTypes.worker_started, timeout=600)

    def receive_expected(self, expected_descriptor: Union[Enum, Set], timeout=None, queue=None):
        keep_indefinitely = False
        if timeout is None:
            keep_indefinitely = True

        if queue is None:
            queue = self.inbox_queue
        # logging.info(f"Receiving response {expected_descriptor}")

        if keep_indefinitely:
            # logging.info("Blocking until received")
            while True:
                try:
                    response: Union[Message, Exception] = queue.get(timeout=10)
                    break
                except Empty:
                    assert self.worker_proc.is_alive(), f"Worker in {self.__class__.__name__} is dead"
                    # logging.info(f"Worker in {self.__class__.__name__} is still alive")
        else:
            # logging.info(f"Waiting {timeout} seconds to receive")
            response: Union[Message, Exception] = queue.get(timeout=timeout)

        if isinstance(response, Exception):
            raise response

        if not isinstance(expected_descriptor, set):
            expected_descriptor = {expected_descriptor}
        else:
            pass

        assert response.descriptor in expected_descriptor, f"Expected {expected_descriptor}, but received {response.descriptor}"
        # logging.info(f"Received successfully")

        return response.content

    def send_request(self, request_descriptor, content=None):
        # logging.info(f"Sending request {request_descriptor}")
        self.outbox_queue.put(
            Message(
                descriptor=request_descriptor,
                content=content
            )
        )

    def send_request_and_receive_response(
            self, request_descriptor, response_descriptor, content=None
    ):
        self.send_request(request_descriptor, content)
        return self.receive_expected(response_descriptor)

    def __iter__(self):
        self.send_request(
            request_descriptor=MPIteratorWorker.InboxTypes.iterate,
            content=None
        )

        while True:
            received = self.receive_expected(
                {MPIteratorWorker.OutboxTypes.next, MPIteratorWorker.OutboxTypes.stop_iteration},
                queue=self.iteration_queue
            )
            if received is None:
                break
            yield received


def test_pipeline():
    import sys

    data_path = sys.argv[1]
    partition = sys.argv[2]
    tokenizer = sys.argv[3]

    dataset_spec = SourceGraphDatasetNoPandas.create_dataset_specification(
        data_path, partition,
        use_node_types=False,
        use_edge_types=True,
        no_global_edges=True,
        remove_reverse=False,
        # custom_reverse=["global_mention"],
        tokenizer_path=tokenizer,
        # type_nodes=True,
        # k_hops=2
        storage_class=OnDiskGraphStorageWithFastIterationNoPandas,
        storage_kwargs={"path": join(data_path, "dataset.db")}
    )
    from SourceCodeTools.models.graph.train.objectives.NodeClassificationObjective import ClassifierTargetMapper
    dataloader = SGNodesEfficientDataLoader(
        SourceGraphDatasetNoPandas, dataset_spec, labels_for="nodes", number_of_hops=3, batch_size=16, preload_for="file",
        masker_fn=None, label_loader_class=ClassifierTargetMapper,
        label_loader_params={},
        device="cpu",
        negative_sampling_strategy="w2v", neg_sampling_factor=1, base_path=None, objective_name=None,
        embedding_table_size=300000,
        labels=SourceGraphDatasetNoPandas.load_type_prediction(dataset_spec)
    )

    dl = dataloader.get_dataloader("train")

    for b in tqdm(dl):
        pass


if __name__ == "__main__":
    test_pipeline()