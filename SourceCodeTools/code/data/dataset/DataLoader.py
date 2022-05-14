import torch
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from tqdm import tqdm

from SourceCodeTools.code.data.dataset.Dataset import SGPartitionStrategies
from SourceCodeTools.code.data.dataset.partition_strategies import SGLabelSpec


class SGNodesDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def subgraph_iterator(self):
        return self.dataset.iterate_packages()

    def _iterate_subgraphs(
            self, node_data, edge_data, masker_fn=None, node_label_loader=None, edge_label_loader=None,
            grouping_strategy=None
    ):
        assert grouping_strategy is not None

        if node_label_loader is not None:
            node_data["has_label"] = node_label_loader.has_label_mask()

        if edge_label_loader is not None:
            edge_data["has_label"] = edge_label_loader.has_label_mask()

        for group, nodes, edges, subgraph in self.dataset.iterate_subgraphs(
                grouping_strategy, node_label_loader.get_groups(), node_data, edge_data
        ):

            if masker_fn is not None:
                masker = masker_fn(nodes, edges)
            else:
                masker = None

            yield group, subgraph, masker, node_label_loader, edge_label_loader

    @staticmethod
    def get_nodes_from_partition(graph, partition, labels_for):
        nodes = {}

        for node_type in graph.ntypes:
            node_data = graph.nodes[node_type].data
            partition_mask = node_data[partition]
            with torch.no_grad():
                if partition_mask.any().item() is True:
                    nodes_for_batching_mask = node_data["valid"] & partition_mask
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
                if val.size > 0:
                    python_seeds[key] = val.tolist()
        else:
            python_seeds = seeds.tolist()
        return python_seeds

    def create_batches(self, subgraph_generator, number_of_hops, batch_size, partition, labels_for):
        for group, subgraph, masker, node_labels_loader, edge_labels_loader in tqdm(subgraph_generator):

            # TODO shuffle subgraphs

            sampler = MultiLayerFullNeighborSampler(number_of_hops)
            nodes_for_batching = self.get_nodes_from_partition(subgraph, partition, labels_for)
            loader = NodeDataLoader(
                subgraph, nodes_for_batching, sampler, batch_size=batch_size, shuffle=True, num_workers=0
            )

            with torch.no_grad():
                for ind, (input_nodes, seeds, blocks) in enumerate(loader):
                    if masker is not None:
                        input_mask = masker.get_mask(mask_for=seeds, input_nodes=input_nodes)
                    else:
                        input_mask = None

                    indices = self.seeds_to_python(seeds)  # dgl returns torch tensor

                    if isinstance(indices, dict):
                        raise NotImplementedError("Using node types is currently not supported. Set use_node_types=False")

                    positive_indices = node_labels_loader.sample_positive(indices)
                    negative_indices = node_labels_loader.sample_negative_w2v(indices)

                    batch = {
                        "group": group,
                        "input_nodes": input_nodes,
                        "input_mask": input_mask,
                        "indices": indices,
                        "positive_indices": positive_indices,
                        "negative_indices": negative_indices,
                        "node_labels_loader": node_labels_loader,
                        "edge_labels_loader": edge_labels_loader
                        # "update_node_negative_sampler_callback": node_labels_loader.set_embed,
                        # "update_edge_negative_sampler_callback": None,
                    }

                    print()
                    # yield batch

            print()

    def nodes_dataloader(
            self, partition_label, labels_for, number_of_hops, batch_size, preload_for="package", labels=None,
            masker_fn=None, label_loader_class=None, label_loader_params=None
    ):
        """
        Returns generator with batches for node-level prediction
        :param partition_label: one of train|val|test
        :param labels_for:
        :param number_of_hops: GNN depth
        :param batch_size: number of examples without negative
        :param preload_for:
        :param labels: DataFrame with labels
        :param masker_fn: function that takes nodes and edges as input and returns an instance of SubwordMasker
        :param label_loader_class: an instance of ElementEmbedder
        :param label_loader_params: dictionary with parameters to be passed to `label_loader_class`
        :return:
        """

        preload_for = SGPartitionStrategies[preload_for]
        labels_for = SGLabelSpec[labels_for]

        if labels_for == SGLabelSpec.subgraphs:
            assert preload_for == SGPartitionStrategies.file, "Subgraphs objectives are currently " \
                                                              "partitioned only in files"

        labels = self.dataset.get_labels_for_partition(labels, partition_label, labels_for, group_by=preload_for)
        node_label_loader = label_loader_class(labels, **label_loader_params)

        node_data = {}
        edge_data = {}

        for partition_key in ["train_mask", "test_mask", "val_mask"]:
            node_data[partition_key] = self.dataset.partition.create_exclusive(partition_key)
        # get name for the partition mask
        current_partition_key = self.dataset.get_proper_partition_column_name(partition_label)

        self.create_batches(
            self._iterate_subgraphs(
                node_data, edge_data, masker_fn, node_label_loader=node_label_loader, grouping_strategy=preload_for
            ),
            number_of_hops, batch_size, current_partition_key, labels_for
        )
