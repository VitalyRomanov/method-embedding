from dgl.data import FB15k237Dataset, AIFBDataset
import torch

from SourceCodeTools.code.data.sourcetrail.Dataset import SourceGraphDataset
from SourceCodeTools.models.graph.NodeEmbedder import TrivialNodeEmbedder
from SourceCodeTools.models.graph.train.sampling_multitask2 import SamplingMultitaskTrainer


class TestGraph(SourceGraphDataset):
    def __init__(self, data_path,
                 label_from, use_node_types=False,
                 use_edge_types=False, filter=None, self_loops=False,
                 train_frac=0.6, random_seed=None, tokenizer_path=None, min_count_for_objectives=1,
                 no_global_edges=False, remove_reverse=False, package_names=None):
        self.random_seed = random_seed
        self.nodes_have_types = use_node_types
        self.edges_have_types = use_edge_types
        self.labels_from = label_from
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.min_count_for_objectives = min_count_for_objectives
        self.no_global_edges = no_global_edges
        self.remove_reverse = remove_reverse

        dataset = AIFBDataset()
        self.g = dataset[0]


class TestTrainer(SamplingMultitaskTrainer):
    def __init__(self, *args, **kwargs):
        super(TestTrainer, self).__init__(*args, **kwargs)

    def create_node_embedder(self, dataset, tokenizer_path, n_dims=None, pretrained_path=None, n_buckets=500000):
        self.node_embedder = TrivialNodeEmbedder(
            emb_size=n_dims,
            dtype=self.dtype,
            n_buckets=n_buckets
        )


def format_data(dataset):
    graph = dataset[0]
    category = dataset.predict_category
    train_mask = graph.nodes[category].data.pop('train_mask')
    test_mask = graph.nodes[category].data.pop('test_mask')
    labels = graph.nodes[category].data.pop('labels')
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

def node_clf():
    dataset = TestGraph()
    dataset = AIFBDataset()
    graph = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = graph.nodes[category].data.pop('train_mask')
    test_mask = graph.nodes[category].data.pop('test_mask')
    labels = graph.nodes[category].data.pop('labels')

    print()

# def main():
#     dataset = FB15k237Dataset()
#     g = dataset.graph
#     e_type = g.edata['e_type']
#
#     train_mask = g.edata['train_mask']
#     val_mask = g.edata['val_mask']
#     test_mask = g.edata['test_mask']
#
#     # graph = dataset[0]
#     # train_mask = graph.edata['train_mask']
#     # test_mask = g.edata['test_mask']
#
#     train_set = torch.arange(g.number_of_edges())[train_mask]
#     val_set = torch.arange(g.number_of_edges())[val_mask]
#
#     train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
#     src, dst = graph.edges(train_idx)
#     rel = graph.edata['etype'][train_idx]
#
#     print()

if __name__=="__main__":
    node_clf()