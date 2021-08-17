import itertools
import json
import logging
from copy import copy
from datetime import datetime
from os import mkdir
from os.path import isdir, join
from typing import Tuple
import pandas as pd

from dgl.data import WN18Dataset, FB15kDataset, FB15k237Dataset, AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
import torch
from sklearn.model_selection import ParameterGrid

from SourceCodeTools.code.data.sourcetrail.Dataset import SourceGraphDataset
from SourceCodeTools.models.graph import RGGAN
from SourceCodeTools.models.graph.NodeEmbedder import NodeIdEmbedder
from SourceCodeTools.models.graph.train.sampling_multitask2 import SamplingMultitaskTrainer
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from SourceCodeTools.models.training_options import add_gnn_train_args, verify_arguments

rggan_grids = [
    {
        'h_dim': [15],
        'num_bases': [-1],
        'num_steps': [5],
        'dropout': [0.0],
        'use_self_loop': [False],
        'activation': [torch.nn.functional.hardtanh], # torch.nn.functional.hardswish], #[torch.nn.functional.hardtanh], #torch.nn.functional.leaky_relu
        'lr': [1e-3], # 1e-4]
    }
]

rggan_params = list(
    itertools.chain.from_iterable(
        [ParameterGrid(p) for p in rggan_grids]
    )
)



class TestNodeClfGraph(SourceGraphDataset):
    def __init__(self, data_loader,
                 label_from=None, use_node_types=True,
                 use_edge_types=True, filter=None, self_loops=False,
                 train_frac=0.6, random_seed=None, tokenizer_path=None, min_count_for_objectives=1,
                 no_global_edges=False, remove_reverse=False, package_names=None):
        self.random_seed = random_seed
        self.nodes_have_types = use_node_types
        self.edges_have_types = use_edge_types
        self.labels_from = label_from
        # self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.min_count_for_objectives = min_count_for_objectives
        self.no_global_edges = no_global_edges
        self.remove_reverse = remove_reverse

        # dataset = AIFBDataset()
        # dataset = MUTAGDataset()
        # dataset = BGSDataset()
        # dataset = AMDataset()
        dataset = data_loader()

        self.nodes, self.edges, self.typed_id_map = self.create_nodes_edges_df(dataset)

        # index is later used for sampling and is assumed to be unique
        assert len(self.nodes) == len(self.nodes.index.unique())
        assert len(self.edges) == len(self.edges.index.unique())

        if self_loops:
            self.nodes, self.edges = SourceGraphDataset.assess_need_for_self_loops(self.nodes, self.edges)

        if filter is not None:
            for e_type in filter.split(","):
                logging.info(f"Filtering edge type {e_type}")
                self.edges = self.edges.query(f"type != '{e_type}'")

        # if self.remove_reverse:
        #     self.remove_reverse_edges()
        #
        # if self.no_global_edges:
        #     self.remove_global_edges()

        if use_node_types is False and use_edge_types is False:
            new_nodes, new_edges = self.create_nodetype_edges()
            self.nodes = self.nodes.append(new_nodes, ignore_index=True)
            self.edges = self.edges.append(new_edges, ignore_index=True)

        self.nodes['type_backup'] = self.nodes['type']
        if not self.nodes_have_types:
            self.nodes['type'] = "node_"
            self.nodes = self.nodes.astype({'type': 'category'})

        self.add_embeddable_flag()

        # need to do this to avoid issues insode dgl library
        self.edges['type'] = self.edges['type'].apply(lambda x: f"{x}_")
        self.edges['type_backup'] = self.edges['type']
        if not self.edges_have_types:
            self.edges['type'] = "edge_"
            self.edges = self.edges.astype({'type': 'category'})

        # compact labels
        # self.nodes['label'] = self.nodes[label_from]
        # self.nodes = self.nodes.astype({'label': 'category'})
        # self.label_map = compact_property(self.nodes['label'])
        # assert any(pandas.isna(self.nodes['label'])) is False

        logging.info(f"Unique nodes: {len(self.nodes)}, node types: {len(self.nodes['type'].unique())}")
        logging.info(f"Unique edges: {len(self.edges)}, edge types: {len(self.edges['type'].unique())}")

        # self.nodes, self.label_map = self.add_compact_labels()
        self.add_typed_ids()

        # self.add_splits(train_frac=train_frac, package_names=package_names)

        # self.mark_leaf_nodes()

        self.create_hetero_graph()

        self.update_global_id()

        self.nodes.sort_values('global_graph_id', inplace=True)

        # self.splits = SourceGraphDataset.get_global_graph_id_splits(self.nodes)

    def create_nodes_edges_df(self, dataset):
        graph = dataset[0]
        nodes_df = None

        node_id_map = {}
        typed_id_map = {}

        for ntype in graph.ntypes:
            typed_id = graph.nodes(ntype=ntype).tolist()
            type = [ntype] * len(typed_id)
            name = list(map(lambda x: ntype+f"_{x}", typed_id))
            id_ = graph.nodes[ntype].data["_ID"].tolist()
            node_dict = {"id": id_, "type": type, "name": name, "typed_id": typed_id}

            node_id_map[ntype] = dict(zip(typed_id, id_))
            typed_id_map[ntype] = dict(zip(id_, typed_id))

            if "labels" in graph.nodes[ntype].data:
                node_dict["labels"] = graph.nodes[ntype].data["labels"].tolist()
            else:
                node_dict["labels"] = [-1] * len(typed_id)

            if "train_mask" in graph.nodes[ntype].data:
                node_dict["train_mask"] = graph.nodes[ntype].data["train_mask"].bool().tolist()
            else:
                node_dict["train_mask"] = [False] * len(typed_id)

            if "test_mask" in graph.nodes[ntype].data:
                node_dict["test_mask"] = graph.nodes[ntype].data["test_mask"].bool().tolist()
            else:
                node_dict["test_mask"] = [False] * len(typed_id)

            if "val_mask" in graph.nodes[ntype].data:
                node_dict["val_mask"] = graph.nodes[ntype].data["val_mask"].bool().tolist()
            else:
                node_dict["val_mask"] = [False] * len(typed_id)

            if nodes_df is None:
                nodes_df = pd.DataFrame.from_dict(node_dict)
            else:
                nodes_df = nodes_df.append(pd.DataFrame.from_dict(node_dict))

        assert len(nodes_df["id"]) == len(nodes_df["id"].unique()) # thus is a must have assert

        nodes_df = nodes_df.reset_index(drop=True)
        # contiguous_node_index = dict(zip(nodes_df["index"], nodes_df.index))

        edges_df = None
        for srctype, etype, dsttype in graph.canonical_etypes:
            src, dst = graph.edges(etype=(srctype, etype, dsttype))
            src = list(map(lambda x: node_id_map[srctype][x], src.tolist()))
            dst = list(map(lambda x: node_id_map[dsttype][x], dst.tolist()))

            edge_data = {
                "id": graph.edata["_ID"][(srctype, etype, dsttype)].tolist(),
                "type": [etype] * len(src),
                "src": src,
                "dst": dst
            }

            if edges_df is None:
                edges_df = pd.DataFrame.from_dict(edge_data)
            else:
                edges_df = edges_df.append(pd.DataFrame.from_dict(edge_data))

        # assert len(edges_df["id"]) == len(edges_df["id"].unique()) # fails with AMDataset

        edges_df = edges_df.reset_index(drop=True)

        return nodes_df, edges_df, typed_id_map

    def add_embeddable_flag(self):
        self.nodes['embeddable'] = True

    def add_typed_ids(self):
        pass

    def load_node_classes(self):
        labels = self.nodes.query("train_mask == True or test_mask == True or val_mask == True")[["id", "labels"]].rename({
            "id": "src",
            "labels": "dst"
        }, axis=1)
        return labels


class TestLinkPredGraph(TestNodeClfGraph):
    def create_nodes_edges_df(self, dataset):
        graph = dataset[0]
        nodes_df = None

        node_id_map = {}
        typed_id_map = {}

        for ntype in graph.ntypes:
            typed_id = graph.nodes(ntype=ntype).tolist()
            type = [ntype] * len(typed_id)
            name = list(map(lambda x: ntype + f"_{x}", typed_id))
            id_ = graph.nodes[ntype].data["_ID"].tolist()
            node_dict = {"id": id_, "type": type, "name": name, "typed_id": typed_id}

            node_id_map[ntype] = dict(zip(typed_id, id_))
            typed_id_map[ntype] = dict(zip(id_, typed_id))

            if "labels" in graph.nodes[ntype].data:
                node_dict["labels"] = graph.nodes[ntype].data["labels"].tolist()
            else:
                node_dict["labels"] = [-1] * len(typed_id)

            if "train_mask" in graph.nodes[ntype].data:
                node_dict["train_mask"] = graph.nodes[ntype].data["train_mask"].bool().tolist()
            else:
                node_dict["train_mask"] = [False] * len(typed_id)

            if "test_mask" in graph.nodes[ntype].data:
                node_dict["test_mask"] = graph.nodes[ntype].data["test_mask"].bool().tolist()
            else:
                node_dict["test_mask"] = [False] * len(typed_id)

            if "val_mask" in graph.nodes[ntype].data:
                node_dict["val_mask"] = graph.nodes[ntype].data["val_mask"].bool().tolist()
            else:
                node_dict["val_mask"] = [False] * len(typed_id)

            if nodes_df is None:
                nodes_df = pd.DataFrame.from_dict(node_dict)
            else:
                nodes_df = nodes_df.append(pd.DataFrame.from_dict(node_dict))

        assert len(nodes_df["id"]) == len(nodes_df["id"].unique())

        nodes_df = nodes_df.reset_index(drop=True)
        # contiguous_node_index = dict(zip(nodes_df["index"], nodes_df.index))

        edges_df = None
        for srctype, etype, dsttype in graph.canonical_etypes:
            src, dst = graph.edges(etype=(srctype, etype, dsttype))
            src = list(map(lambda x: node_id_map[srctype][x], src.tolist()))
            dst = list(map(lambda x: node_id_map[dsttype][x], dst.tolist()))

            edge_data = {
                "id": graph.edata["_ID"][(srctype, etype, dsttype)].tolist(),
                "type": [etype] * len(src),
                "src": src,
                "dst": dst
            }

            if edges_df is None:
                edges_df = pd.DataFrame.from_dict(edge_data)
            else:
                edges_df = edges_df.append(pd.DataFrame.from_dict(edge_data))

        assert len(edges_df["id"]) == len(edges_df["id"].unique())

        edges_df = edges_df.reset_index(drop=True)

        return nodes_df, edges_df, typed_id_map


class TestTrainer(SamplingMultitaskTrainer):
    def __init__(self, *args, **kwargs):
        super(TestTrainer, self).__init__(*args, **kwargs)

    def create_node_embedder(self, dataset, tokenizer_path, n_dims=None, pretrained_path=None, n_buckets=500000):
        self.node_embedder = NodeIdEmbedder(
            nodes=dataset.nodes,
            emb_size=n_dims,
            dtype=self.dtype,
            n_buckets=len(dataset.nodes) + 1 # override this because bucket size should be the same as number of nodes for this embedder
        )


def main_node_clf(models, args, data_loader):
    for model, param_grid in models.items():
        for params in param_grid:

            if args.h_dim is None:
                params["h_dim"] = args.node_emb_size
            else:
                params["h_dim"] = args.h_dim

            params["num_steps"] = args.n_layers

            date_time = str(datetime.now())
            print("\n\n")
            print(date_time)
            print(f"Model: {model.__name__}, Params: {params}")

            model_attempt = get_name(model, date_time)

            model_base = get_model_base(args, model_attempt)

            dataset = TestNodeClfGraph(data_loader=data_loader)

            args.objectives = "node_clf"

            from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure

            trainer, scores = training_procedure(dataset, model, copy(params), args, model_base, trainer=TestTrainer)

            return scores


def main_link_pred(models, args, data_loader):
    for model, param_grid in models.items():
        for params in param_grid:

            if args.h_dim is None:
                params["h_dim"] = args.node_emb_size
            else:
                params["h_dim"] = args.h_dim

            params["num_steps"] = args.n_layers

            date_time = str(datetime.now())
            print("\n\n")
            print(date_time)
            print(f"Model: {model.__name__}, Params: {params}")

            model_attempt = get_name(model, date_time)

            model_base = get_model_base(args, model_attempt)

            dataset = TestLinkPredGraph(data_loader=data_loader)

            args.objectives = "link_pred"

            from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure

            trainer, scores = training_procedure(dataset, model, copy(params), args, model_base, trainer=TestTrainer)

            return scores

def format_data(dataset):
    graph = dataset[0]
    category = dataset.predict_category
    train_mask = graph.nodes[category].data.pop('train_mask')
    test_mask = graph.nodes[category].data.pop('test_mask')
    labels = graph.nodes[category].data.pop('labels')
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

def node_clf(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    models_ = {
        # GCNSampling: gcnsampling_params,
        # GATSampler: gatsampling_params,
        # RGCNSampling: rgcnsampling_params,
        # RGAN: rgcnsampling_params,
        RGGAN: rggan_params

    }

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)
    args.save_checkpoints = False

    data_loaders = [AIFBDataset, MUTAGDataset, BGSDataset, AMDataset]
    # data_loaders = [AMDataset]

    for dl in data_loaders:
        print(dl.__name__)
        scores = main_node_clf(models_, args, data_loader=dl)
        print("\t", scores)


def link_pred(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    models_ = {
        # GCNSampling: gcnsampling_params,
        # GATSampler: gatsampling_params,
        # RGCNSampling: rgcnsampling_params,
        # RGAN: rgcnsampling_params,
        RGGAN: rggan_params

    }

    if not isdir(args.model_output_dir):
        mkdir(args.model_output_dir)
    args.save_checkpoints = False

    data_loaders = [WN18Dataset, FB15kDataset, FB15k237Dataset]

    for dl in data_loaders:
        print(dl.__name__)
        scores = main_link_pred(models_, args, data_loader=dl)
        print("\t", scores)


    # dataset = AIFBDataset()
    # graph = dataset[0]
    # category = dataset.predict_category
    # num_classes = dataset.num_classes
    # train_mask = graph.nodes[category].data.pop('train_mask')
    # test_mask = graph.nodes[category].data.pop('test_mask')
    # labels = graph.nodes[category].data.pop('labels')
    #
    # print()

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
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    add_gnn_train_args(parser)

    args = parser.parse_args()
    verify_arguments(args)

    node_clf(args)
    # link_pred(args)