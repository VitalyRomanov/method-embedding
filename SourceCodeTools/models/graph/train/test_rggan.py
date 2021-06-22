import json
import logging
from copy import copy
from datetime import datetime
from os import mkdir
from os.path import isdir, join
from typing import Tuple

from dgl.data import FB15k237Dataset, AIFBDataset
import torch

from SourceCodeTools.code.data.sourcetrail.Dataset import SourceGraphDataset
from SourceCodeTools.models.graph import RGGAN
from SourceCodeTools.models.graph.NodeEmbedder import TrivialNodeEmbedder
from SourceCodeTools.models.graph.train.sampling_multitask2 import SamplingMultitaskTrainer, select_device
from SourceCodeTools.models.graph.train.utils import get_name, get_model_base
from scripts.training.params import rggan_params
from scripts.training.train import add_train_args, verify_arguments


class TestGraph(SourceGraphDataset):
    def __init__(self, data_path=None,
                 label_from=None, use_node_types=False,
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


def training_procedure(
        dataset, model_name, model_params, args, model_base_path
) -> Tuple[SamplingMultitaskTrainer, dict]:

    device = select_device(args)

    model_params['num_classes'] = args.node_emb_size
    model_params['use_gcn_checkpoint'] = args.use_gcn_checkpoint
    model_params['use_att_checkpoint'] = args.use_att_checkpoint
    model_params['use_gru_checkpoint'] = args.use_gru_checkpoint

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
        'measure_ndcg': args.measure_ndcg,
        'dilate_ndcg': args.dilate_ndcg,
        "objectives": args.objectives.split(",")
    }

    trainer = TestTrainer(
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
        print("Training interrupted")
    except Exception as e:
        raise e

    trainer.eval()
    scores = trainer.final_evaluation()

    trainer.to('cpu')

    return trainer, scores


def main(models, args):
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

            dataset = TestGraph()

            def write_params(args, params):
                args = copy(args.__dict__)
                args.update(params)
                args['activation'] = args['activation'].__name__
                with open(join(model_base, "params.json"), "w") as mdata:
                    mdata.write(json.dumps(args, indent=4))

            write_params(args, params)

            if args.training_mode == "multitask":

                from SourceCodeTools.models.graph.train.sampling_multitask2 import training_procedure

                trainer, scores = \
                    training_procedure(dataset, model, copy(params), args, model_base)

                trainer.save_checkpoint(model_base)
            else:
                raise ValueError("Unknown training mode:", args.training_mode)

            print("Saving...", end="")

            params['activation'] = params['activation'].__name__

            metadata = {
                "base": model_base,
                "name": model_attempt,
                "parameters": params,
                "layers": "embeddings.pkl",
                "mappings": "nodes.csv",
                "state": "state_dict.pt",
                "scores": scores,
                "time": date_time,
            }

            metadata.update(args.__dict__)

            # pickle.dump(dataset, open(join(model_base, "dataset.pkl"), "wb"))
            import pickle
            pickle.dump(trainer.get_embeddings(), open(join(model_base, metadata['layers']), "wb"))

            with open(join(model_base, "metadata.json"), "w") as mdata:
                mdata.write(json.dumps(metadata, indent=4))

            print("done")


def format_data(dataset):
    graph = dataset[0]
    category = dataset.predict_category
    train_mask = graph.nodes[category].data.pop('train_mask')
    test_mask = graph.nodes[category].data.pop('test_mask')
    labels = graph.nodes[category].data.pop('labels')
    train_labels = labels[train_mask]
    test_labels = labels[test_mask]

def node_clf():
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    add_train_args(parser)

    args = parser.parse_args()
    verify_arguments(args)

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

    main(models_, args)


    dataset = TestGraph()
    TestTrainer(dataset)
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
    node_clf()