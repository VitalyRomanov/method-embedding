from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import dgl
from math import ceil
from time import time
from os.path import join
import logging

from SourceCodeTools.graph.model.train.utils import create_elem_embedder, BestScoreTracker
from SourceCodeTools.graph.model.LinkPredictor import LinkPredictor


def _compute_accuracy(pred_, true_):
    return torch.sum(pred_ == true_).item() / len(true_)


class SamplingMultitaskTrainer:

    def __init__(self,
                 dataset=None, model_name=None, model_params=None,
                 trainer_params=None, restore=None, device=None
                 ):

        self.graph_model = model_name(dataset.g, **model_params).to(device)
        self.model_params = model_params
        self.trainer_params = trainer_params
        self.device = device
        self.epoch = 0

        self.ee_node_name = create_elem_embedder(
            dataset.load_node_names(), dataset.nodes,
            self.elem_emb_size, compact_dst=True
        ).to(device)

        self.ee_var_use = create_elem_embedder(
            dataset.load_var_use(), dataset.nodes,
            self.elem_emb_size, compact_dst=True
        ).to(device)

        self.ee_api_call = create_elem_embedder(
            dataset.load_api_call(), dataset.nodes,
            self.elem_emb_size, compact_dst=False
        ).to(device)

        self.lp_node_name = LinkPredictor(self.ee_node_name.emb_size + self.graph_model.emb_size).to(device)
        self.lp_var_use = LinkPredictor(self.ee_var_use.emb_size + self.graph_model.emb_size).to(device)
        self.lp_api_call = LinkPredictor(self.graph_model.emb_size + self.graph_model.emb_size).to(device)

        if restore:
            self.restore_from_checkpoint(self.model_base_path)

        self.optimizer = self._create_optimizer()

        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.991)
        self.best_score = BestScoreTracker()

        self._create_loaders(*self._get_training_targets())

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

    def _extract_embed(self, node_embed, input_nodes):
        emb = {}
        for node_type, nid in input_nodes.items():
            emb[node_type] = node_embed[node_type][nid]
        return emb

    def _logits_batch(self, input_nodes, blocks):

        cumm_logits = []

        if self.use_types:
            emb = self._extract_embed(self.graph_model.node_embed(), input_nodes)
        else:
            if self.ntypes is not None:
                # single node type
                key = next(iter(self.ntypes))
                input_nodes = {key: input_nodes}
                emb = self._extract_embed(self.graph_model.node_embed(), input_nodes)
            else:
                emb = self.graph_model.node_embed()[input_nodes]

        logits = self.graph_model(emb, blocks)

        if self.use_types:
            for ntype in self.graph_model.g.ntypes:

                logits_ = logits.get(ntype, None)
                if logits_ is None:
                    continue

                cumm_logits.append(logits_)
        else:
            if self.ntypes is not None:
                # single node type
                key = next(iter(self.ntypes))
                logits_ = logits[key]
            else:
                logits_ = logits

            cumm_logits.append(logits_)

        return torch.cat(cumm_logits)

    def _logits_embedder(self, node_embeddings, elem_embedder, link_predictor, seeds, negative_factor=1):
        k = negative_factor
        indices = seeds
        batch_size = len(seeds)

        node_embeddings_batch = node_embeddings
        element_embeddings = elem_embedder(elem_embedder[indices.tolist()].to(self.device))

        positive_batch = torch.cat([node_embeddings_batch, element_embeddings], 1)
        labels_pos = torch.ones(batch_size, dtype=torch.long)

        node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
        negative_indices = torch.LongTensor(elem_embedder.sample_negative(batch_size * k)).to(self.device)
        negative_random = elem_embedder(negative_indices)
        negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
        labels_neg = torch.zeros(batch_size * k, dtype=torch.long)

        batch = torch.cat([positive_batch, negative_batch], 0)
        labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)

        logits = link_predictor(batch)

        return logits, labels

    def _handle_non_unique(self, non_unique_ids):
        id_list = non_unique_ids.tolist()
        unique_ids = list(set(id_list))
        new_position = dict(zip(unique_ids, range(len(unique_ids))))
        slice_map = torch.tensor(list(map(lambda x: new_position[x], id_list)), dtype=torch.long)
        return torch.tensor(unique_ids, dtype=torch.long), slice_map

    def _logits_nodes(self, node_embeddings,
                      elem_embedder, link_predictor, create_dataloader,
                      src_seeds, negative_factor=1):
        k = negative_factor
        indices = src_seeds
        batch_size = len(src_seeds)

        node_embeddings_batch = node_embeddings
        next_call_indices = elem_embedder[indices.tolist()]  # this assumes indices is torch tensor

        # dst targets are not unique
        unique_dst, slice_map = self._handle_non_unique(next_call_indices)
        assert all(unique_dst[slice_map] == next_call_indices)

        dataloader = create_dataloader(unique_dst)
        input_nodes, dst_seeds, blocks = next(iter(dataloader))
        blocks = [blk.to(self.device) for blk in blocks]
        assert dst_seeds.shape == unique_dst.shape
        assert all(dst_seeds == unique_dst)
        unique_dst_embeddings = self._logits_batch(input_nodes, blocks)  # use_types, ntypes)
        next_call_embeddings = unique_dst_embeddings[slice_map.to(self.device)]
        positive_batch = torch.cat([node_embeddings_batch, next_call_embeddings], 1)
        labels_pos = torch.ones(batch_size, dtype=torch.long)

        node_embeddings_neg_batch = node_embeddings_batch.repeat(k, 1)
        negative_indices = torch.tensor(elem_embedder.sample_negative(
            batch_size * k), dtype=torch.long)  # embeddings are sampled from 3/4 unigram distribution
        unique_negative, slice_map = self._handle_non_unique(negative_indices)
        assert all(unique_negative[slice_map] == negative_indices)

        dataloader = create_dataloader(unique_negative)
        input_nodes, dst_seeds, blocks = next(iter(dataloader))
        blocks = [blk.to(self.device) for blk in blocks]
        assert dst_seeds.shape == unique_negative.shape
        assert all(dst_seeds == unique_negative)
        unique_negative_random = self._logits_batch(input_nodes, blocks)  # use_types, ntypes)
        negative_random = unique_negative_random[slice_map.to(self.device)]
        negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
        labels_neg = torch.zeros(batch_size * k, dtype=torch.long)

        batch = torch.cat([positive_batch, negative_batch], 0)
        labels = torch.cat([labels_pos, labels_neg], 0).to(self.device)

        logits = link_predictor(batch)

        return logits, labels

    def _logits_node_name(self, input_nodes, seeds, blocks):
        src_embs = self._logits_batch(input_nodes, blocks)
        logits, labels = self._logits_embedder(src_embs, self.ee_node_name, self.lp_node_name, seeds)
        return logits, labels

    def _logits_var_use(self, input_nodes, seeds, blocks):
        src_embs = self._logits_batch(input_nodes, blocks)
        logits, labels = self._logits_embedder(src_embs, self.ee_var_use, self.lp_var_use, seeds)
        return logits, labels

    def _logits_api_call(self, input_nodes, seeds, blocks):
        src_embs = self._logits_batch(input_nodes, blocks)
        logits, labels = self._logits_nodes(src_embs, self.ee_api_call, self.lp_api_call, self._create_api_call_loader,
                                            seeds)
        return logits, labels

    def _get_training_targets(self):
        if hasattr(self.graph_model.g, 'ntypes'):
            self.ntypes = self.graph_model.g.ntypes
            # labels = {ntype: self.graph_model.g.nodes[ntype].data['labels'] for ntype in self.ntypes}
            self.use_types = True

            if len(self.graph_model.g.ntypes) == 1:
                # key = next(iter(labels.keys()))
                # labels = labels[key]
                self.use_types = False
        
            train_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['train_mask']).squeeze()
                for ntype in self.ntypes
            }
            val_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['val_mask']).squeeze()
                for ntype in self.ntypes
            }
            test_idx = {
                ntype: torch.nonzero(self.graph_model.g.nodes[ntype].data['test_mask']).squeeze()
                for ntype in self.ntypes
            }
        else:
            self.ntypes = None
            # labels = g.ndata['labels']
            train_idx = self.graph_model.g.ndata['train_mask']
            val_idx = self.graph_model.g.ndata['val_mask']
            test_idx = self.graph_model.g.ndata['test_mask']
            self.use_types = False

        return train_idx, val_idx, test_idx

    def _evaluate_embedder(self, ee, lp, loader, neg_sampling_factor=1):

        total_loss = 0
        total_acc = 0
        count = 0

        for input_nodes, seeds, blocks in loader:
            blocks = [blk.to(self.device) for blk in blocks]

            src_embs = self._logits_batch(input_nodes, blocks)
            logits, labels = self._logits_embedder(src_embs, ee, lp, seeds, neg_sampling_factor)

            logp = nn.functional.log_softmax(logits, 1)
            loss = nn.functional.cross_entropy(logp, labels)
            acc = _compute_accuracy(logp.argmax(dim=1), labels)

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count

    def _evaluate_nodes(self, ee, lp, create_api_call_loader, loader,
                        neg_sampling_factor=1
                        ):

        total_loss = 0
        total_acc = 0
        count = 0

        for input_nodes, seeds, blocks in loader:
            blocks = [blk.to(self.device) for blk in blocks]

            src_embs = self._logits_batch(input_nodes, blocks)
            logits, labels = self._logits_nodes(src_embs, ee, lp, create_api_call_loader, seeds, neg_sampling_factor)

            logp = nn.functional.log_softmax(logits, 1)
            loss = nn.functional.cross_entropy(logp, labels)
            acc = _compute_accuracy(logp.argmax(dim=1), labels)

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count

    def _evaluate_objectives(
            self, loader_node_name, loader_var_use,
            loader_api_call, neg_sampling_factor
    ):

        node_name, node_name_acc = self._evaluate_embedder(
            self.ee_node_name, self.lp_node_name, loader_node_name, neg_sampling_factor=neg_sampling_factor
        )

        var_use_loss, var_use_acc = self._evaluate_embedder(
            self.ee_var_use, self.lp_var_use, loader_var_use, neg_sampling_factor=neg_sampling_factor
        )

        api_call_loss, api_call_acc = self._evaluate_nodes(self.ee_api_call, self.lp_api_call,
                                                           self._create_api_call_loader, loader_api_call,
                                                           neg_sampling_factor=neg_sampling_factor)

        loss = node_name + var_use_loss + api_call_loss

        return loss, node_name_acc, var_use_acc, api_call_acc

    def _idx_len(self, idx):
        if isinstance(idx, dict):
            length = 0
            for key in idx:
                length += len(idx[key])
        else:
            length = len(idx)
        return length

    def _get_loaders(self, train_idx, val_idx, test_idx, batch_size):
        layers = len(self.graph_model.layers)
        # train sampler
        sampler = dgl.dataloading.MultiLayerNeighborSampler([self.sampling_neighbourhood_size] * layers)
        loader = dgl.dataloading.NodeDataLoader(
            self.graph_model.g, train_idx, sampler, batch_size=batch_size, shuffle=False, num_workers=0)

        # validation sampler
        # we do not use full neighbor to save computation resources
        val_sampler = dgl.dataloading.MultiLayerNeighborSampler([self.sampling_neighbourhood_size] * layers)
        val_loader = dgl.dataloading.NodeDataLoader(
            self.graph_model.g, val_idx, val_sampler, batch_size=batch_size, shuffle=False, num_workers=0)

        # we do not use full neighbor to save computation resources
        test_sampler = dgl.dataloading.MultiLayerNeighborSampler([self.sampling_neighbourhood_size] * layers)
        test_loader = dgl.dataloading.NodeDataLoader(
            self.graph_model.g, test_idx, test_sampler, batch_size=batch_size, shuffle=False, num_workers=0)

        return loader, val_loader, test_loader

    def _create_loaders(self, train_idx, val_idx, test_idx):

        train_idx_node_name, val_idx_node_name, test_idx_node_name = self.ee_node_name.create_idx_pools(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )
        train_idx_var_use, val_idx_var_use, test_idx_var_use = self.ee_var_use.create_idx_pools(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )
        train_idx_api_call, val_idx_api_call, test_idx_api_call = self.ee_api_call.create_idx_pools(
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx
        )

        logging.info(
            f"Pool sizes : train {self._idx_len(train_idx_node_name)}, "
            f"val {self._idx_len(val_idx_node_name)}, "
            f"test {self._idx_len(test_idx_node_name)}."
        )
        logging.info(
            f"Pool sizes : train {self._idx_len(train_idx_var_use)}, "
            f"val {self._idx_len(val_idx_var_use)}, "
            f"test {self._idx_len(test_idx_var_use)}."
        )
        logging.info(
            f"Pool sizes : train {self._idx_len(train_idx_api_call)}, "
            f"val {self._idx_len(val_idx_api_call)}, "
            f"test {self._idx_len(test_idx_api_call)}."
        )

        batch_size_node_name = self.batch_size

        def estimate_batch_size(indexes):
            return ceil(self._idx_len(indexes) / ceil(self._idx_len(train_idx_node_name) / batch_size_node_name))

        batch_size_var_use = estimate_batch_size(train_idx_var_use)
        batch_size_api_call = estimate_batch_size(train_idx_api_call)

        self.loader_node_name, self.test_loader_node_name, self.val_loader_node_name = self._get_loaders(
            train_idx=train_idx_node_name, val_idx=val_idx_node_name, test_idx=test_idx_node_name,
            batch_size=batch_size_node_name
        )
        self.loader_var_use, self.test_loader_var_use, self.val_loader_var_use = self._get_loaders(
            train_idx=train_idx_var_use, val_idx=val_idx_var_use, test_idx=test_idx_var_use,
            batch_size=batch_size_var_use
        )
        self.loader_api_call, self.test_loader_api_call, self.val_loader_api_call = self._get_loaders(
            train_idx=train_idx_api_call, val_idx=val_idx_api_call, test_idx=test_idx_api_call,
            batch_size=batch_size_api_call
        )

    def _create_api_call_loader(self, indices):
        sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [self.sampling_neighbourhood_size] * len(self.graph_model.layers))
        return dgl.dataloading.NodeDataLoader(
            self.graph_model.g, indices, sampler, batch_size=len(indices), num_workers=0)

    def _create_optimizer(self):
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(
            [
                {'params': self.graph_model.parameters()},
                {'params': self.ee_node_name.parameters()},
                {'params': self.ee_var_use.parameters()},
                {'params': self.ee_api_call.parameters()},
                {'params': self.lp_node_name.parameters()},
                {'params': self.lp_var_use.parameters()},
                {'params': self.lp_api_call.parameters()},
            ], lr=self.lr
        )
        return optimizer

    def train_all(self):
        """
        Training procedure for the model with node classifier
        :return:
        """

        for epoch in range(self.epochs):
            self.epoch = epoch

            start = time()

            for i, ((input_nodes_node_name, seeds_node_name, blocks_node_name),
                    (input_nodes_var_use, seeds_var_use, blocks_var_use),
                    (input_nodes_api_call, seeds_api_call, blocks_api_call)) in \
                    enumerate(zip(
                        self.loader_node_name,
                        self.loader_var_use,
                        self.loader_api_call)):

                blocks_node_name = [blk.to(self.device) for blk in blocks_node_name]
                blocks_var_use = [blk.to(self.device) for blk in blocks_var_use]
                blocks_api_call = [blk.to(self.device) for blk in blocks_api_call]

                logits_node_name, labels_node_name = self._logits_node_name(
                    input_nodes_node_name, seeds_node_name, blocks_node_name
                )

                logits_var_use, labels_var_use = self._logits_var_use(
                    input_nodes_var_use, seeds_var_use, blocks_var_use
                )

                logits_api_call, labels_api_call = self._logits_api_call(
                    input_nodes_api_call, seeds_api_call, blocks_api_call
                )

                train_acc_node_name = _compute_accuracy(logits_node_name.argmax(dim=1), labels_node_name)
                train_acc_var_use = _compute_accuracy(logits_var_use.argmax(dim=1), labels_var_use)
                train_acc_api_call = _compute_accuracy(logits_api_call.argmax(dim=1), labels_api_call)

                train_logits = torch.cat([logits_node_name, logits_var_use, logits_api_call], 0)
                train_labels = torch.cat([labels_node_name, labels_var_use, labels_api_call], 0)

                logp = nn.functional.log_softmax(train_logits, 1)
                loss = nn.functional.nll_loss(logp, train_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            _, val_acc_node_name, val_acc_var_use, val_acc_api_call = self._evaluate_objectives(
                self.val_loader_node_name, self.val_loader_var_use, self.val_loader_api_call, self.neg_sampling_factor
            )

            _, test_acc_node_name, test_acc_var_use, test_acc_api_call = self._evaluate_objectives(
                self.test_loader_node_name, self.test_loader_var_use, self.test_loader_api_call,
                self.neg_sampling_factor
            )

            end = time()

            self.best_score.track_best(
                epoch=epoch, loss=loss.item(),
                train_acc_fname=train_acc_node_name, val_acc_fname=val_acc_node_name, test_acc_fname=test_acc_node_name,
                train_acc_varuse=train_acc_var_use, val_acc_varuse=val_acc_var_use, test_acc_varuse=test_acc_var_use,
                train_acc_apicall=train_acc_api_call, val_acc_apicall=val_acc_api_call,
                test_acc_apicall=test_acc_api_call,
                time=end - start
            )

            self.save_checkpoint(self.model_base_path)

            self.lr_scheduler.step()

    def save_checkpoint(self, checkpoint_path=None, checkpoint_name=None, **kwargs):

        checkpoint_path = join(checkpoint_path, "saved_state.pt")

        param_dict = {
            'graph_model': self.graph_model.state_dict(),
            'ee_node_name': self.ee_node_name.state_dict(),
            'ee_var_use': self.ee_var_use.state_dict(),
            'ee_api_call': self.ee_api_call.state_dict(),
            "lp_node_name": self.lp_node_name.state_dict(),
            "lp_var_use": self.lp_var_use.state_dict(),
            "lp_api_call": self.ee_api_call.state_dict(),
            "epoch": self.epoch
        }

        if len(kwargs) > 0:
            param_dict.update(kwargs)

        torch.save(param_dict, checkpoint_path)

    def restore_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(join(checkpoint_path, "saved_state.pt"))
        self.graph_model.load_state_dict(checkpoint['graph_model'])
        self.ee_node_name.load_state_dict(checkpoint['ee_node_name'])
        self.ee_var_use.load_state_dict(checkpoint['ee_var_use'])
        self.ee_api_call.load_state_dict(checkpoint['ee_api_call'])
        self.lp_node_name.load_state_dict(checkpoint['lp_node_name'])
        self.lp_var_use.load_state_dict(checkpoint['lp_var_use'])
        self.lp_api_call.load_state_dict(checkpoint['lp_api_call'])
        logging.info(f"Restored from epoch {checkpoint['epoch']}")

    def final_evaluation(self):

        loss, train_acc_node_name, train_acc_var_use, train_acc_api_call = self._evaluate_objectives(
            self.loader_node_name, self.loader_var_use, self.loader_api_call, 1
        )

        _, val_acc_node_name, val_acc_var_use, val_acc_api_call = self._evaluate_objectives(
            self.val_loader_node_name, self.val_loader_var_use, self.val_loader_api_call, 1
        )

        _, test_acc_node_name, test_acc_var_use, test_acc_api_call = self._evaluate_objectives(
            self.test_loader_node_name, self.test_loader_var_use, self.test_loader_api_call, 1
        )

        scores = {
            # "loss": loss.item(),
            "train_acc_node_name": train_acc_node_name,
            "val_acc_node_name": val_acc_node_name,
            "test_acc_node_name": test_acc_node_name,
            "train_acc_var_use": train_acc_var_use,
            "val_acc_var_use": val_acc_var_use,
            "test_acc_var_use": test_acc_var_use,
            "train_acc_api_call": train_acc_api_call,
            "val_acc_api_call": val_acc_api_call,
            "test_acc_api_call": test_acc_api_call,
        }

        print(
            f'Final Eval : node name Train Acc {scores["train_acc_node_name"]:.4f}, '
            f'node name Val Acc {scores["val_acc_node_name"]:.4f}, '
            f'node name Test Acc {scores["test_acc_node_name"]:.4f}, '
            f'var use Train Acc {scores["train_acc_var_use"]:.4f}, '
            f'var use Val Acc {scores["val_acc_var_use"]:.4f}, '
            f'var use Test Acc {scores["test_acc_var_use"]:.4f}, '
            f'api call Train Acc {scores["train_acc_api_call"]:.4f}, '
            f'api call Val Acc {scores["val_acc_api_call"]:.4f}, '
            f'api call Test Acc {scores["test_acc_api_call"]:.4f}'
        )

        return scores

    def eval(self):
        self.graph_model.eval()
        self.ee_node_name.eval()
        self.ee_var_use.eval()
        self.ee_api_call.eval()
        self.lp_node_name.eval()
        self.lp_var_use.eval()
        self.lp_api_call.eval()

    def to(self, device):
        # self.graph_model.to(device)
        self.ee_node_name.to(device)
        self.ee_var_use.to(device)
        self.ee_api_call.to(device)
        self.lp_node_name.to(device)
        self.lp_var_use.to(device)
        self.lp_api_call.to(device)


def select_device(args):
    device = 'cpu'
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = 'cuda:%d' % args.gpu
    return device


def training_procedure(
        dataset, model_name, model_params, epochs, args, model_base_path
) -> Tuple[SamplingMultitaskTrainer, dict]:

    device = select_device(args)

    model_params['num_classes'] = args.node_emb_size

    trainer_params = {
        'lr': model_params.pop('lr'),
        'batch_size': args.batch_size,
        'sampling_neighbourhood_size': args.num_per_neigh,
        'neg_sampling_factor': args.neg_sampling_factor,
        'epochs': epochs,
        # 'node_name_file': args.fname_file,
        # 'var_use_file': args.varuse_file,
        # 'call_seq_file': args.call_seq_file,
        'elem_emb_size': args.elem_emb_size,
        'model_base_path': model_base_path
    }

    trainer = SamplingMultitaskTrainer(
        dataset=dataset,
        model_name=model_name,
        model_params=model_params,
        trainer_params=trainer_params,
        restore=args.restore_state,
        device=device
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
