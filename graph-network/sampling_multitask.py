import torch
import torch.nn as nn
import dgl
from math import ceil
from time import time

from utils import create_elem_embedder, track_best_multitask, create_idx_pools
from ElementEmbedder import ElementEmbedder
from LinkPredictor import LinkPredictor


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        emb[ntype] = node_embed[ntype][nid]
    return emb


def logits_batch(model, input_nodes, blocks, use_types, ntypes):
    cumm_logits = []

    if use_types:
        emb = extract_embed(model.node_embed(), input_nodes)
    else:
        if ntypes is not None:
            # single node type
            key = next(iter(ntypes))
            input_nodes = {key: input_nodes}
            emb = extract_embed(model.node_embed(), input_nodes)
        else:
            emb = model.node_embed()[input_nodes]

    logits = model(emb, blocks)

    if use_types:
        for ntype in ntypes:

            logits_ = logits.get(ntype, None)
            if logits_ is None: continue

            cumm_logits.append(logits_)
    else:
        if ntypes is not None:
            # single node type
            key = next(iter(ntypes))
            logits_ = logits[key]
        else:
            logits_ = logits

        cumm_logits.append(logits_)

    return torch.cat(cumm_logits)


def logits_embedder(node_embeddings, elem_embeder, link_predictor, seeds, negative_factor=1, device='cpu'):
    K = negative_factor
    indices = seeds
    batch_size = len(seeds)

    node_embeddings_batch = node_embeddings
    element_embeddings = elem_embeder(elem_embeder[indices.tolist()].to(device))

    positive_batch = torch.cat([node_embeddings_batch, element_embeddings], 1)
    labels_pos = torch.ones(batch_size, dtype=torch.long)

    node_embeddings_neg_batch = node_embeddings_batch.repeat(K, 1)
    negative_indices = torch.LongTensor(elem_embeder.sample_negative(batch_size * K)).to(device)
    negative_random = elem_embeder(negative_indices)
    negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
    labels_neg = torch.zeros(batch_size * K, dtype=torch.long)

    batch = torch.cat([positive_batch, negative_batch], 0)
    labels = torch.cat([labels_pos, labels_neg], 0).to(device)

    logits = link_predictor(batch)

    return logits, labels

def handle_non_unique(non_unique_ids):
    id_list = non_unique_ids.tolist()
    unique_ids = list(set(id_list))
    new_position = dict(zip(unique_ids, range(len(unique_ids))))
    slice_map = torch.tensor(list(map(lambda x: new_position[x], id_list)), dtype=torch.long)
    return torch.tensor(unique_ids, dtype=torch.long), slice_map

def logits_nodes(model, node_embeddings,
                 elem_embeder, link_predictor, create_dataloader,
                 src_seeds, use_types, ntypes, negative_factor=1, device='cpu'):
    K = negative_factor
    indices = src_seeds
    batch_size = len(src_seeds)

    node_embeddings_batch = node_embeddings
    next_call_indices = elem_embeder[indices.tolist()] # this assumes indices is torch tensor

    # dst targets are not unique
    unique_dst, slice_map = handle_non_unique(next_call_indices)
    assert all(unique_dst[slice_map] == next_call_indices)

    dataloader = create_dataloader(unique_dst)
    input_nodes, dst_seeds, blocks = next(iter(dataloader))
    blocks = [blk.to(device) for blk in blocks]
    assert dst_seeds.shape == unique_dst.shape
    assert all(dst_seeds == unique_dst)
    unique_dst_embeddings = logits_batch(model, input_nodes, blocks, use_types, ntypes)
    next_call_embeddings = unique_dst_embeddings[slice_map.to(device)]
    positive_batch = torch.cat([node_embeddings_batch, next_call_embeddings], 1)
    labels_pos = torch.ones(batch_size, dtype=torch.long)

    node_embeddings_neg_batch = node_embeddings_batch.repeat(K, 1)
    negative_indices = torch.tensor(elem_embeder.sample_negative(
        batch_size * K), dtype=torch.long)  # embeddings are sampled from 3/4 unigram distribution
    unique_negative, slice_map = handle_non_unique(negative_indices)
    assert all(unique_negative[slice_map] == negative_indices)

    dataloader = create_dataloader(unique_negative)
    input_nodes, dst_seeds, blocks = next(iter(dataloader))
    blocks = [blk.to(device) for blk in blocks]
    assert dst_seeds.shape == unique_negative.shape
    assert all(dst_seeds == unique_negative)
    unique_negative_random = logits_batch(model, input_nodes, blocks, use_types, ntypes)
    negative_random = unique_negative_random[slice_map.to(device)]
    negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
    labels_neg = torch.zeros(batch_size * K, dtype=torch.long)

    batch = torch.cat([positive_batch, negative_batch], 0)
    labels = torch.cat([labels_pos, labels_neg], 0).to(device)

    logits = link_predictor(batch)

    return logits, labels


def labels_batch(seeds, labels, use_types, ntypes):
    cumm_labels = []

    if use_types:
        for ntype in ntypes:
            cumm_labels.append(labels[ntype][seeds[ntype]])
    else:
        cumm_labels.append(labels[seeds])

    return torch.cat(cumm_labels)


def evaluate(model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall,
             create_apicall_loader, loader,
             use_types, ntypes=None, device='cpu', neg_samplig_factor=1):
    # model.eval()
    total_loss = 0.
    total_acc_fname = 0.
    total_acc_varuse = 0.
    total_acc_apicall = 0.
    count = 0

    for input_nodes, seeds, blocks in loader:
        blocks = [blk.to(device) for blk in blocks]

        src_embs = logits_batch(model, input_nodes, blocks, use_types, ntypes)
        logits_fname, labels_fname = logits_embedder(src_embs, ee_fname, lp_fname, seeds, neg_samplig_factor, device=device)
        logits_varuse, labels_varuse = logits_embedder(src_embs, ee_varuse, lp_varuse, seeds, neg_samplig_factor, device=device)
        logits_apicall, labels_apicall = logits_nodes(model, src_embs,
                                                      ee_apicall, lp_apicall, create_apicall_loader,
                                                      seeds, use_types, ntypes, neg_samplig_factor, device=device)

        acc_fname = torch.sum(logits_fname.argmax(dim=1) == labels_fname).item() / len(labels_fname)
        acc_varuse = torch.sum(logits_varuse.argmax(dim=1) == labels_varuse).item() / len(labels_varuse)
        acc_apicall = torch.sum(logits_apicall.argmax(dim=1) == labels_apicall).item() / len(labels_apicall)

        logits = torch.cat([logits_fname, logits_varuse, logits_apicall], 0)
        labels = torch.cat([labels_fname, labels_varuse, labels_apicall], 0)

        logp = nn.functional.log_softmax(logits, 1)
        loss = nn.functional.nll_loss(logp, labels)

        total_loss += loss.item()
        total_acc_fname += acc_fname
        total_acc_varuse += acc_varuse
        total_acc_apicall += acc_apicall
        count += 1

    return total_loss / count, total_acc_fname / count, total_acc_varuse / count, total_acc_apicall / count


def evaluate_embedder(model, ee, lp, loader,
             use_types, ntypes=None, device='cpu', neg_samplig_factor=1):

    total_loss = 0
    total_acc = 0
    count = 0

    for input_nodes, seeds, blocks in loader:
        blocks = [blk.to(device) for blk in blocks]

        src_embs = logits_batch(model, input_nodes, blocks, use_types, ntypes)
        logits, labels = logits_embedder(src_embs, ee, lp, seeds, neg_samplig_factor, device=device)

        logp = nn.functional.log_softmax(logits, 1)
        loss = nn.functional.cross_entropy(logp, labels)
        acc = torch.sum(logp.argmax(dim=1) == labels).item() / len(labels)

        total_loss += loss.item()
        total_acc += acc
        count += 1
    return total_loss / count, total_acc / count


def evaluate_nodes(model, ee, lp,
             create_apicall_loader, loader,
             use_types, ntypes=None, device='cpu', neg_samplig_factor=1):

    total_loss = 0
    total_acc = 0
    count = 0

    for input_nodes, seeds, blocks in loader:
        blocks = [blk.to(device) for blk in blocks]

        src_embs = logits_batch(model, input_nodes, blocks, use_types, ntypes)
        logits, labels = logits_nodes(model, src_embs,
                                          ee, lp, create_apicall_loader,
                                          seeds, use_types, ntypes, neg_samplig_factor, device=device)

        logp = nn.functional.log_softmax(logits, 1)
        loss = nn.functional.cross_entropy(logp, labels)
        acc = torch.sum(logp.argmax(dim=1) == labels).item() / len(labels)

        total_loss += loss.item()
        total_acc += acc
        count += 1
    return total_loss / count, total_acc / count


def final_evaluation(g, model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, device):

    batch_size = 128
    num_per_neigh = 10
    neg_sampling_factor = 1
    L = len(model.layers)

    train_idx, test_idx, val_idx, labels, use_types, ntypes = get_training_targets(g)

    train_idx_fname, test_idx_fname, val_idx_fname = ee_fname.create_idx_pools(train_idx, test_idx, val_idx)
    train_idx_varuse, test_idx_varuse, val_idx_varuse = ee_varuse.create_idx_pools(train_idx, test_idx, val_idx)
    train_idx_apicall, test_idx_apicall, val_idx_apicall = ee_apicall.create_idx_pools(train_idx, test_idx, val_idx)

    loader_fname, test_loader_fname, val_loader_fname = get_loaders(g, train_idx_fname, test_idx_fname, val_idx_fname,
                                                                    num_per_neigh, L,
                                                                    batch_size)
    loader_varuse, test_loader_varuse, val_loader_varuse = get_loaders(g, train_idx_varuse, test_idx_varuse,
                                                                       val_idx_varuse, num_per_neigh, L,
                                                                       batch_size)
    loader_apicall, test_loader_apicall, val_loader_apicall = get_loaders(g, train_idx_apicall, test_idx_apicall,
                                                                          val_idx_apicall, num_per_neigh, L,
                                                                          batch_size)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * L)
    create_apicall_loader = lambda indices: dgl.dataloading.NodeDataLoader(
        g, indices, sampler, batch_size=len(indices), num_workers=0)

    loss, train_acc_fname = evaluate_embedder(model, ee_fname, lp_fname, loader_fname,
                                         use_types, ntypes=ntypes, device=device,
                                         neg_samplig_factor=1)
    _, train_acc_varuse = evaluate_embedder(model, ee_varuse, lp_varuse, loader_varuse,
                                          use_types, ntypes=ntypes, device=device,
                                          neg_samplig_factor=1)
    _, train_acc_apicall = evaluate_nodes(model, ee_apicall, lp_apicall,
                                           create_apicall_loader, loader_apicall,
                                           use_types, ntypes=ntypes, device=device,
                                           neg_samplig_factor=1)

    _, val_acc_fname = evaluate_embedder(model, ee_fname, lp_fname, val_loader_fname,
                                         use_types, ntypes=ntypes, device=device,
                                         neg_samplig_factor=1)
    _, val_acc_varuse = evaluate_embedder(model, ee_varuse, lp_varuse, val_loader_varuse,
                                          use_types, ntypes=ntypes, device=device,
                                          neg_samplig_factor=1)
    _, val_acc_apicall = evaluate_nodes(model, ee_apicall, lp_apicall,
                                           create_apicall_loader, val_loader_apicall,
                                           use_types, ntypes=ntypes, device=device,
                                           neg_samplig_factor=1)

    _, test_acc_fname = evaluate_embedder(model, ee_fname, lp_fname, test_loader_fname,
                                          use_types, ntypes=ntypes, device=device,
                                          neg_samplig_factor=1)
    _, test_acc_varuse = evaluate_embedder(model, ee_varuse, lp_varuse, test_loader_varuse,
                                           use_types, ntypes=ntypes, device=device,
                                           neg_samplig_factor=1)
    _, test_acc_apicall = evaluate_nodes(model, ee_apicall, lp_apicall,
                                            create_apicall_loader, test_loader_apicall,
                                            use_types, ntypes=ntypes, device=device,
                                            neg_samplig_factor=1)
    scores = {
        # "loss": loss.item(),
        "train_acc_fname": train_acc_fname,
        "val_acc_fname": val_acc_fname,
        "test_acc_fname": test_acc_fname,
        "train_acc_varuse": train_acc_varuse,
        "val_acc_varuse": val_acc_varuse,
        "test_acc_varuse": test_acc_varuse,
        "train_acc_apicall": train_acc_apicall,
        "val_acc_apicall": val_acc_apicall,
        "test_acc_apicall": test_acc_apicall,
    }

    print(
        'Final Eval : fname Train Acc %.4f, fname Val Acc %.4f, fname Test Acc %.4f, varuse Train Acc %.4f, varuse Val Acc %.4f, varuse Test Acc %.4f, apicall Train Acc %.4f, apicall Val Acc %.4f, apicall Test Acc %.4f' % (
            # scores["loss"],
            scores["train_acc_fname"],
            scores["val_acc_fname"],
            scores["test_acc_fname"],
            scores["train_acc_varuse"],
            scores["val_acc_varuse"],
            scores["test_acc_varuse"],
            scores["train_acc_apicall"],
            scores["val_acc_apicall"],
            scores["test_acc_apicall"],
        ))

    return scores


def get_training_targets(g):
    if hasattr(g, 'ntypes'):
        ntypes = g.ntypes
        labels = {ntype: g.nodes[ntype].data['labels'] for ntype in g.ntypes}
        use_types = True
        if len(g.ntypes) == 1:
            key = next(iter(labels.keys()))
            labels = labels[key]
            use_types = False
        train_idx = {ntype: torch.nonzero(g.nodes[ntype].data['train_mask']).squeeze() for ntype in g.ntypes}
        test_idx = {ntype: torch.nonzero(g.nodes[ntype].data['test_mask']).squeeze() for ntype in g.ntypes}
        val_idx = {ntype: torch.nonzero(g.nodes[ntype].data['val_mask']).squeeze() for ntype in g.ntypes}
    else:
        ntypes = None
        labels = g.ndata['labels']
        train_idx = g.ndata['train_mask']
        test_idx = g.ndata['test_mask']
        val_idx = g.ndata['val_mask']
        use_types = False

    return train_idx, test_idx, val_idx, labels, use_types, ntypes


def get_loaders(g, train_idx, test_idx, val_idx, num_per_neigh, layers, batch_size):
    # train sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
    loader = dgl.dataloading.NodeDataLoader(
        g, train_idx, sampler, batch_size=batch_size, shuffle=False, num_workers=0)

    # validation sampler
    # we do not use full neighbor to save computation resources
    val_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
    val_loader = dgl.dataloading.NodeDataLoader(
        g, val_idx, val_sampler, batch_size=batch_size, shuffle=False, num_workers=0)

    # we do not use full neighbor to save computation resources
    test_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
    test_loader = dgl.dataloading.NodeDataLoader(
        g, test_idx, test_sampler, batch_size=batch_size, shuffle=False, num_workers=0)

    return loader, test_loader, val_loader

def idx_len(idx):
    if isinstance(idx, dict):
        length = 0
        for key in idx:
            length += len(idx[key])
    else:
        length = len(idx)
    return length

def train(g, model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, epochs, device, lr):
    """
    Training procedure for the model with node classifier.
    :param model:
    :param g_labels:
    :param splits:
    :param epochs:
    :return:
    """

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters()},
            {'params': ee_fname.parameters()},
            {'params': ee_varuse.parameters()},
            {'params': ee_apicall.parameters()},
            {'params': lp_fname.parameters()},
            {'params': lp_varuse.parameters()},
            {'params': lp_apicall.parameters()},
        ], lr=lr
    )

    best_val_acc_fname = 0.
    best_test_acc_fname = 0.
    best_val_acc_varuse = 0.
    best_test_acc_varuse = 0.
    best_val_acc_apicall = 0.
    best_test_acc_apicall = 0.

    ref_batch_size = 128
    num_per_neigh = 10
    neg_samplig_factor = 3
    L = len(model.layers)

    train_idx, test_idx, val_idx, _, use_types, ntypes = get_training_targets(g)

    train_idx_fname, test_idx_fname, val_idx_fname = ee_fname.create_idx_pools(train_idx, test_idx, val_idx)
    train_idx_varuse, test_idx_varuse, val_idx_varuse = ee_varuse.create_idx_pools(train_idx, test_idx, val_idx)
    train_idx_apicall, test_idx_apicall, val_idx_apicall = ee_apicall.create_idx_pools(train_idx, test_idx, val_idx)

    print(f"Pool sizes : train {idx_len(train_idx_fname)}, test {idx_len(test_idx_fname)}, val {idx_len(val_idx_fname)}")
    print(f"Pool sizes : train {idx_len(train_idx_varuse)}, test {idx_len(test_idx_varuse)}, val {idx_len(val_idx_varuse)}")
    print(f"Pool sizes : train {idx_len(train_idx_apicall)}, test {idx_len(test_idx_apicall)}, val {idx_len(val_idx_apicall)}")

    batch_size_fname = ref_batch_size
    batch_size_varuse = ceil(idx_len(train_idx_varuse) / ceil(idx_len(train_idx_fname) / batch_size_fname))
    batch_size_apicall = ceil(idx_len(train_idx_apicall) / ceil(idx_len(train_idx_fname) / batch_size_fname))

    loader_fname, test_loader_fname, val_loader_fname = get_loaders(g, train_idx_fname, test_idx_fname, val_idx_fname, num_per_neigh, L,
                                                                    batch_size_fname)
    loader_varuse, test_loader_varuse, val_loader_varuse = get_loaders(g, train_idx_varuse, test_idx_varuse, val_idx_varuse, num_per_neigh, L,
                                                                    batch_size_varuse)
    loader_apicall, test_loader_apicall, val_loader_apicall = get_loaders(g, train_idx_apicall, test_idx_apicall, val_idx_apicall, num_per_neigh, L,
                                                                    batch_size_apicall)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * L)
    create_apicall_loader = lambda indices: dgl.dataloading.NodeDataLoader(
        g, indices, sampler, batch_size=len(indices), num_workers=0)

    for epoch in range(epochs):

        start = time()

        for i, ((input_nodes_fname, seeds_fname, blocks_fname),
                (input_nodes_varuse, seeds_varuse, blocks_varuse),
                (input_nodes_apicall, seeds_apicall, blocks_apicall)) in \
                enumerate(zip(loader_fname, loader_varuse, loader_apicall)):

            blocks_fname = [blk.to(device) for blk in blocks_fname]
            blocks_varuse = [blk.to(device) for blk in blocks_varuse]
            blocks_apicall = [blk.to(device) for blk in blocks_apicall]

            src_embs_fname = logits_batch(model, input_nodes_fname, blocks_fname, use_types, ntypes)
            logits_fname, labels_fname = logits_embedder(src_embs_fname, ee_fname, lp_fname, seeds_fname, neg_samplig_factor, device=device)

            src_embs_varuse = logits_batch(model, input_nodes_varuse, blocks_varuse, use_types, ntypes)
            logits_varuse, labels_varuse = logits_embedder(src_embs_varuse, ee_varuse, lp_varuse, seeds_varuse, neg_samplig_factor, device=device)

            src_embs_apicall = logits_batch(model, input_nodes_apicall, blocks_apicall, use_types, ntypes)
            logits_apicall, labels_apicall = logits_nodes(model, src_embs_apicall,
                                                          ee_apicall, lp_apicall, create_apicall_loader,
                                                          seeds_apicall, use_types, ntypes, neg_samplig_factor, device=device)

            # TODO
            # some issues are possible because of the lack of softmax
            train_acc_fname = torch.sum(logits_fname.argmax(dim=1) == labels_fname).item() / len(labels_fname)
            train_acc_varuse = torch.sum(logits_varuse.argmax(dim=1) == labels_varuse).item() / len(labels_varuse)
            train_acc_apicall = torch.sum(logits_apicall.argmax(dim=1) == labels_apicall).item() / len(labels_apicall)

            train_logits = torch.cat([logits_fname, logits_varuse, logits_apicall], 0)
            train_labels = torch.cat([labels_fname, labels_varuse, labels_apicall], 0)

            logp = nn.functional.log_softmax(train_logits, 1)
            loss = nn.functional.nll_loss(logp, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, val_acc_fname = evaluate_embedder(model, ee_fname, lp_fname, val_loader_fname,
                                             use_types, ntypes=ntypes, device=device,
                                             neg_samplig_factor=neg_samplig_factor)
        _, val_acc_varuse = evaluate_embedder(model, ee_varuse, lp_varuse, val_loader_varuse,
                                             use_types, ntypes=ntypes, device=device,
                                             neg_samplig_factor=neg_samplig_factor)
        _, val_acc_apicall = evaluate_nodes(model, ee_apicall, lp_apicall,
                                               create_apicall_loader, val_loader_apicall,
                                               use_types, ntypes=ntypes, device=device, neg_samplig_factor=neg_samplig_factor)

        _, test_acc_fname = evaluate_embedder(model, ee_fname, lp_fname, test_loader_fname,
                                             use_types, ntypes=ntypes, device=device,
                                             neg_samplig_factor=neg_samplig_factor)
        _, test_acc_varuse = evaluate_embedder(model, ee_varuse, lp_varuse, test_loader_varuse,
                                              use_types, ntypes=ntypes, device=device,
                                              neg_samplig_factor=neg_samplig_factor)
        _, test_acc_apicall = evaluate_nodes(model, ee_apicall, lp_apicall,
                                               create_apicall_loader, test_loader_apicall,
                                               use_types, ntypes=ntypes, device=device,
                                               neg_samplig_factor=neg_samplig_factor)
        end = time()

        track_best_multitask(epoch, loss.item(), train_acc_fname, val_acc_fname, test_acc_fname,
                             train_acc_varuse, val_acc_varuse, test_acc_varuse,
                             train_acc_apicall, val_acc_apicall, test_acc_apicall,
                             best_val_acc_fname, best_test_acc_fname,
                             best_val_acc_varuse, best_test_acc_varuse,
                             best_val_acc_apicall, best_test_acc_apicall,
                             time=end-start)

        torch.save({
            'm': model.state_dict(),
            'ee_fname': ee_fname.state_dict(),
            'ee_varuse': ee_varuse.state_dict(),
            'ee_apicall': ee_apicall.state_dict(),
            "lp_fname": lp_fname.state_dict(),
            "lp_varuse": lp_varuse.state_dict(),
            "lp_apicall": lp_apicall.state_dict(),
            "epoch": epoch
        }, "saved_state.pt")

    # return {
    #     "loss": loss.item(),
    #     "train_acc": train_acc,
    #     "val_acc": val_acc,
    #     "test_acc": test_acc,
    # }


def training_procedure(dataset, model, params, EPOCHS, args):
    NODE_EMB_SIZE = 100
    ELEM_EMB_SIZE = 100

    device = 'cpu'
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        device = 'cuda:%d' % args.gpu

    g = dataset.g#.to(device)

    lr = params.pop('lr')

    m = model(g, num_classes=NODE_EMB_SIZE,
              **params).to(device)

    ee_fname = create_elem_embedder(args.fname_file, dataset.nodes, ELEM_EMB_SIZE, True).to(device)
    ee_varuse = create_elem_embedder(args.varuse_file, dataset.nodes, ELEM_EMB_SIZE, True).to(device)
    ee_apicall = create_elem_embedder(args.call_seq_file, dataset.nodes, ELEM_EMB_SIZE, False).to(device)

    lp_fname = LinkPredictor(ee_fname.emb_size + m.emb_size).to(device)
    lp_varuse = LinkPredictor(ee_varuse.emb_size + m.emb_size).to(device)
    lp_apicall = LinkPredictor(m.emb_size + m.emb_size).to(device)

    if args.restore_state:
        checkpoint = torch.load("saved_state.pt")
        m.load_state_dict(checkpoint['m'])
        ee_fname.load_state_dict(checkpoint['ee_fname'])
        ee_varuse.load_state_dict(checkpoint['ee_varuse'])
        ee_apicall.load_state_dict(checkpoint['ee_apicall'])
        lp_fname.load_state_dict(checkpoint['lp_fname'])
        lp_varuse.load_state_dict(checkpoint['lp_fname'])
        lp_apicall.load_state_dict(checkpoint['lp_apicall'])
        print(f"Restored from epoch {checkpoint['epoch']}")
        checkpoint = None

    try:
        train(g, m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, EPOCHS, device, lr)
    except KeyboardInterrupt:
        print("Training interrupted")
    except:
        raise Exception()

    m.eval()
    ee_fname.eval()
    ee_varuse.eval()
    ee_apicall.eval()
    lp_fname.eval()
    lp_varuse.eval()
    lp_apicall.eval()
    scores = final_evaluation(dataset.g, m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, device)

    return m.to('cpu'), ee_fname.to('cpu'), ee_varuse.to('cpu'), ee_apicall.to('cpu'), \
           lp_fname.to('cpu'), lp_varuse.to('cpu'), lp_apicall.to('cpu'), scores
