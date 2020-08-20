import torch
import torch.nn as nn
import dgl

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


def logits_embedder(node_embeddings, elem_embeder, link_predictor, seeds, negative_factor=1):
    K = negative_factor
    indices = seeds
    batch_size = len(seeds)

    node_embeddings_batch = node_embeddings
    element_embeddings = elem_embeder(elem_embeder[indices])

    positive_batch = torch.cat([node_embeddings_batch, element_embeddings], 1)
    labels_pos = torch.ones(batch_size, dtype=torch.long)

    node_embeddings_neg_batch = node_embeddings_batch.repeat(K, 1)
    negative_indices = torch.LongTensor(elem_embeder.sample_negative(batch_size * K))
    negative_random = elem_embeder(negative_indices)
    negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
    labels_neg = torch.zeros(batch_size * K, dtype=torch.long)

    batch = torch.cat([positive_batch, negative_batch], 0)
    labels = torch.cat([labels_pos, labels_neg], 0)

    logits = link_predictor(batch)

    return logits, labels


def logits_nodes(model, node_embeddings,
                 elem_embeder, link_predictor, create_dataloader,
                 src_seeds, use_types, ntypes, negative_factor=1):
    K = negative_factor
    indices = src_seeds
    batch_size = len(src_seeds)

    node_embeddings_batch = node_embeddings
    next_call_indices = elem_embeder[indices]

    dataloader = create_dataloader(next_call_indices)
    input_nodes, dst_seeds, blocks = next(iter(dataloader))
    assert dst_seeds.shape == next_call_indices.shape
    assert dst_seeds == next_call_indices
    next_call_embeddings = logits_batch(model, input_nodes, blocks, use_types, ntypes)
    positive_batch = torch.cat([node_embeddings_batch, next_call_embeddings], 1)
    labels_pos = torch.ones(batch_size, dtype=torch.long)

    node_embeddings_neg_batch = node_embeddings_batch.repeat(K, 1)
    negative_indices = elem_embeder.sample_negative(
        batch_size * K)  # embeddings are sampled from 3/4 unigram distribution

    dataloader = create_dataloader(negative_indices)
    input_nodes, dst_seeds, blocks = next(iter(dataloader))
    assert dst_seeds.shape == negative_indices.shape
    assert dst_seeds == negative_indices
    negative_random = logits_batch(model, input_nodes, blocks, use_types, ntypes)
    negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
    labels_neg = torch.zeros(batch_size * K, dtype=torch.long)

    batch = torch.cat([positive_batch, negative_batch], 0)
    labels = torch.cat([labels_pos, labels_neg], 0)

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
        logits_fname, labels_fname = logits_embedder(src_embs, ee_fname, lp_fname, seeds, neg_samplig_factor)
        logits_varuse, labels_varuse = logits_embedder(src_embs, ee_varuse, lp_varuse, seeds, neg_samplig_factor)
        logits_apicall, labels_apicall = logits_nodes(model, src_embs,
                                                      ee_apicall, lp_apicall, create_apicall_loader,
                                                      seeds, use_types, ntypes, neg_samplig_factor)

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


def final_evaluation(g, model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, gpu):
    device = 'cpu'
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu)
        device = 'cuda:%d' % gpu

    batch_size = 4096
    num_per_neigh = 4
    neg_sampling_factor = 1
    L = len(model.layers)

    train_idx, test_idx, val_idx, labels, use_types, ntypes = get_training_targets(g)

    loader, test_loader, val_loader = get_loaders(g, train_idx, test_idx, val_idx, num_per_neigh, L, batch_size)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * L)
    create_apicall_loader = lambda indices: dgl.dataloading.NodeDataLoader(
        g, indices, sampler, batch_size=len(indices), num_workers=4)

    loss, train_acc_fname, train_acc_varuse, train_acc_apicall = evaluate(model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall,
                               create_apicall_loader, loader, use_types, ntypes=ntypes, device=device, neg_samplig_factor=neg_sampling_factor)
    _, val_acc_fname, val_acc_varuse, val_acc_apicall = evaluate(model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall,
                          create_apicall_loader, val_loader, use_types, ntypes=ntypes, device=device, neg_samplig_factor=neg_sampling_factor)
    _, test_acc_fname, test_acc_varuse, test_acc_apicall = evaluate(model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall,
                           create_apicall_loader, test_loader, use_types, ntypes=ntypes, device=device, neg_samplig_factor=neg_sampling_factor)

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
        g, train_idx, sampler, batch_size=batch_size, shuffle=True, num_workers=0)

    # validation sampler
    # we do not use full neighbor to save computation resources
    val_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
    val_loader = dgl.dataloading.NodeDataLoader(
        g, val_idx, val_sampler, batch_size=batch_size, shuffle=True, num_workers=0)

    # we do not use full neighbor to save computation resources
    test_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
    test_loader = dgl.dataloading.NodeDataLoader(
        g, test_idx, test_sampler, batch_size=batch_size, shuffle=True, num_workers=0)

    return loader, test_loader, val_loader


def train(g, model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, epochs, gpu, lr):
    """
    Training procedure for the model with node classifier.
    :param model:
    :param g_labels:
    :param splits:
    :param epochs:
    :return:
    """
    pool_fname = set(ee_fname.elements['id'].to_list())
    pool_varuse = set(ee_varuse.elements['id'].to_list())
    pool_apicall = set(ee_apicall.elements['id'].to_list())

    device = 'cpu'
    # use_cuda = gpu >= 0 and torch.cuda.is_available()
    # if use_cuda:
    #     torch.cuda.set_device(gpu)
    #     device = 'cuda:%d' % gpu

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc_fname = 0.
    best_test_acc_fname = 0.
    best_val_acc_varuse = 0.
    best_test_acc_varuse = 0.
    best_val_acc_apicall = 0.
    best_test_acc_apicall = 0.

    batch_size = 4096
    num_per_neigh = 4
    neg_samplig_factor = 3
    L = len(model.layers)

    train_idx, test_idx, val_idx, _, use_types, ntypes = get_training_targets(g)

    train_idx_fname, test_idx_fname, val_idx_fname = create_idx_pools((train_idx, test_idx, val_idx), pool_fname)
    train_idx_varuse, test_idx_varuse, val_idx_varuse = create_idx_pools((train_idx, test_idx, val_idx), pool_varuse)
    train_idx_apicall, test_idx_apicall, val_idx_apicall = create_idx_pools((train_idx, test_idx, val_idx), pool_apicall)

    print(f"Pool sizes : train {len(train_idx_fname)}, test {len(test_idx_fname)}, val {len(val_idx_fname)}")
    print(f"Pool sizes : train {len(train_idx_varuse)}, test {len(test_idx_varuse)}, val {len(val_idx_varuse)}")
    print(f"Pool sizes : train {len(train_idx_apicall)}, test {len(test_idx_apicall)}, val {len(val_idx_apicall)}")

    loader_fname, test_loader_fname, val_loader_fname = get_loaders(g, train_idx_fname, test_idx_fname, val_idx_fname, num_per_neigh, L,
                                                                    batch_size)
    loader_varuse, test_loader_varuse, val_loader_varuse = get_loaders(g, train_idx_varuse, test_idx_varuse, val_idx_varuse, num_per_neigh, L,
                                                                    batch_size)
    loader_apicall, test_loader_apicall, val_loader_apicall = get_loaders(g, train_idx_apicall, test_idx_apicall, val_idx_apicall, num_per_neigh, L,
                                                                    batch_size)

    sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * L)
    create_apicall_loader = lambda indices: dgl.dataloading.NodeDataLoader(
        g, indices, sampler, batch_size=len(indices), num_workers=4)

    for epoch in range(epochs):
        for i, (input_nodes, seeds, blocks) in enumerate(loader):
            src_embs = logits_batch(model, input_nodes, blocks, use_types, ntypes)

            logits_fname, labels_fname = logits_embedder(src_embs, ee_fname, lp_fname, seeds, neg_samplig_factor)
            logits_varuse, labels_varuse = logits_embedder(src_embs, ee_varuse, lp_varuse, seeds, neg_samplig_factor)
            logits_apicall, labels_apicall = logits_nodes(model, src_embs,
                                                          ee_apicall, lp_apicall, create_apicall_loader,
                                                          seeds, use_types, ntypes, neg_samplig_factor)

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

        _, val_acc_fname, val_acc_varuse, val_acc_apicall = \
            evaluate(model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall,
                     create_apicall_loader, val_loader, use_types, ntypes=ntypes, device=device, neg_samplig_factor=neg_samplig_factor)
        _, test_acc_fname, test_acc_varuse, test_acc_apicall = evaluate(model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall,
                     create_apicall_loader, test_loader, use_types, ntypes=ntypes, device=device, neg_samplig_factor=neg_samplig_factor)

        track_best_multitask(epoch, loss.item(), train_acc_fname, val_acc_fname, test_acc_fname,
                             train_acc_varuse, val_acc_varuse, test_acc_varuse,
                             train_acc_apicall, val_acc_apicall, test_acc_apicall,
                             best_val_acc_fname, best_test_acc_fname,
                             best_val_acc_varuse, best_test_acc_varuse,
                             best_val_acc_apicall, best_test_acc_apicall)

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

    lr = params.pop('lr')

    m = model(dataset.g, num_classes=NODE_EMB_SIZE,
              **params)  # .cuda()

    ee_fname = create_elem_embedder(args.fname_file, dataset.nodes, ELEM_EMB_SIZE, True)
    ee_varuse = create_elem_embedder(args.varuse_file, dataset.nodes, ELEM_EMB_SIZE, True)
    ee_apicall = create_elem_embedder(args.call_seq_file, dataset.nodes, ELEM_EMB_SIZE, False)

    lp_fname = LinkPredictor(ee_fname.emb_size + m.emb_size)
    lp_varuse = LinkPredictor(ee_varuse.emb_size + m.emb_size)
    lp_apicall = LinkPredictor(m.emb_size + m.emb_size)

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
        train(dataset.g, m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, EPOCHS, args.gpu, lr)
    except KeyboardInterrupt:
        print("Training interrupted")
    except:
        raise Exception()
    finally:
        m.eval()
        ee_fname.eval()
        ee_varuse.eval()
        ee_apicall.eval()
        lp_fname.eval()
        lp_varuse.eval()
        lp_apicall.eval()
        scores = final_evaluation(dataset.g, m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, args.gpu)

    return m, scores
