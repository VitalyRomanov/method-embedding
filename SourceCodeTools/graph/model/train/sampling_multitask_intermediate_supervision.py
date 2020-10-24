import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
import dgl
from math import ceil
from time import time
from os.path import join

from SourceCodeTools.graph.model.train.utils import create_elem_embedder, track_best_multitask, create_idx_pools
from SourceCodeTools.graph.model.ElementEmbedder import ElementEmbedder
from SourceCodeTools.graph.model.LinkPredictor import LinkPredictor


from SourceCodeTools.graph.model.train.sampling_multitask import extract_embed, logits_embedder, logits_nodes, idx_len, get_loaders, get_training_targets


def logits_batch(model, input_nodes, blocks, use_types, ntypes):
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

    layers = model(emb, blocks, return_all=True)

    def merge_logits(logits, use_types):

        cumm_logits = []

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

    return [merge_logits(l, use_types) for l in layers]


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

        logits_labels_fname = [logits_embedder(logitsfname, ee_fname, lpfname, seeds,
                                               neg_samplig_factor, device=device)
                               for logitsfname, lpfname in zip(src_embs, lp_fname)]

        logits_labels_varuse = [logits_embedder(logitsvaruse, ee_varuse, lpvaruse, seeds,
                                                neg_samplig_factor, device=device)
                                for logitsvaruse, lpvaruse in zip(src_embs, lp_varuse)]

        logits_labels_apicall = [logits_nodes(model, logitsapicall,
                                              ee_apicall, lpapicall, create_apicall_loader,
                                              seeds, use_types, ntypes, neg_samplig_factor,
                                              device=device)
                                 for logitsapicall, lpapicall in zip(src_embs, lp_apicall)]

        logits_fname, labels_fname = logits_labels_fname[-1]
        logits_varuse, labels_varuse = logits_labels_varuse[-1]
        logits_apicall, labels_apicall = logits_labels_apicall[-1]
        # logits_fname, labels_fname = logits_embedder(src_embs, ee_fname, lp_fname, seeds, neg_samplig_factor, device=device)
        # logits_varuse, labels_varuse = logits_embedder(src_embs, ee_varuse, lp_varuse, seeds, neg_samplig_factor, device=device)
        # logits_apicall, labels_apicall = logits_nodes(model, src_embs,
        #                                               ee_apicall, lp_apicall, create_apicall_loader,
        #                                               seeds, use_types, ntypes, neg_samplig_factor, device=device)

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
        # logits, labels = logits_embedder(src_embs, ee, lp, seeds, neg_samplig_factor, device=device)
        logits_labels = [logits_embedder(logits_, ee, lp_, seeds,
                                               neg_samplig_factor, device=device)
                               for logits_, lp_ in zip(src_embs, lp)]

        logits, labels = logits_labels[-1]

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
        # logits, labels = logits_nodes(model, src_embs,
        #                                   ee, lp, create_apicall_loader,
        #                                   seeds, use_types, ntypes, neg_samplig_factor, device=device)
        logits_labels = [logits_nodes(model, logits_,
                                              ee, lp_, create_apicall_loader,
                                              seeds, use_types, ntypes, neg_samplig_factor,
                                              device=device)
                                 for logits_, lp_ in zip(src_embs, lp)]

        logits, labels = logits_labels[-1]

        logp = nn.functional.log_softmax(logits, 1)
        loss = nn.functional.cross_entropy(logp, labels)
        acc = torch.sum(logp.argmax(dim=1) == labels).item() / len(labels)

        total_loss += loss.item()
        total_acc += acc
        count += 1
    return total_loss / count, total_acc / count


def final_evaluation(g, model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, device, batch_size):

    # batch_size = 128
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


def get_seed_nodes(layers, blocks, seeds):

    seed_layers = []

    for layer, block in zip(layers, blocks):
        ids = block.dstdata["_ID"]
        ids_ = ids.tolist()
        seeds_ = seeds.tolist()
        locations = torch.LongTensor([ids_.index(seed) for seed in seeds_])
        # TODO
        # assert -1 not in locations
        seed_layers.append(layer[locations])

    return seed_layers


def train(MODEL_BASE, g, model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, epochs, device, lr, ref_batch_size):
    """
    Training procedure for the model with node classifier.
    :param model:
    :param g_labels:
    :param splits:
    :param epochs:
    :return:
    """

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    adam_params = [
        {'params': model.parameters()},
        {'params': ee_fname.parameters()},
        {'params': ee_varuse.parameters()},
        {'params': ee_apicall.parameters()},
    ]
    adam_params.extend({'params': lp.parameters()} for lp in lp_fname)
    adam_params.extend({'params': lp.parameters()} for lp in lp_varuse)
    adam_params.extend({'params': lp.parameters()} for lp in lp_apicall)

    optimizer = torch.optim.Adam(
        # [
        #     {'params': model.parameters()},
        #     {'params': ee_fname.parameters()},
        #     {'params': ee_varuse.parameters()},
        #     {'params': ee_apicall.parameters()},
        #     {'params': lp_fname.parameters()},
        #     {'params': lp_varuse.parameters()},
        #     {'params': lp_apicall.parameters()},
        # ],
        adam_params,
        lr=lr
    )

    lr_scheduler = ExponentialLR(optimizer, gamma=0.991)
    best_val_acc_fname = 0.
    best_test_acc_fname = 0.
    best_val_acc_varuse = 0.
    best_test_acc_varuse = 0.
    best_val_acc_apicall = 0.
    best_test_acc_apicall = 0.

    # ref_batch_size = 128
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

            # TODO
            # can take intermediate representations when self_loops is True
            # make supervision for intermediate representations as well
            blocks_fname = [blk.to(device) for blk in blocks_fname]
            blocks_varuse = [blk.to(device) for blk in blocks_varuse]
            blocks_apicall = [blk.to(device) for blk in blocks_apicall]

            assert all(
                map(lambda x: x in blocks_fname[2].dstdata['_ID'].tolist(), blocks_fname[3].dstdata['_ID'].tolist()))
            assert all(
                map(lambda x: x in blocks_fname[1].dstdata['_ID'].tolist(), blocks_fname[3].dstdata['_ID'].tolist()))

            src_embs_fname = logits_batch(model, input_nodes_fname, blocks_fname, use_types, ntypes)
            src_embs_fname = get_seed_nodes(src_embs_fname, blocks_fname, seeds_fname)
            logits_labels_fname = [logits_embedder(logitsfname, ee_fname, lpfname, seeds_fname,
                                                   neg_samplig_factor, device=device)
                                   for logitsfname, lpfname in zip(src_embs_fname, lp_fname)]

            src_embs_varuse = logits_batch(model, input_nodes_varuse, blocks_varuse, use_types, ntypes)
            src_embs_varuse = get_seed_nodes(src_embs_varuse, blocks_varuse, seeds_varuse)
            logits_labels_varuse = [logits_embedder(logitsvaruse, ee_varuse, lpvaruse, seeds_varuse,
                                                   neg_samplig_factor, device=device)
                                   for logitsvaruse, lpvaruse in zip(src_embs_varuse, lp_varuse)]

            src_embs_apicall = logits_batch(model, input_nodes_apicall, blocks_apicall, use_types, ntypes)
            src_embs_apicall = get_seed_nodes(src_embs_apicall, blocks_apicall, seeds_apicall)
            logits_labels_apicall = [logits_nodes(model, logitsapicall,
                                                          ee_apicall, lpapicall, create_apicall_loader,
                                                          seeds_apicall, use_types, ntypes, neg_samplig_factor,
                                                          device=device)
                                    for logitsapicall, lpapicall in zip(src_embs_apicall, lp_apicall)]

            logits_fname, labels_fname = logits_labels_fname[-1]
            logits_varuse, labels_varuse = logits_labels_varuse[-1]
            logits_apicall, labels_apicall = logits_labels_apicall[-1]

            # TODO
            # some issues are possible because of the lack of softmax
            train_acc_fname = torch.sum(logits_fname.argmax(dim=1) == labels_fname).item() / len(labels_fname)
            train_acc_varuse = torch.sum(logits_varuse.argmax(dim=1) == labels_varuse).item() / len(labels_varuse)
            train_acc_apicall = torch.sum(logits_apicall.argmax(dim=1) == labels_apicall).item() / len(labels_apicall)

            train_logits, train_labels = zip(*logits_labels_fname)
            train_logits = list(train_logits)
            train_labels = list(train_labels)
            temp_logits, temp_labels = zip(*logits_labels_varuse)
            train_logits.extend(temp_logits)
            train_labels.extend(temp_labels)
            temp_logits, temp_labels = zip(*logits_labels_apicall)
            train_logits.extend(temp_logits)
            train_labels.extend(temp_labels)

            train_logits = torch.cat(train_logits, 0)
            train_labels = torch.cat(train_labels, 0)

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
            'lp_fname': [lp.state_dict() for lp in lp_fname],
            'lp_varuse': [lp.state_dict() for lp in lp_varuse],
            'lp_apicall': [lp.state_dict() for lp in lp_apicall],
            "epoch": epoch
        }, join(MODEL_BASE, "saved_state.pt"))

        lr_scheduler.step()


def training_procedure(dataset, model, params, EPOCHS, args, MODEL_BASE):
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

    n_layers = len(m.layers)
    lp_fname = [LinkPredictor(ee_fname.emb_size + m.emb_size).to(device) for _ in range(n_layers)]
    lp_varuse = [LinkPredictor(ee_varuse.emb_size + m.emb_size).to(device) for _ in range(n_layers)]
    lp_apicall = [LinkPredictor(m.emb_size + m.emb_size).to(device) for _ in range(n_layers)]

    if args.restore_state:
        checkpoint = torch.load(join(MODEL_BASE, "saved_state.pt"))
        m.load_state_dict(checkpoint['m'])
        ee_fname.load_state_dict(checkpoint['ee_fname'])
        ee_varuse.load_state_dict(checkpoint['ee_varuse'])
        ee_apicall.load_state_dict(checkpoint['ee_apicall'])
        lp_fname.load_state_dict(checkpoint['lp_fname'])
        lp_varuse.load_state_dict(checkpoint['lp_varuse'])
        lp_apicall.load_state_dict(checkpoint['lp_apicall'])
        print(f"Restored from epoch {checkpoint['epoch']}")
        checkpoint = None

    try:
        train(MODEL_BASE, g, m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, EPOCHS, device, lr, args.batch_size)
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
    scores = final_evaluation(dataset.g, m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, device, args.batch_size)

    return m.to('cpu'), ee_fname.to('cpu'), ee_varuse.to('cpu'), ee_apicall.to('cpu'), \
           lp_fname.to('cpu'), lp_varuse.to('cpu'), lp_apicall.to('cpu'), scores
