# import torch
# import torch.nn as nn
# import dgl
# from os.path import join
#
#
# def extract_embed(node_embed, input_nodes):
#     emb = {}
#     for ntype, nid in input_nodes.items():
#         nid = input_nodes[ntype]
#         emb[ntype] = node_embed[ntype][nid]
#     return emb
#
#
# def logits_batch(model, input_nodes, blocks, use_types, ntypes):
#     cumm_logits = []
#
#     if use_types:
#         emb = extract_embed(model.node_embed(), input_nodes)
#     else:
#         if ntypes is not None:
#             # single node type
#             key = next(iter(ntypes))
#             input_nodes = {key: input_nodes}
#             emb = extract_embed(model.node_embed(), input_nodes)
#         else:
#             emb = model.node_embed()[input_nodes]
#
#     logits = model(emb, blocks)
#
#     if use_types:
#         for ntype in ntypes:
#
#             logits_ = logits.get(ntype, None)
#             if logits_ is None: continue
#
#             cumm_logits.append(logits_)
#     else:
#         if ntypes is not None:
#             # single node type
#             key = next(iter(ntypes))
#             logits_ = logits[key]
#         else:
#             logits_ = logits
#
#         cumm_logits.append(logits_)
#
#     return torch.cat(cumm_logits)
#
#
# def labels_batch(seeds, labels, use_types, ntypes):
#     cumm_labels = []
#
#     if use_types:
#         for ntype in ntypes:
#             cumm_labels.append(labels[ntype][seeds[ntype]])
#     else:
#         cumm_labels.append(labels[seeds])
#
#     return torch.cat(cumm_labels)
#
#
# def evaluate(model, loader, labels, use_types, device, ntypes=None):
#     # model.eval()
#     total_loss = 0
#     total_acc = 0
#     count = 0
#
#     for input_nodes, seeds, blocks in loader:
#         blocks = [blk.to(device) for blk in blocks]
#
#         logits = logits_batch(model, input_nodes, blocks, use_types, ntypes)
#         lbls = labels_batch(seeds, labels, use_types, ntypes)
#
#         loss = nn.functional.cross_entropy(logits, lbls)
#         acc = torch.sum(logits.argmax(dim=1) == lbls).item() / len(lbls)
#
#         total_loss += loss.item()
#         total_acc += acc
#         count += 1
#     return total_loss / count, total_acc / count
#
#
# def final_evaluation(g, model, gpu):
#     device = 'cpu'
#     use_cuda = gpu >= 0 and torch.cuda.is_available()
#     if use_cuda:
#         torch.cuda.set_device(gpu)
#         device = 'cuda:%d' % gpu
#
#     batch_size = 4096
#     num_per_neigh = 4
#     L = len(model.layers)
#
#     train_idx, test_idx, val_idx, labels, use_types, ntypes = get_training_targets(g)
#
#     loader, test_loader, val_loader = get_loaders(g, train_idx, test_idx, val_idx, num_per_neigh, L, batch_size)
#
#     loss, train_acc = evaluate(model, loader, labels, use_types, device, ntypes)
#     _, val_acc = evaluate(model, val_loader, labels, use_types, device, ntypes)
#     _, test_acc = evaluate(model, test_loader, labels, use_types, device, ntypes)
#
#     scores = {
#         "loss": loss,
#         "train_acc": train_acc,
#         "val_acc": val_acc,
#         "test_acc": test_acc,
#     }
#
#     print('Final Eval Loss %.4f, Train Acc %.4f, Val Acc %.4f, Test Acc %.4f' % (
#         scores["loss"],
#         scores["train_acc"],
#         scores["val_acc"],
#         scores["test_acc"],
#     ))
#
#     return scores
#
#
# def get_training_targets(g):
#     if hasattr(g, 'ntypes'):
#         ntypes = g.ntypes
#         labels = {ntype: g.nodes[ntype].data['labels'] for ntype in g.ntypes}
#         use_types = True
#         if len(g.ntypes) == 1:
#             key = next(iter(labels.keys()))
#             labels = labels[key]
#             use_types = False
#         train_idx = {ntype: torch.nonzero(g.nodes[ntype].data['train_mask']).squeeze() for ntype in g.ntypes}
#         test_idx = {ntype: torch.nonzero(g.nodes[ntype].data['test_mask']).squeeze() for ntype in g.ntypes}
#         val_idx = {ntype: torch.nonzero(g.nodes[ntype].data['val_mask']).squeeze() for ntype in g.ntypes}
#     else:
#         ntypes = None
#         labels = g.ndata['labels']
#         train_idx = g.ndata['train_mask']
#         test_idx = g.ndata['test_mask']
#         val_idx = g.ndata['val_mask']
#         use_types = False
#
#     return train_idx, test_idx, val_idx, labels, use_types, ntypes
#
#
# def get_loaders(g, train_idx, test_idx, val_idx, num_per_neigh, layers, batch_size):
#     # train sampler
#     sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
#     loader = dgl.dataloading.NodeDataLoader(
#         g, train_idx, sampler, batch_size=batch_size, shuffle=True, num_workers=0)
#
#     # validation sampler
#     # we do not use full neighbor to save computation resources
#     val_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
#     val_loader = dgl.dataloading.NodeDataLoader(
#         g, val_idx, val_sampler, batch_size=batch_size, shuffle=True, num_workers=0)
#
#     # we do not use full neighbor to save computation resources
#     test_sampler = dgl.dataloading.MultiLayerNeighborSampler([num_per_neigh] * layers)
#     test_loader = dgl.dataloading.NodeDataLoader(
#         g, test_idx, test_sampler, batch_size=batch_size, shuffle=True, num_workers=0)
#
#     return loader, test_loader, val_loader
#
#
# def train(MODEL_BASE, g, model, epochs, gpu, lr):
#     """
#     Training procedure for the model with node classifier.
#     :param model:
#     :param g_labels:
#     :param splits:
#     :param epochs:
#     :return:
#     """
#
#     device = 'cpu'
#     use_cuda = gpu >= 0 and torch.cuda.is_available()
#     if use_cuda:
#         torch.cuda.set_device(gpu)
#         device = 'cuda:%d' % gpu
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     best_val_acc = torch.tensor(0)
#     best_test_acc = torch.tensor(0)
#
#     batch_size = 4096
#     num_per_neigh = 4
#     L = len(model.layers)
#
#     train_idx, test_idx, val_idx, labels, use_types, ntypes = get_training_targets(g)
#
#     loader, test_loader, val_loader = get_loaders(g, train_idx, test_idx, val_idx, num_per_neigh, L, batch_size)
#
#     for epoch in range(epochs):
#         for i, (input_nodes, seeds, blocks) in enumerate(loader):
#             logits = logits_batch(model, input_nodes, blocks, use_types, ntypes)
#             lbls = labels_batch(seeds, labels, use_types, ntypes)
#
#             loss = nn.functional.cross_entropy(logits, lbls)
#             train_acc = torch.sum(logits.argmax(dim=1) == lbls).item() / len(lbls)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         _, val_acc = evaluate(model, val_loader, labels, use_types, device, ntypes)
#         _, test_acc = evaluate(model, test_loader, labels, use_types, device, ntypes)
#
#         if best_val_acc < val_acc:
#             best_val_acc = val_acc
#             best_test_acc = test_acc
#
#         print('Epoch %d, Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
#             epoch, loss.item(), train_acc,
#             val_acc, best_val_acc,
#             test_acc, best_test_acc,
#         ))
#
#         torch.save({
#             'm': model.state_dict(),
#             "epoch": epoch
#         }, join(MODEL_BASE, "saved_state.pt"))
#
#     return {
#         "loss": loss.item(),
#         "train_acc": train_acc,
#         "val_acc": val_acc,
#         "test_acc": test_acc,
#     }
#
#
# def training_procedure(dataset, model, params, EPOCHS, args, MODEL_BASE):
#     lr = params.pop('lr')
#
#     m = model(dataset.g, num_classes=dataset.num_classes,
#               **params)  # .cuda()
#
#     if args.restore_state:
#         checkpoint = torch.load(join(MODEL_BASE, "saved_state.pt"))
#         m.load_state_dict(checkpoint['m'])
#         dataset.g.ndata['features'] = checkpoint['features']
#         print(f"Restored from epoch {checkpoint['epoch']}")
#         checkpoint = None
#
#     try:
#         train(MODEL_BASE, dataset.g, m, EPOCHS, args.gpu, lr)
#     except KeyboardInterrupt:
#         print("Training interrupted")
#     except:
#         raise Exception()
#     finally:
#         m.eval()
#         scores = final_evaluation(dataset.g, m, args.gpu)
#
#     return m, scores
