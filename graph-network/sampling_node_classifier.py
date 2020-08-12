import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from dgl.contrib.sampling import NeighborSampler
import dgl
import numpy as np

def evaluate(logits, labels, train_idx, test_idx, val_idx):

    pred = logits.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    return train_acc, val_acc, test_acc

def evaluate_batch(logits, labels):
    return torch.sum(logits == labels).item() / len(labels)

def evaluate_with_batches(train_batches, test_batches, val_batches):
    train_log, train_lab = train_batches
    train_acc = evaluate_batch(torch.cat(train_log), torch.cat(train_lab))

    test_log, test_lab = test_batches
    test_acc = evaluate_batch(torch.cat(test_log), torch.cat(test_lab))

    val_log, val_lab = val_batches
    val_acc = evaluate_batch(torch.cat(val_log), torch.cat(val_lab))

    return train_acc, val_acc, test_acc

def final_evaluation(model, g_labels, splits):
    train_idx, test_idx, val_idx = splits #get_train_test_val_indices(g_labels)
    labels = torch.tensor(g_labels)

    logits = model()
    evaluate(logits, labels, train_idx, test_idx, val_idx)
    logp = nn.functional.log_softmax(logits, 1)
    loss = nn.functional.nll_loss(logp[train_idx], labels[train_idx])

    train_acc, val_acc, test_acc = evaluate(logits, labels, train_idx, test_idx, val_idx)

    scores = {
        "loss": loss.item(),
        "train_acc": train_acc.item(),
        "val_acc": val_acc.item(),
        "test_acc": test_acc.item(),
    }

    print('Final Eval Loss %.4f, Train Acc %.4f, Val Acc %.4f, Test Acc %.4f' % (
        scores["loss"],
        scores["train_acc"],
        scores["val_acc"],
        scores["test_acc"],
    ))

    return scores


def train(dataset, g, model, g_labels, splits, epochs):
    """
    Training procedure for the model with node classifier.
    :param model:
    :param g_labels:
    :param splits:
    :param epochs:
    :return:
    """

    train_idx, test_idx, val_idx = splits #get_train_test_val_indices(g_labels)
    labels = torch.tensor(g_labels)#.cuda()

    heldout_idx = test_idx.tolist() + val_idx.tolist()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)

    batch_size = 128
    num_neighbors = 200
    L = len(model.layers)

    train_sampler = NeighborSampler(g, batch_size,
                                       num_neighbors,
                                       neighbor_type='in',
                                       shuffle=True,
                                       num_hops=L,
                                       seed_nodes=train_idx)

    test_sampler = NeighborSampler(g, batch_size,
                                     num_neighbors,
                                     neighbor_type='in',
                                     shuffle=True,
                                     num_hops=L,
                                     seed_nodes=test_idx)

    val_sampler = NeighborSampler(g, batch_size,
                                     num_neighbors,
                                     neighbor_type='in',
                                     shuffle=True,
                                     num_hops=L,
                                     seed_nodes=val_idx)

    for epoch in range(epochs):
        train_logits = []; train_labels = []
        test_logits = []; test_labels = []
        val_logits = []; val_labels = []

        for batch_ind, nf in enumerate(train_sampler):
            # print(g.ndata['features'][0,:])
            nf.copy_from_parent()
            batch_nids = torch.LongTensor(nf.layer_parent_nid(-1))

            logits = model(nf)

            logp = nn.functional.log_softmax(logits, 1)
            batch_labels = labels[batch_nids]
            loss = nn.functional.nll_loss(logp, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # p = nn.functional.softmax(logits, 1)

            train_labels.append(batch_labels)
            train_logits.append(logp.argmax(dim=1))

            # if batch_ind % 50 == 0:
            #     print("\r%d/%d batches complete, loss: %.4f" % (
            #     batch_ind, 0, loss.item(),), end="\n")

            # nf.copy_to_parent()

        for nf in test_sampler:
            nf.copy_from_parent()

            logits = model(nf)
            batch_nids = torch.LongTensor(nf.layer_parent_nid(-1))

            p = nn.functional.softmax(logits, 1)
            batch_labels = labels[batch_nids]

            test_labels.append(batch_labels)
            test_logits.append(p.argmax(dim=1))

        for nf in val_sampler:
            nf.copy_from_parent()

            logits = model(nf)
            batch_nids = torch.LongTensor(nf.layer_parent_nid(-1))

            p = nn.functional.softmax(logits, 1)
            batch_labels = labels[batch_nids]

            val_labels.append(batch_labels)
            val_logits.append(p.argmax(dim=1))


        train_acc, val_acc, test_acc = evaluate_with_batches((train_logits, train_labels),
                                                             (test_logits, test_labels),
                                                             (val_logits, val_labels))

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        print('Epoch %d, Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            epoch,
            loss.item(),
            train_acc,
            val_acc,
            best_val_acc,
            test_acc,
            best_test_acc,
        ))

        torch.save({
            'm': model.state_dict(),
            'feat': g.ndata['features'],
            "epoch": epoch
        }, "saved_state.pt")

    return heldout_idx

def training_procedure(dataset, model, params, EPOCHS, restore_state):

    dataset.g.ndata['features'] = nn.Parameter(
        torch.Tensor(
            dataset.g.number_of_nodes(), params['in_dim']
        )
    )#.cuda()

    nn.init.kaiming_uniform_(dataset.g.ndata['features'])
    if 'num_classes' in params: params.pop('num_classes') # do I need this?

    dataset.g.readonly(readonly_state=True)

    m = model(dataset.g, num_classes=dataset.num_classes,
              **params)#.cuda()

    if restore_state:
        checkpoint = torch.load("saved_state.pt")
        m.load_state_dict(checkpoint['m'])
        dataset.g.ndata['features'] = checkpoint['features']
        print(f"Restored from epoch {checkpoint['epoch']}")
        checkpoint = None

    try:
        train(dataset, dataset.g, m, dataset.labels, dataset.splits, EPOCHS)
    except KeyboardInterrupt:
        print("Training interrupted")
    except:
        raise Exception()
    finally:
        m.eval()
        m = m#.cpu()
        dataset.g.ndata['features'] = dataset.g.ndata['features']#.cpu()
        scores = final_evaluation(m, dataset.labels, dataset.splits)

    return m, scores
