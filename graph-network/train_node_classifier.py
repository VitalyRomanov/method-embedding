import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import numpy as np
from utils import create_idx_pools, evaluate_no_classes

def evaluate(logits, labels, train_idx, test_idx, val_idx):

    pred = logits.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    return train_acc, val_acc, test_acc

def final_evaluation(model, g_labels, splits):
    # train_idx, test_idx, val_idx = splits #get_train_test_val_indices(g_labels)
    # labels = torch.tensor(g_labels)

    if isinstance(g_labels, tuple):
        sparse_labels = True
    else:
        sparse_labels = False

    if sparse_labels:
        idx, lbls = g_labels

        pool_fname = set(idx)
        train_idx, test_idx, val_idx = create_idx_pools(splits, pool_fname)

        train_lbls, test_lbls, val_lbls = get_split_lables(idx, lbls, train_idx, test_idx, val_idx)

        train_lbls = torch.LongTensor(train_lbls)
        test_lbls = torch.LongTensor(test_lbls)
        val_lbls = torch.LongTensor(val_lbls)
    else:
        train_idx, test_idx, val_idx = splits #get_train_test_val_indices(g_labels)
        labels = torch.LongTensor(g_labels)
        train_lbls = labels[train_idx]
        test_lbls = labels[test_idx]
        val_lbls = labels[val_idx]

    logits = model()
    # evaluate(logits, labels, train_idx, test_idx, val_idx)
    logp = nn.functional.log_softmax(logits, 1)
    # loss = nn.functional.nll_loss(logp[train_idx], labels[train_idx])
    loss = nn.functional.nll_loss(logp[train_idx], train_lbls)

    # train_acc, val_acc, test_acc = evaluate(logits, labels, train_idx, test_idx, val_idx)
    train_acc = evaluate_no_classes(logits[train_idx], train_lbls)
    val_acc = evaluate_no_classes(logits[val_idx], val_lbls)
    test_acc = evaluate_no_classes(logits[test_idx], test_lbls)

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


def get_split_lables(idx, lbls, train_idx, test_idx, val_idx):
    idx = idx.tolist()
    train_idx = train_idx.tolist()
    test_idx = test_idx.tolist()
    val_idx = val_idx.tolist()

    train_idx = np.fromiter(map(lambda x: idx.index(x), train_idx), dtype=np.int32)
    test_idx = np.fromiter(map(lambda x: idx.index(x), test_idx), dtype=np.int32)
    val_idx = np.fromiter(map(lambda x: idx.index(x), val_idx), dtype=np.int32)

    return lbls[train_idx], lbls[test_idx], lbls[val_idx]

def train(model, g_labels, splits, epochs):
    """
    Training procedure for the model with node classifier.
    :param model:
    :param g_labels:
    :param splits:
    :param epochs:
    :return:
    """

    if isinstance(g_labels, tuple):
        sparse_labels = True
    else:
        sparse_labels = False

    if sparse_labels:
        idx, lbls = g_labels

        pool_fname = set(idx)
        train_idx, test_idx, val_idx = create_idx_pools(splits, pool_fname)

        train_lbls, test_lbls, val_lbls = get_split_lables(idx, lbls, train_idx, test_idx, val_idx)

        train_lbls = torch.LongTensor(train_lbls)
        test_lbls = torch.LongTensor(test_lbls)
        val_lbls = torch.LongTensor(val_lbls)
    else:
        train_idx, test_idx, val_idx = splits #get_train_test_val_indices(g_labels)
        labels = torch.LongTensor(g_labels)
        train_lbls = labels[train_idx]
        test_lbls = labels[test_idx]
        val_lbls = labels[val_idx]

    heldout_idx = test_idx.tolist() + val_idx.tolist()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)

    for epoch in range(epochs):
        logits = model()

        # if sparse_labels:
        train_acc = evaluate_no_classes(logits[train_idx], train_lbls)
        val_acc = evaluate_no_classes(logits[val_idx], val_lbls)
        test_acc = evaluate_no_classes(logits[test_idx], test_lbls)
        # else:
        #     train_acc, val_acc, test_acc = evaluate(logits, labels, train_idx, test_idx, val_idx)

        # pred = logits.argmax(1)
        # train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
        # val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        # test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

        logp = nn.functional.log_softmax(logits, 1)
        # if sparse_labels:
        loss = nn.functional.nll_loss(logp[train_idx], train_lbls)
        # else:
        #     loss = nn.functional.nll_loss(logp[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        print('Epoch %d, Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            epoch,
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))

        torch.save({
            'm': model.state_dict(),
            "epoch": epoch
        }, "saved_state.pt")

    return heldout_idx

def training_procedure(dataset, model, params, EPOCHS, args):

    if args.override_labels:
        import pandas as pd
        from Dataset import compact_property
        labels = pd.read_csv(args.data_file).astype({'src': "int32", "dst": "str"})

        function2nodeid = dict(zip(dataset.nodes['id'].values, dataset.nodes['global_graph_id'].values))
        lbl2id = compact_property(labels['dst'])

        idx = np.fromiter(map(lambda x: function2nodeid[x], labels['src'].values), dtype=np.int32)
        lbl = np.fromiter(map(lambda x:lbl2id[x], labels['dst']), dtype=np.int32)

        labels = (idx, lbl)
        uniq_classes = len(lbl2id)
    else:
        labels = dataset.labels
        uniq_classes = dataset.num_classes


    m = model(dataset.g,
              num_classes=uniq_classes,
              # produce_logits=True,
              **params)

    if args.restore_state:
        checkpoint = torch.load("saved_state.pt")
        m.load_state_dict(checkpoint['m'])
        print(f"Restored from epoch {checkpoint['epoch']}")
        checkpoint = None

    # try:
    train(m, labels, dataset.splits, EPOCHS)
    # except KeyboardInterrupt:
    #     print("Training interrupted")
    # finally:
    #     m.eval()
    #     scores = final_evaluation(m, labels, dataset.splits)
    #
    # return m, scores