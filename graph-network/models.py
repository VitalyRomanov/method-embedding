import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

# https://docs.dgl.ai/tutorials/hetero/1_basics.html#working-with-heterogeneous-graphs-in-dgl

from gat import GAT
from rgcn_hetero import EntityClassify

from data import get_train_test_val_indices

def final_evaluation(model, g_labels):
    train_idx, test_idx, val_idx = get_train_test_val_indices(g_labels)
    labels = torch.tensor(g_labels)

    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)

    logits = model()
    logp = nn.functional.log_softmax(logits, 1)
    loss = nn.functional.nll_loss(logp[train_idx], labels[train_idx])

    pred = logits.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    print('Final Eval Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
        loss.item(),
        train_acc.item(),
        val_acc.item(),
        best_val_acc.item(),
        test_acc.item(),
        best_test_acc.item(),
    ))

    return {
        "loss": loss.item(),
        "train_acc": train_acc.item(),
        "val_acc": val_acc.item(),
        "best_val_acc": best_val_acc.item(),
        "test_acc": test_acc.item(),
        "best_test_acc": best_test_acc.item(),
    }

