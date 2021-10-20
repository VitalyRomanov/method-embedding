import torch


def compute_accuracy(pred_, true_):
    return torch.sum(pred_ == true_).item() / len(true_)