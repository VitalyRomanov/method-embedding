import torch

def _compute_accuracy(pred_, true_):
    return torch.sum(pred_ == true_).item() / len(true_)