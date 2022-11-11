import torch


def compute_accuracy(pred_, true_):
    return torch.sum(pred_ == true_).item() / len(true_)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def get_length_mask(target, lens):
    mask = torch.arange(target.size(1)).to(target.device)[None, :] < lens[:, None]
    return mask