import numpy as np

def get_num_batches(length, batch_size_suggestion):
    batch_size = min(batch_size_suggestion, length)

    num_batches = length // batch_size  # +1 when len(elem_embeder) < batch_size
    return num_batches, batch_size


def get_name(model, timestamp):
    return "{} {}".format(model.__name__, timestamp).replace(":", "-").replace(" ", "-").replace(".", "-")

def create_idx_pools(splits, pool):
    train_idx, test_idx, val_idx = splits
    train_idx = np.fromiter(pool.intersection(train_idx), dtype=np.int64)
    test_idx = np.fromiter(pool.intersection(test_idx), dtype=np.int64)
    val_idx = np.fromiter(pool.intersection(val_idx), dtype=np.int64)
    return train_idx, test_idx, val_idx

def evaluate_no_classes(logits, labels):
    pred = logits.argmax(1)
    acc = (pred == labels).float().mean()
    return acc