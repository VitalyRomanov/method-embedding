import numpy as np
import pandas as pd
from ElementEmbedder import ElementEmbedder

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


def create_elem_embedder(file_path, nodes, emb_size, compact_dst):
    element_data = pd.read_csv(file_path)
    function2nodeid = dict(zip(nodes['id'].values, nodes['global_graph_id'].values))
    element_data['id'] = element_data['src'].apply(lambda x: function2nodeid.get(x, None))
    if compact_dst is False: # creating api call embedder
        element_data['dst'] = element_data['dst'].apply(lambda x: function2nodeid.get(x, None))
        element_data.drop_duplicates(['id', 'dst'], inplace=True, ignore_index=True)
    element_data = element_data.dropna(axis=0)
    ee = ElementEmbedder(element_data, emb_size, compact_dst=compact_dst)
    return ee