import numpy as np
import pandas as pd
from os.path import join, isdir
from os import mkdir

from SourceCodeTools.graph.model.ElementEmbedder import ElementEmbedder

def get_num_batches(length, batch_size_suggestion):
    batch_size = min(batch_size_suggestion, length)

    num_batches = length // batch_size  # +1 when len(elem_embeder) < batch_size
    return num_batches, batch_size


def get_name(model, timestamp):
    return "{} {}".format(model.__name__, timestamp).replace(":", "-").replace(" ", "-").replace(".", "-")

def get_model_base(args, model_attempt):
    if args.restore_state:
        MODEL_BASE = args.model_output_dir
    else:
        MODEL_BASE = join(args.model_output_dir, model_attempt)
        if not isdir(MODEL_BASE):
            mkdir(MODEL_BASE)

    return MODEL_BASE



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

    id2nodeid = dict(zip(nodes['id'].tolist(), nodes['global_graph_id'].tolist()))
    id2typedid = dict(zip(nodes['id'].tolist(), nodes['typed_id'].tolist()))
    id2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))

    element_data['id'] = element_data['src'].apply(lambda x: id2nodeid.get(x, None))
    element_data['src_type'] = element_data['src'].apply(lambda x: id2type.get(x, None))
    element_data['src_typed_id'] = element_data['src'].apply(lambda x: id2typedid.get(x, None))
    element_data = element_data.astype({
        'id': 'Int32',
        'src_type': 'Int32',
        'src_typed_id': 'Int32',
    })

    if compact_dst is False: # creating api call embedder
        element_data = element_data.rename({'dst': 'dst_orig'}, axis=1)
        element_data['dst'] = element_data['dst_orig'].apply(lambda x: id2nodeid.get(x, None))
        element_data['dst_type'] = element_data['dst_orig'].apply(lambda x: id2type.get(x, None))
        element_data['dst_typed_id'] = element_data['dst_orig'].apply(lambda x: id2typedid.get(x, None))
        element_data.drop_duplicates(['id', 'dst'], inplace=True, ignore_index=True) # this line apparenly filters parallel edges
        element_data = element_data.astype({
            'dst': 'Int32',
            'dst_type': 'Int32',
            'dst_typed_id': 'Int32',
        })

    element_data = element_data.dropna(axis=0)
    ee = ElementEmbedder(element_data, emb_size, compact_dst=compact_dst)
    return ee


def track_best_multitask(epoch, loss,
               train_acc_fname, val_acc_fname, test_acc_fname,
               train_acc_varuse, val_acc_varuse, test_acc_varuse,
               train_acc_apicall, val_acc_apicall, test_acc_apicall,
               best_val_acc_fname, best_test_acc_fname,
               best_val_acc_varuse, best_test_acc_varuse,
               best_val_acc_apicall, best_test_acc_apicall, time=0):
    # TODO
    # does not really track now
    if best_val_acc_fname < val_acc_fname:
        best_val_acc_fname = val_acc_fname
        best_test_acc_fname = test_acc_fname

    if best_val_acc_varuse < val_acc_varuse:
        best_val_acc_varuse = val_acc_varuse
        best_test_acc_varuse = test_acc_varuse

    if best_val_acc_apicall < val_acc_apicall:
        best_val_acc_apicall = val_acc_apicall
        best_test_acc_apicall = test_acc_apicall

    print(
        'Epoch %d, Time: %d s, Loss %.4f, fname Train Acc %.4f, fname Val Acc %.4f (Best %.4f), fname Test Acc %.4f (Best %.4f), varuse Train Acc %.4f, varuse Val Acc %.4f (Best %.4f), varuse Test Acc %.4f (Best %.4f), apicall Train Acc %.4f, apicall Val Acc %.4f (Best %.4f), apicall Test Acc %.4f (Best %.4f)' % (
            epoch,
            time,
            loss,
            train_acc_fname,
            val_acc_fname,
            best_val_acc_fname,
            test_acc_fname,
            best_test_acc_fname,
            train_acc_varuse,
            val_acc_varuse,
            best_val_acc_varuse,
            test_acc_varuse,
            best_test_acc_varuse,
            train_acc_apicall,
            val_acc_apicall,
            best_val_acc_apicall,
            test_acc_apicall,
            best_test_acc_apicall,
        ))