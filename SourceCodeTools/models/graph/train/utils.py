import logging
import sys

from os.path import join, isdir
from os import mkdir

from SourceCodeTools.models.graph.ElementEmbedder import ElementEmbedder


def get_num_batches(length, batch_size_suggestion):
    batch_size = min(batch_size_suggestion, length)

    num_batches = length // batch_size  # +1 when len(elem_embedder) < batch_size
    return num_batches, batch_size


def get_name(model, timestamp):
    return "{} {}".format(model.__name__, timestamp).replace(":", "-").replace(" ", "-").replace(".", "-")


def get_model_base(args, model_attempt, force_new=False):
    if args.restore_state and not force_new:
        model_base = args.model_output_dir
    else:
        model_base = join(args.model_output_dir, model_attempt)
        if not isdir(model_base):
            mkdir(model_base)

    return model_base


# def create_idx_pools(splits, pool):
#     train_idx, val_idx, test_idx = splits
#     train_idx = np.fromiter(pool.intersection(train_idx), dtype=np.int64)
#     val_idx = np.fromiter(pool.intersection(val_idx), dtype=np.int64)
#     test_idx = np.fromiter(pool.intersection(test_idx), dtype=np.int64)
#     return train_idx, val_idx, test_idx


def evaluate_no_classes(logits, labels):
    pred = logits.argmax(1)
    acc = (pred == labels).float().mean()
    return acc


# def create_elem_embedder(element_data, nodes, emb_size, compact_dst):
#     # element_data = unpersist(file_path)
#
#     if len(element_data) == 0:
#         logging.error(f"Not enough data for the embedder: {len(element_data)}. Exiting...")
#         sys.exit()
#
#     id2nodeid = dict(zip(nodes['id'].tolist(), nodes['global_graph_id'].tolist()))
#     id2typedid = dict(zip(nodes['id'].tolist(), nodes['typed_id'].tolist()))
#     id2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))
#
#     element_data['id'] = element_data['src'].apply(lambda x: id2nodeid.get(x, None))
#     element_data['src_type'] = element_data['src'].apply(lambda x: id2type.get(x, None))
#     element_data['src_typed_id'] = element_data['src'].apply(lambda x: id2typedid.get(x, None))
#     element_data = element_data.astype({
#         'id': 'Int32',
#         'src_type': 'category',
#         'src_typed_id': 'Int32',
#     })
#
#     if compact_dst is False:  # creating api call embedder
#         element_data = element_data.rename({'dst': 'dst_orig'}, axis=1)
#         element_data['dst'] = element_data['dst_orig'].apply(lambda x: id2nodeid.get(x, None))
#         element_data['dst_type'] = element_data['dst_orig'].apply(lambda x: id2type.get(x, None))
#         element_data['dst_typed_id'] = element_data['dst_orig'].apply(lambda x: id2typedid.get(x, None))
#         element_data.drop_duplicates(['id', 'dst'], inplace=True, ignore_index=True)  # this line apparenly filters parallel edges
#         element_data = element_data.astype({
#             'dst': 'Int32',
#             'dst_type': 'category',
#             'dst_typed_id': 'Int32',
#         })
#
#     element_data = element_data.dropna(axis=0)
#     ee = ElementEmbedder(element_data, emb_size, compact_dst=compact_dst)
#     return ee


class BestScoreTracker:
    def __init__(self):
        self.best_val_acc_node_name = 0.
        self.best_test_acc_node_name = 0.
        self.best_val_acc_var_use = 0.
        self.best_test_acc_var_use = 0.
        self.best_val_acc_api_call = 0.
        self.best_test_acc_api_call = 0.

    def track_best(
            self, epoch, loss,
            train_acc_node_name=0., val_acc_node_name=0., test_acc_node_name=0.,
            train_acc_var_use=0., val_acc_var_use=0., test_acc_var_use=0.,
            train_acc_api_call=0., val_acc_api_call=0., test_acc_api_call=0.,
            time=0
    ):
        if val_acc_node_name is not None:
            if self.best_val_acc_node_name < val_acc_node_name:
                self.best_val_acc_node_name = val_acc_node_name
                self.best_test_acc_node_name = test_acc_node_name

        if val_acc_var_use is not None:
            if self.best_val_acc_var_use < val_acc_var_use:
                self.best_val_acc_var_use = val_acc_var_use
                self.best_test_acc_var_use = test_acc_var_use

        if val_acc_api_call is not None:
            if self.best_val_acc_api_call < val_acc_api_call:
                self.best_val_acc_api_call = val_acc_api_call
                self.best_test_acc_api_call = test_acc_api_call

        epoch_info = "'Epoch %d, Time: %d s, Loss %.4f, " % (
            epoch, time, loss
        )

        score_info = \
            'node name Train Acc %.4f, node name Val Acc %.4f (Best %.4f), node name Test Acc %.4f (Best %.4f), ' \
            'var use Train Acc %.4f, var use Val Acc %.4f (Best %.4f), var use Test Acc %.4f (Best %.4f), ' \
            'api call Train Acc %.4f, api call Val Acc %.4f (Best %.4f), api call Test Acc %.4f (Best %.4f)' % (
                train_acc_node_name, val_acc_node_name,
                self.best_val_acc_node_name, test_acc_node_name, self.best_test_acc_node_name,
                train_acc_var_use, val_acc_var_use,
                self.best_val_acc_var_use, test_acc_var_use, self.best_test_acc_var_use,
                train_acc_api_call, val_acc_api_call,
                self.best_val_acc_api_call, test_acc_api_call, self.best_test_acc_api_call,
            )

        logging.info(epoch_info + score_info)
