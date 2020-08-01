import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import numpy as np
import pandas as pd
import pickle
from utils import get_num_batches

from ElementEmbedder import ElementEmbedder
from LinkPredictor import LinkPredictor

def evaluate_no_classes(logits, labels):
    pred = logits.argmax(1)
    acc = (pred == labels).float().mean()
    return acc


def track_best(epoch, loss, train_acc, val_acc, test_acc, best_val_acc, best_test_acc):
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


def prepare_batch_no_classes(node_embeddings: torch.Tensor,
                             elem_embeder: ElementEmbedder,
                             link_predictor: LinkPredictor,
                             indices: np.array,
                             batch_size: int,
                             negative_factor: int):
    """
    Creates a batch using graph embeddings. ElementEmbedder return indices of the nodes that are adjacent to the nodes
    given by indices.
    :param node_embeddings: torch Tensor that contains embeddings for nodes
    :param elem_embeder: object that stores mapping from nodes to their neighbors give bu "call_next" edges
    :param link_predictor: Simple classifier to predict the presence of a link
    :param indices: indices in the current batch
    :param batch_size:
    :param negative_factor: by what what factor there should be more negative samples than positive
    :return:
    """
    K = negative_factor

    # TODO
    # batch size seems to be redundant

    # get embeddings for nodes in the current batch
    node_embeddings_batch = node_embeddings[indices]
    next_call_indices = elem_embeder[indices]
    next_call_embeddings = node_embeddings[next_call_indices]
    positive_batch = torch.cat([node_embeddings_batch, next_call_embeddings], 1)
    labels_pos = torch.ones(batch_size, dtype=torch.long)

    node_embeddings_neg_batch = node_embeddings_batch.repeat(K, 1)
    negative_indices = elem_embeder.sample_negative(batch_size * K) # embeddings are sampled from 3/4 unigram distribution
    negative_random = node_embeddings[negative_indices]
    negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
    labels_neg = torch.zeros(batch_size * K, dtype=torch.long)

    batch = torch.cat([positive_batch, negative_batch], 0)
    labels = torch.cat([labels_pos, labels_neg], 0)

    logits = link_predictor(batch)

    return logits, labels


def final_evaluation_no_classes(model, elem_embeder, link_predictor, splits):
    train_idx, test_idx, val_idx = splits

    pool = set(elem_embeder.elements['id'].to_list())
    train_idx = np.fromiter(pool.intersection(train_idx), dtype=np.int64)
    test_idx = np.fromiter(pool.intersection(test_idx), dtype=np.int64)
    val_idx = np.fromiter(pool.intersection(val_idx), dtype=np.int64)

    node_embeddings = model()
    subsample = np.random.choice(train_idx, size=1000, replace=False)
    train_logits, train_labels = prepare_batch_no_classes(node_embeddings,
                                                          elem_embeder,
                                                          link_predictor,
                                                          subsample,
                                                          subsample.size,
                                                          1)

    test_logits, test_labels = prepare_batch_no_classes(node_embeddings,
                                                        elem_embeder,
                                                        link_predictor,
                                                        test_idx,
                                                        test_idx.size,
                                                        1)

    val_logits, val_labels = prepare_batch_no_classes(node_embeddings,
                                                      elem_embeder,
                                                      link_predictor,
                                                      val_idx,
                                                      val_idx.size,
                                                      1)

    train_acc, val_acc, test_acc = evaluate_no_classes(train_logits, train_labels), \
                                   evaluate_no_classes(test_logits, test_labels), \
                                   evaluate_no_classes(val_logits, val_labels)

    logp = nn.functional.log_softmax(train_logits, 1)
    loss = nn.functional.nll_loss(logp, train_labels)

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


def train_no_classes(model, elem_embeder, link_predictor, splits, epochs):
    # there should be no (significant) leak of training signal from the train to test set. the src nodes appear
    # either in train or in test set. If an node A is from train set, node B is from test set, and C is a common target,
    # then edge A->C is used for training, B->C used for testing. But in future experiments embedding for C is trained
    # from scratch, o there should be no leak
    train_idx, test_idx, val_idx = splits

    pool = set(elem_embeder.elements['id'].to_list())

    print(f"Original set sizes: train {train_idx.size} test {test_idx.size} val {val_idx.size}")

    train_idx = np.fromiter(pool.intersection(train_idx), dtype=np.int64)
    test_idx = np.fromiter(pool.intersection(test_idx), dtype=np.int64)
    val_idx = np.fromiter(pool.intersection(val_idx), dtype=np.int64)

    # train_idx = np.array(list(filter(lambda x: x in pool, train_idx)), dtype=np.int64)
    # test_idx = np.array(list(filter(lambda x: x in pool, test_idx)), dtype=np.int64)
    # val_idx = np.array(list(filter(lambda x: x in pool, val_idx)), dtype=np.int64)

    print(f"Set sizes after filtering: train {train_idx.size} test {test_idx.size} val {val_idx.size}")

    # this is heldout because it was not used during training
    heldout_idx = test_idx.tolist() + val_idx.tolist()

    optimizer = torch.optim.Adagrad(
        [
            {'params': model.parameters(), 'lr': 1e-2},
            {'params': link_predictor.parameters(), 'lr': 1e-2}
        ], lr=0.01)

    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)

    batch_size = 4096
    K = 3  # negative oversampling factor

    num_batches, batch_size = get_num_batches(len(elem_embeder), batch_size)

    for epoch in range(epochs):

        # since we train in batches, we need to iterate over the nodes
        # since indexes are sampled randomly, it is a little bit hard to make sure we cover all data
        # instead, we sample nodes the same number of times that there are different nodes in the dataset,
        # hoping to cover all the data

        # num_batches = get_num_batches(len(elem_embeder), batch_size)
        # num_batches = len(elem_embeder) // batch_size + 1 # +1 when len(elem_embeder) < batch_size
        # if len(elem_embeder) < batch_size:
        #     batch_size = len(elem_embeder)

        for batch_ind in range(num_batches):
            node_embeddings = model()

            random_batch = np.random.choice(train_idx, size=batch_size)


            train_logits, train_labels = prepare_batch_no_classes(node_embeddings,
                                                                  elem_embeder,
                                                                  link_predictor,
                                                                  random_batch,
                                                                  batch_size,
                                                                  K)

            train_acc = evaluate_no_classes(train_logits, train_labels)

            logp = nn.functional.log_softmax(train_logits, 1)
            loss = nn.functional.nll_loss(logp, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_ind % 1 == 0:
                print("\r%d/%d batches complete, acc: %.4f / %.4f" % (batch_ind, num_batches, train_acc.item(), np.average(train_labels.numpy())), end="\n")


        test_logits, test_labels = prepare_batch_no_classes(node_embeddings,
                                                            elem_embeder,
                                                            link_predictor,
                                                            test_idx,
                                                            test_idx.size,
                                                            1)

        val_logits, val_labels = prepare_batch_no_classes(node_embeddings,
                                                          elem_embeder,
                                                          link_predictor,
                                                          val_idx,
                                                          val_idx.size,
                                                          1)

        test_acc, val_acc = evaluate_no_classes(test_logits, test_labels), \
                            evaluate_no_classes(val_logits, val_labels)

        # print(np.average(test_labels.numpy()), np.average(val_labels.numpy()))
        track_best(epoch, loss, train_acc, val_acc, test_acc, best_val_acc, best_test_acc)
        # pickle.dump(node_embeddings.detach().numpy(), open("nodes.pkl", "wb"))
        # elem_embeder.elements.to_csv("edges.csv", index=False)
        torch.save({
            'm': model.state_dict(),
            'ee': elem_embeder.state_dict(),
            "lp": link_predictor.state_dict(),
            "epoch": epoch
        }, "saved_state.pt")

def training_procedure(dataset, model, params, EPOCHS, call_seq_file, restore_state):
    NODE_EMB_SIZE = 100
    ELEM_EMB_SIZE = 100

    m = model(dataset.g,
              num_classes=NODE_EMB_SIZE,
              **params)

    # get data for models with large number of classes, possibly several labels for
    # a single input id


    element_data = pd.read_csv(call_seq_file)
    orig_shape = len(element_data)
    function2nodeid = dict(zip(dataset.nodes['id'].values, dataset.nodes['global_graph_id'].values))
    # some nodes can disappear after preliminary filtering
    # use NA to mark such nides and filter them from the training data
    element_data['id'] = element_data['src'].apply(lambda x: function2nodeid.get(x, pd.NA))
    element_data['dst'] = element_data['dst'].apply(lambda x: function2nodeid.get(x, pd.NA))
    element_data.dropna(axis=0, inplace=True)
    element_data.drop_duplicates(['id', 'dst'], inplace=True, ignore_index=True)

    print("Droppped {} after preprocessing target edges".format(orig_shape - len(element_data)))

    # element_data.to_csv("id2graphid.csv", index=False)
    # import sys; sys.exit()
    # element_data['dst'] = element_data['name'].apply(lambda name: name.split(".")[-1])
    from ElementEmbedder import ElementEmbedder
    ee = ElementEmbedder(element_data, ELEM_EMB_SIZE, compact_dst=False)

    from LinkPredictor import LinkPredictor
    lp = LinkPredictor(ee.emb_size + m.emb_size)

    if restore_state:
        checkpoint = torch.load("saved_state.pt")
        m.load_state_dict(checkpoint['m'])
        ee.load_state_dict(checkpoint['ee'])
        lp.load_state_dict(checkpoint['lp'])
        print(f"Restored from epoch {checkpoint['epoch']}")
        checkpoint = None

    try:
        train_no_classes(m, ee, lp, dataset.splits, EPOCHS)
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        m.eval()
        ee.eval()
        lp.eval()
        scores = final_evaluation_no_classes(m, ee, lp, dataset.splits)

    return  m, ee, lp, scores