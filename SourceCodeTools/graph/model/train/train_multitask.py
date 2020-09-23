import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
import numpy as np
import pandas as pd
from SourceCodeTools.mltools.torch.RAdam import RAdam
from SourceCodeTools.graph.model.train.utils import get_num_batches, create_idx_pools, create_elem_embedder

from SourceCodeTools.graph.model.ElementEmbedder import ElementEmbedder
from SourceCodeTools.graph.model.LinkPredictor import LinkPredictor

from os.path import join

def evaluate_no_classes(logits, labels):
    pred = logits.argmax(1)
    acc = (pred == labels).float().mean()
    return acc


def track_best(epoch, loss,
               train_acc_fname, val_acc_fname, test_acc_fname,
               train_acc_varuse, val_acc_varuse, test_acc_varuse,
               train_acc_apicall, val_acc_apicall, test_acc_apicall,
               best_val_acc_fname, best_test_acc_fname,
               best_val_acc_varuse, best_test_acc_varuse,
               best_val_acc_apicall, best_test_acc_apicall):
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
        'Epoch %d, Loss %.4f, fname Train Acc %.4f, fname Val Acc %.4f (Best %.4f), fname Test Acc %.4f (Best %.4f), varuse Train Acc %.4f, varuse Val Acc %.4f (Best %.4f), varuse Test Acc %.4f (Best %.4f), apicall Train Acc %.4f, apicall Val Acc %.4f (Best %.4f), apicall Test Acc %.4f (Best %.4f)' % (
            epoch,
            loss.item(),
            train_acc_fname.item(),
            val_acc_fname.item(),
            best_val_acc_fname.item(),
            test_acc_fname.item(),
            best_test_acc_fname.item(),
            train_acc_varuse.item(),
            val_acc_varuse.item(),
            best_val_acc_varuse.item(),
            test_acc_varuse.item(),
            best_test_acc_varuse.item(),
            train_acc_apicall.item(),
            val_acc_apicall.item(),
            best_val_acc_apicall.item(),
            test_acc_apicall.item(),
            best_test_acc_apicall.item(),
        ))

def prepare_batch_with_nodes(node_embeddings: torch.Tensor,
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

def prepare_batch_with_embeder(node_embeddings,
                             elem_embeder,
                             link_predictor,
                             indices,
                             batch_size,
                             negative_factor):
    K = negative_factor

    element_embeddings = elem_embeder(elem_embeder[indices])
    node_embeddings_batch = node_embeddings[indices]
    positive_batch = torch.cat([node_embeddings_batch, element_embeddings], 1)
    labels_pos = torch.ones(batch_size, dtype=torch.long)

    node_embeddings_neg_batch = node_embeddings_batch.repeat(K, 1)
    negative_indices = torch.LongTensor(elem_embeder.sample_negative(batch_size * K))
    negative_random = elem_embeder(negative_indices)
    negative_batch = torch.cat([node_embeddings_neg_batch, negative_random], 1)
    labels_neg = torch.zeros(batch_size * K, dtype=torch.long)

    batch = torch.cat([positive_batch, negative_batch], 0)
    labels = torch.cat([labels_pos, labels_neg], 0)

    logits = link_predictor(batch)

    return logits, labels


def final_evaluation_no_classes(model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, splits):
    pool_fname = set(ee_fname.elements['id'].to_list())
    pool_varuse = set(ee_varuse.elements['id'].to_list())
    pool_apicall = set(ee_apicall.elements['id'].to_list())

    train_idx_fname, test_idx_fname, val_idx_fname = create_idx_pools(splits, pool_fname)
    train_idx_varuse, test_idx_varuse, val_idx_varuse = create_idx_pools(splits, pool_varuse)
    train_idx_apicall, test_idx_apicall, val_idx_apicall = create_idx_pools(splits, pool_apicall)

    node_embeddings = model()
    random_batch_fname = np.random.choice(train_idx_fname, size=1000)
    random_batch_varuse = np.random.choice(train_idx_varuse, size=1000)
    random_batch_apicall = np.random.choice(train_idx_apicall, size=1000)

    train_logits_fname, train_labels_fname = prepare_batch_with_embeder(node_embeddings,
                                                                      ee_fname,
                                                                      lp_fname,
                                                                      random_batch_fname,
                                                                      random_batch_fname.size,
                                                                      1)

    train_logits_varuse, train_labels_varuse = prepare_batch_with_embeder(node_embeddings,
                                                                        ee_varuse,
                                                                        lp_varuse,
                                                                        random_batch_varuse,
                                                                        random_batch_varuse.size,
                                                                        1)

    train_logits_apicall, train_labels_apicall = prepare_batch_with_nodes(node_embeddings,
                                                                          ee_apicall,
                                                                          lp_apicall,
                                                                          random_batch_apicall,
                                                                          random_batch_apicall.size,
                                                                          1)

    test_logits_fname, test_labels_fname = prepare_batch_with_embeder(node_embeddings,
                                                                    ee_fname,
                                                                    lp_fname,
                                                                    test_idx_fname,
                                                                    test_idx_fname.size,
                                                                    1)
    test_logits_varuse, test_labels_varuse = prepare_batch_with_embeder(node_embeddings,
                                                                      ee_varuse,
                                                                      lp_varuse,
                                                                      test_idx_varuse,
                                                                      test_idx_varuse.size,
                                                                      1)
    test_logits_apicall, test_labels_apicall = prepare_batch_with_nodes(node_embeddings,
                                                                        ee_apicall,
                                                                        lp_apicall,
                                                                        test_idx_apicall,
                                                                        test_idx_apicall.size,
                                                                        1)

    val_logits_fname, val_labels_fname = prepare_batch_with_embeder(node_embeddings,
                                                                  ee_fname,
                                                                  lp_fname,
                                                                  val_idx_fname,
                                                                  val_idx_fname.size,
                                                                  1)
    val_logits_varuse, val_labels_varuse = prepare_batch_with_embeder(node_embeddings,
                                                                    ee_varuse,
                                                                    lp_varuse,
                                                                    val_idx_varuse,
                                                                    val_idx_varuse.size,
                                                                    1)
    val_logits_apicall, val_labels_apicall = prepare_batch_with_nodes(node_embeddings,
                                                                      ee_apicall,
                                                                      lp_apicall,
                                                                      val_idx_apicall,
                                                                      val_idx_apicall.size,
                                                                      1)

    train_acc_fname, train_acc_varuse, train_acc_apicall = evaluate_no_classes(train_logits_fname, train_labels_fname), \
        evaluate_no_classes(train_logits_varuse, train_labels_varuse), \
        evaluate_no_classes(train_logits_apicall, train_labels_apicall), \

    if len(test_logits_fname) > 0:
        test_acc_fname, test_acc_varuse, test_acc_apicall = evaluate_no_classes(test_logits_fname, test_labels_fname), \
            evaluate_no_classes(test_logits_varuse, test_labels_varuse), \
            evaluate_no_classes(test_logits_apicall, test_labels_apicall)
    else:
        test_acc_fname, test_acc_varuse, test_acc_apicall = torch.tensor(0), torch.tensor(0), torch.tensor(0)

    if len(val_logits_fname) > 0:
        val_acc_fname, val_acc_varuse, val_acc_apicall = evaluate_no_classes(val_logits_fname, val_labels_fname), \
            evaluate_no_classes(val_logits_varuse, val_labels_varuse), \
            evaluate_no_classes(val_logits_apicall, val_labels_apicall)
    else:
        val_acc_fname, val_acc_varuse, val_acc_apicall = torch.tensor(0), torch.tensor(0), torch.tensor(0)

    # logp = nn.functional.log_softmax(train_logits, 1)
    # loss = nn.functional.nll_loss(logp, train_labels)

    scores = {
        # "loss": loss.item(),
        "train_acc_fname": train_acc_fname.item(),
        "val_acc_fname": val_acc_fname.item(),
        "test_acc_fname": test_acc_fname.item(),
        "train_acc_varuse": train_acc_varuse.item(),
        "val_acc_varuse": val_acc_varuse.item(),
        "test_acc_varuse": test_acc_varuse.item(),
        "train_acc_apicall": train_acc_apicall.item(),
        "val_acc_apicall": val_acc_apicall.item(),
        "test_acc_apicall": test_acc_apicall.item(),
    }

    print(
        'Final Eval : fname Train Acc %.4f, fname Val Acc %.4f, fname Test Acc %.4f, varuse Train Acc %.4f, varuse Val Acc %.4f, varuse Test Acc %.4f, apicall Train Acc %.4f, apicall Val Acc %.4f, apicall Test Acc %.4f' % (
            # scores["loss"],
            scores["train_acc_fname"],
            scores["val_acc_fname"],
            scores["test_acc_fname"],
            scores["train_acc_varuse"],
            scores["val_acc_varuse"],
            scores["test_acc_varuse"],
            scores["train_acc_apicall"],
            scores["val_acc_apicall"],
            scores["test_acc_apicall"],
        ))

    return scores





def train(MODEL_BASE, model, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, splits, epochs):
    pool_fname = set(ee_fname.elements['id'].to_list())
    pool_varuse = set(ee_varuse.elements['id'].to_list())
    pool_apicall = set(ee_apicall.elements['id'].to_list())

    train_idx_fname, test_idx_fname, val_idx_fname = create_idx_pools(splits, pool_fname)
    train_idx_varuse, test_idx_varuse, val_idx_varuse = create_idx_pools(splits, pool_varuse)
    train_idx_apicall, test_idx_apicall, val_idx_apicall = create_idx_pools(splits, pool_apicall)

    print(train_idx_fname.size, test_idx_fname.size, val_idx_fname.size)
    print(train_idx_varuse.size, test_idx_varuse.size, val_idx_varuse.size)
    print(train_idx_apicall.size, test_idx_apicall.size, val_idx_apicall.size)

    # this is heldout because it was not used during training
    # heldout_idx = test_idx.tolist() + val_idx.tolist()
    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters()},
            {'params': ee_fname.parameters()},
            {'params': ee_varuse.parameters()},
            {'params': ee_apicall.parameters()},
            {'params': lp_fname.parameters()},
            {'params': lp_varuse.parameters()},
            {'params': lp_apicall.parameters()},
        ], lr=0.01
    )
    # optimizer = torch.optim.Adagrad(
    #     [
    #         {'params': model.parameters(), 'lr': 1e-1},
    #         {'params': ee_fname.parameters(), 'lr': 1e-1},
    #         {'params': ee_varuse.parameters(), 'lr': 1e-1},
    #         {'params': ee_apicall.parameters(), 'lr': 1e-1},
    #         {'params': lp_fname.parameters(), 'lr': 1e-2},
    #         {'params': lp_varuse.parameters(), 'lr': 1e-2},
    #         {'params': lp_apicall.parameters(), 'lr': 1e-2},
    #     ], lr=0.01)
    # optimizer = RAdam(
    #     [
    #         {'params': model.parameters(),},
    #         {'params': ee_fname.parameters(),},
    #         {'params': ee_varuse.parameters(),},
    #         {'params': ee_apicall.parameters(),},
    #         {'params': lp_fname.parameters(),},
    #         {'params': lp_varuse.parameters(),},
    #         {'params': lp_apicall.parameters(),},
    #     ], lr=0.01)

    best_val_acc_fname = torch.tensor(0)
    best_test_acc_fname = torch.tensor(0)
    best_val_acc_varuse = torch.tensor(0)
    best_test_acc_varuse = torch.tensor(0)
    best_val_acc_apicall = torch.tensor(0)
    best_test_acc_apicall = torch.tensor(0)

    batch_size = 4096
    K = 3  # negative oversampling factor

    num_batches, batch_size = get_num_batches(len(ee_fname), batch_size)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.005, epochs=epochs, steps_per_epoch=num_batches,
    #                                     pct_start=0.3, div_factor=100., final_div_factor=100.)

    for epoch in range(epochs):

        # since we train in batches, we need to iterate over the nodes
        # since indexes are sampled randomly, it is a little bit hard to make sure we cover all data
        # instead, we sample nodes the same number of times that there are different nodes in the dataset,
        # hoping to cover all the data
        # num_batches = get_num_batches(len(ee_fname), batch_size)
        # num_batches = len(ee_fname) // batch_size

        for batch_ind in range(num_batches):
            node_embeddings = model()

            random_batch_fname = np.random.choice(train_idx_fname, size=batch_size)
            random_batch_varuse = np.random.choice(train_idx_varuse, size=batch_size)
            random_batch_apicall = np.random.choice(train_idx_apicall, size=batch_size)

            train_logits_fname, train_labels_fname = prepare_batch_with_embeder(node_embeddings,
                                                                              ee_fname,
                                                                              lp_fname,
                                                                              random_batch_fname,
                                                                              batch_size,
                                                                              K)

            train_logits_varuse, train_labels_varuse = prepare_batch_with_embeder(node_embeddings,
                                                                                ee_varuse,
                                                                                lp_varuse,
                                                                                random_batch_varuse,
                                                                                batch_size,
                                                                                K)

            train_logits_apicall, train_labels_apicall = prepare_batch_with_nodes(node_embeddings,
                                                                                  ee_apicall,
                                                                                  lp_apicall,
                                                                                  random_batch_apicall,
                                                                                  batch_size,
                                                                                  K)

            train_acc_fname = evaluate_no_classes(train_logits_fname, train_labels_fname)
            train_acc_varuse = evaluate_no_classes(train_logits_varuse, train_labels_varuse)
            train_acc_apicall = evaluate_no_classes(train_logits_apicall, train_labels_apicall)

            train_logits = torch.cat([train_logits_fname, train_logits_varuse, train_logits_apicall], 0)
            train_labels = torch.cat([train_labels_fname, train_labels_varuse, train_labels_apicall], 0)

            logp = nn.functional.log_softmax(train_logits, 1)
            loss = nn.functional.nll_loss(logp, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            if batch_ind % 1 == 0:
                print("\r%d/%d batches complete, loss: %.4f, fname acc: %.4f, varuse acc: %.4f, apicall acc: %.4f" % (
                batch_ind, num_batches, loss.item(), train_acc_fname.item(),
                train_acc_varuse.item(), train_acc_apicall.item()), end="\n")

        test_logits_fname, test_labels_fname = prepare_batch_with_embeder(node_embeddings,
                                                                        ee_fname,
                                                                        lp_fname,
                                                                        test_idx_fname,
                                                                        test_idx_fname.size,
                                                                        1)
        test_logits_varuse, test_labels_varuse = prepare_batch_with_embeder(node_embeddings,
                                                                          ee_varuse,
                                                                          lp_varuse,
                                                                          test_idx_varuse,
                                                                          test_idx_varuse.size,
                                                                          1)
        test_logits_apicall, test_labels_apicall = prepare_batch_with_nodes(node_embeddings,
                                                                            ee_apicall,
                                                                            lp_apicall,
                                                                            test_idx_apicall,
                                                                            test_idx_apicall.size,
                                                                            1)

        val_logits_fname, val_labels_fname = prepare_batch_with_embeder(node_embeddings,
                                                                      ee_fname,
                                                                      lp_fname,
                                                                      val_idx_fname,
                                                                      val_idx_fname.size,
                                                                      1)
        val_logits_varuse, val_labels_varuse = prepare_batch_with_embeder(node_embeddings,
                                                                        ee_varuse,
                                                                        lp_varuse,
                                                                        val_idx_varuse,
                                                                        val_idx_varuse.size,
                                                                        1)
        val_logits_apicall, val_labels_apicall = prepare_batch_with_nodes(node_embeddings,
                                                                          ee_apicall,
                                                                          lp_apicall,
                                                                          val_idx_apicall,
                                                                          val_idx_apicall.size,
                                                                          1)

        if len(test_logits_fname) > 0:
            test_acc_fname, test_acc_varuse, test_acc_apicall = evaluate_no_classes(test_logits_fname, test_labels_fname), \
            evaluate_no_classes(test_logits_varuse, test_labels_varuse), \
            evaluate_no_classes(test_logits_apicall, test_labels_apicall)
        else:
            test_acc_fname, test_acc_varuse, test_acc_apicall = best_test_acc_fname, \
                                                                best_test_acc_varuse, \
                                                                best_test_acc_apicall


        if len(val_logits_fname) > 0:
            val_acc_fname, val_acc_varuse, val_acc_apicall = evaluate_no_classes(val_logits_fname, val_labels_fname), \
            evaluate_no_classes(val_logits_varuse, val_labels_varuse), \
            evaluate_no_classes(val_logits_apicall, val_labels_apicall)
        else:
            val_acc_fname, val_acc_varuse, val_acc_apicall = best_val_acc_fname, \
                                                                best_val_acc_varuse, \
                                                                best_val_acc_apicall


        # test_acc_fname, test_acc_varuse, test_acc_apicall, val_acc_fname, val_acc_varuse, val_acc_apicall = \
        #     evaluate_no_classes(test_logits_fname, test_labels_fname), \
        #     evaluate_no_classes(test_logits_varuse, test_labels_varuse), \
        #     evaluate_no_classes(test_logits_apicall, test_labels_apicall), \
        #     evaluate_no_classes(val_logits_fname, val_labels_fname), \
        #     evaluate_no_classes(val_logits_varuse, val_labels_varuse), \
        #     evaluate_no_classes(val_logits_apicall, val_labels_apicall)

        track_best(epoch, loss, train_acc_fname, val_acc_fname, test_acc_fname,
                   train_acc_varuse, val_acc_varuse, test_acc_varuse,
                   train_acc_apicall, val_acc_apicall, test_acc_apicall,
                   best_val_acc_fname, best_test_acc_fname,
                   best_val_acc_varuse, best_test_acc_varuse,
                   best_val_acc_apicall, best_test_acc_apicall)

        torch.save({
            'm': model.state_dict(),
            'ee_fname': ee_fname.state_dict(),
            'ee_varuse': ee_varuse.state_dict(),
            'ee_apicall': ee_apicall.state_dict(),
            "lp_fname": lp_fname.state_dict(),
            "lp_varuse": lp_varuse.state_dict(),
            "lp_apicall": lp_apicall.state_dict(),
            "epoch": epoch
        }, join(MODEL_BASE, "saved_state.pt"))


def training_procedure(dataset, model, params, EPOCHS, api_seq_file, fname_file, var_use_file, restore_state, MODEL_BASE):
    NODE_EMB_SIZE = 100
    ELEM_EMB_SIZE = 100

    lr = params.pop('lr')

    m = model(dataset.g,
              num_classes=NODE_EMB_SIZE,
              **params)

    # def create_elem_embedder(file_path, nodes, emb_size, compact_dst):
    #     element_data = pd.read_csv(file_path)
    #     function2nodeid = dict(zip(nodes['id'].values, nodes['global_graph_id'].values))
    #     element_data['id'] = element_data['src'].apply(lambda x: function2nodeid.get(x, None))
    #     if not compact_dst: # creating api call embedder
    #         element_data['dst'] = element_data['dst'].apply(lambda x: function2nodeid.get(x, None))
    #         element_data.drop_duplicates(['id', 'dst'], inplace=True, ignore_index=True)
    #     element_data = element_data.dropna(axis=0)
    #     ee = ElementEmbedder(element_data, emb_size, compact_dst=compact_dst)
    #     return ee

    ee_fname = create_elem_embedder(fname_file, dataset.nodes, ELEM_EMB_SIZE, True)
    ee_varuse = create_elem_embedder(var_use_file, dataset.nodes, ELEM_EMB_SIZE, True)
    ee_apicall = create_elem_embedder(api_seq_file, dataset.nodes, ELEM_EMB_SIZE, False)

    lp_fname = LinkPredictor(ee_fname.emb_size + m.emb_size)
    lp_varuse = LinkPredictor(ee_varuse.emb_size + m.emb_size)
    lp_apicall = LinkPredictor(m.emb_size + m.emb_size)

    if restore_state:
        checkpoint = torch.load(join(MODEL_BASE, "saved_state.pt"))
        m.load_state_dict(checkpoint['m'])
        ee_fname.load_state_dict(checkpoint['ee_fname'])
        ee_varuse.load_state_dict(checkpoint['ee_varuse'])
        ee_apicall.load_state_dict(checkpoint['ee_apicall'])
        lp_fname.load_state_dict(checkpoint['lp_fname'])
        lp_varuse.load_state_dict(checkpoint['lp_varuse'])
        lp_apicall.load_state_dict(checkpoint['lp_apicall'])
        print(f"Restored from epoch {checkpoint['epoch']}")
        checkpoint = None

    # from train_vector_sim_with_classifier import train_no_classes, final_evaluation_no_classes

    try:
        train(MODEL_BASE, m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, dataset.splits, EPOCHS)
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        m.eval()
        ee_fname.eval()
        ee_varuse.eval()
        ee_apicall.eval()
        lp_fname.eval()
        lp_varuse.eval()
        lp_apicall.eval()
        scores = final_evaluation_no_classes(m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall,
                                             dataset.splits)

    return m, ee_fname, ee_varuse, ee_apicall, lp_fname, lp_varuse, lp_apicall, scores
