from __future__ import unicode_literals, print_function
import spacy
import sys, json, os
import pickle
from SourceCodeTools.proc.entity.util import inject_tokenizer, read_data, deal_with_incorrect_offsets, el_hash, overlap
from spacy.gold import biluo_tags_from_offsets, offsets_from_biluo_tags

from spacy.gold import GoldParse
from spacy.scorer import Scorer

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import numpy as np
from copy import copy

from SourceCodeTools.proc.entity.ast_tools import get_declarations

from SourceCodeTools.graph.model.Embedder import Embedder
# from tf_model import create_batches

from SourceCodeTools.proc.entity.tf_model import estimate_crf_transitions, TypePredictor, train
from SourceCodeTools.proc.entity.type_prediction import create_batches_with_mask

max_len = 400

from SourceCodeTools.proc.entity.ClassWeightNormalizer import ClassWeightNormalizer

from SourceCodeTools.proc.entity.type_prediction import tags_to_mask, declarations_to_tags


def prepare_data_with_mentions(sents, model):
    sents_w = []
    sents_t = []
    sents_r = []
    unlabeled_decls = []

    nlp = inject_tokenizer(spacy.blank("en"))

    def try_int(val):
        try:
            return int(val)
        except:
            return val

    for s in sents:
        doc = nlp(s[0])
        ents = s[1]['entities']
        repl = s[1]['replacements']
        decls = get_declarations(s[0])

        unlabeled_dec = prepare_mask(ents, decls)

        tokens = [t.text for t in doc]
        ents_tags = biluo_tags_from_offsets(doc, ents)
        repl_tags = biluo_tags_from_offsets(doc, repl)
        unlabeled_dec = biluo_tags_from_offsets(doc, unlabeled_dec)

        assert len(tokens) == len(ents_tags) == len(repl_tags) == len(unlabeled_dec)

        while "-" in ents_tags:
            ents_tags[ents_tags.index("-")] = "O"

        while "-" in repl_tags:
            repl_tags[repl_tags.index("-")] = "O"

        while "-" in unlabeled_dec:
            unlabeled_dec[unlabeled_dec.index("-")] = "O"

        # decls = declarations_to_tags(doc, decls)

        repl_tags = [try_int(t.split("-")[-1]) for t in repl_tags]

        sents_w.append(tokens)
        sents_t.append(ents_tags)
        sents_r.append(repl_tags)
        unlabeled_decls.append(unlabeled_dec)

    return sents_w, sents_t, sents_r, unlabeled_decls


def prepare_mask(entities, declarations):
    for_mask = []
    for decl in declarations:
        for e in entities:
            if overlap(decl, e):
                break
        else:
            for_mask.append(decl)
    return for_mask


def filter_declarations(entities, declarations):
    valid = {}

    for decl in declarations:
        for e in entities:
            if overlap(decl, e):
                valid[decl] = declarations[decl]

    return valid


def load_pkl_emb(path):
    embedder = pickle.load(open(path, "rb"))
    if isinstance(embedder, list):
        embedder = embedder[-1]
    return embedder

def load_w2v_map(w2v_path):

    embs = []
    w_map = dict()

    with open(w2v_path) as w2v:
        n_vectors, n_dims = map(int, w2v.readline().strip().split())
        for ind in range(n_vectors):
            e = w2v.readline().strip().split()

            word = e[0]
            w_map[word] = len(w_map)

            embs.append(list(map(float, e[1:])))

    return Embedder(w_map, np.array(embs))

    # model = KeyedVectors.load_word2vec_format(model_p)
    # voc_len = len(model.vocab)
    #
    # vectors = np.zeros((voc_len, 100), dtype=np.float32)
    #
    # w2i = dict()
    #
    # for ind, word in enumerate(model.vocab.keys()):
    #     w2i[word] = ind
    #     vectors[ind, :] = model[word]
    #
    # # w2i["*P*"] = len(w2i)
    #
    # return model, w2i, vectors

def create_tag_map(sents):
    tags = set()

    for s in sents:
        tags.update(set(s))

    tagmap = dict(zip(tags, range(len(tags))))

    aid, iid = zip(*tagmap.items())
    inv_tagmap = dict(zip(iid, aid))

    return tagmap, inv_tagmap




def parse_biluo(biluo):
    spans = []

    expected = {"B", "U", "0"}
    expected_tag = None

    c_start = 0

    for ind, t in enumerate(biluo):
        if t[0] not in expected:
            expected = {"B", "U", "0"}
            continue

        if t[0] == "U":
            c_start = ind
            c_end = ind + 1
            c_type = t.split("-")[1]
            spans.append((c_start, c_end, c_type))
            expected = {"B", "U", "0"}
            expected_tag = None
        elif t[0] == "B":
            c_start = ind
            expected = {"I", "L"}
            expected_tag = t.split("-")[1]
        elif t[0] == "I":
            if t.split("-")[1] != expected_tag:
                expected = {"B", "U", "0"}
                expected_tag = None
                continue
        elif t[0] == "L":
            if t.split("-")[1] != expected_tag:
                expected = {"B", "U", "0"}
                expected_tag = None
                continue
            c_end = ind + 1
            c_type = expected_tag
            spans.append((c_start, c_end, c_type))
            expected = {"B", "U", "0"}
            expected_tag = None
        elif t[0] == "0":
            expected = {"B", "U", "0"}
            expected_tag = None

    return spans



def scorer(pred, labels, inverse_tag_map, eps=1e-8):
    pred_biluo = [inverse_tag_map[p] for p in pred]
    labels_biluo = [inverse_tag_map[p] for p in labels]

    pred_spans = set(parse_biluo(pred_biluo))
    true_spans = set(parse_biluo(labels_biluo))

    tp = len(pred_spans.intersection(true_spans))
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1

import tensorflow as tf
def logits_to_annotations(predicted, lengths, tag_map, inv_tag_map):
    mask = tf.sequence_mask(lengths, predicted.shape[1])
    argmax = predicted
    nulled = argmax * tf.cast(mask, dtype=tf.int64)

    # out_tag = tag_map['O']

    out_tags = tf.cast(tf.logical_not(mask), dtype=tf.int64) * tag_map['O']
    tag_labels = tf.maximum(nulled, out_tags)

    ents = []
    for i in range(tag_labels.shape[0]):
        line = tag_labels[i].numpy()
        tag_line = [inv_tag_map[v] for v in line]
        ent = parse_biluo(tag_line)
        ents.append(ent)

    return ents



def main_tf(TRAIN_DATA, TEST_DATA,
            tokenizer_path=None, graph_emb_path=None, word_emb_path=None,
            checkpoint_path=None, n_iter=30, max_len=100,
            suffix_prefix_dims=50, suffix_prefix_buckets=1000,
            learning_rate=0.01, learning_rate_decay=1.0, batch_size=32, finetune=False):

    # train_s, train_e, train_r, train_unlabeled_decls = prepare_data_with_mentions(TRAIN_DATA, tokenizer_path)
    test_s, test_e, test_r, test_unlabeled_decls = prepare_data_with_mentions(TEST_DATA, tokenizer_path)

    t_map, inv_t_map = pickle.load(open(os.path.join(checkpoint_path, "tag_types.pkl"), "rb"))

    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    with open(os.path.join(checkpoint_path, "params.json"), "r") as json_params:
        params = json.loads(json_params.read().strip())
        params.pop("train_losses")
        params.pop("test_losses")
        params.pop("train_f1")
        params.pop("test_f1")
        params.pop("epochs")
        params.pop("learning_rate")
        params.pop("learning_rate_decay")

    # params = {
    #             "h_sizes": [40, 40, 40],
    #             "dense_size": 30,
    #             "pos_emb_size": 30,
    #             "cnn_win_size": 5,
    #             "suffix_prefix_dims": 50,
    #             "suffix_prefix_buckets": 2000,
    #         }

    # batches = create_batches_with_mask(batch_size, max_len, train_s, train_r, train_e, train_unlabeled_decls, graph_emb.ind,
    #                                    word_emb.ind, t_map, cw, element_hash_size=suffix_prefix_buckets)
    test_batch = create_batches_with_mask(len(test_s), max_len, test_s, test_r, test_e, test_unlabeled_decls, graph_emb.ind,
                                          word_emb.ind, t_map, element_hash_size=suffix_prefix_buckets)[0]

    model = TypePredictor(word_emb, graph_emb, train_embeddings=finetune,
                                  num_classes=len(t_map), seq_len=max_len, **params)

    model.load_weights(os.path.join(checkpoint_path, "checkpoint"))

    true_annotations = logits_to_annotations(test_batch['tags'], test_batch['lens'], t_map, inv_t_map)
    est_annotations = logits_to_annotations(
        tf.math.argmax(model(token_ids=test_batch['tok_ids'],prefix_ids=test_batch['prefix'], suffix_ids=test_batch['suffix'], graph_ids=test_batch['graph_ids'], training=False), axis=-1),
        test_batch['lens'],
        t_map, inv_t_map
    )

    from SourceCodeTools.proc.entity.entity_render import render_annotations

    html = render_annotations(zip(TEST_DATA, est_annotations, true_annotations))
    with open("render.html", "w") as render:
        render.write(html)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tokenizer', dest='tokenizer', default=None,
                        help='')
    parser.add_argument('--data_path', dest='data_path', default=None,
                        help='Path to the file with nodes')
    parser.add_argument('--graph_emb_path', dest='graph_emb_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--word_emb_path', dest='word_emb_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.01, type=float,
                        help='')
    parser.add_argument('--learning_rate_decay', dest='learning_rate_decay', default=1.0, type=float,
                        help='')
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=int,
                        help='')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--hyper_search', action='store_true')
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', default=None,
                        help='')

    args = parser.parse_args()

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
               'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}

    TRAIN_DATA, TEST_DATA = read_data(args.data_path, normalize=True, allowed=allowed, include_replacements=True)
    main_tf(TRAIN_DATA, TEST_DATA, args.tokenizer,
                graph_emb_path=args.graph_emb_path,
                word_emb_path=args.word_emb_path,
                checkpoint_path=args.checkpoint_path,
                learning_rate=args.learning_rate,
                learning_rate_decay=args.learning_rate_decay,
                batch_size=args.batch_size,
                finetune=args.finetune)