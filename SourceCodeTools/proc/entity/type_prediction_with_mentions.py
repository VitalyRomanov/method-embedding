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

from SourceCodeTools.proc.entity.tf_model_with_mentions import TypePredictor, train

max_len = 400


from SourceCodeTools.proc.entity.type_prediction import ClassWeightNormalizer, evaluate, declarations_to_tags


def filter_declarations(entities, declarations):
    valid = {}

    for decl in declarations:
        for e in entities:
            if overlap(decl, e):
                valid[decl] = declarations[decl]

    return valid


def prepare_data_with_mentions(sents, model_path):
    sents_w = []
    sents_t = []
    sents_r = []
    sents_decls = []

    # nlp = spacy.load(model_path)  # load existing spaCy model
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

        decls = filter_declarations(ents, decls)

        tokens = [t.text for t in doc]
        ents_tags = biluo_tags_from_offsets(doc, ents)
        repl_tags = biluo_tags_from_offsets(doc, repl)

        while "-" in ents_tags:
            ents_tags[ents_tags.index("-")] = "O"

        while "-" in repl_tags:
            repl_tags[repl_tags.index("-")] = "O"

        decls = declarations_to_tags(doc, decls)

        repl_tags = [try_int(t.split("-")[-1]) for t in repl_tags]

        sents_w.append(tokens)
        sents_t.append(ents_tags)
        sents_r.append(repl_tags)
        sents_decls.append(decls)

    return sents_w, sents_t, sents_r, sents_decls


from SourceCodeTools.proc.entity.type_prediction import load_pkl_emb, load_w2v_map, create_tag_map

def create_batches_with_mentions(batch_size, seq_len, sents, repl, tags, decls_mentions, graphmap, wordmap, tagmap, class_weights, element_hash_size=1000):
    pad_id = len(wordmap)
    rpad_id = len(graphmap)
    n_sents = len(sents)

    b_sents = []
    b_repls = []
    b_tags = []
    b_cw = []
    b_lens = []
    b_pref = []
    b_suff = []
    b_decls = []
    b_ment = []


    for ind, (s, rr, tt, dec_men)  in enumerate(zip(sents, repl, tags, decls_mentions)):
        blank_s = np.ones((seq_len,), dtype=np.int32) * pad_id
        blank_r = np.ones((seq_len,), dtype=np.int32) * rpad_id
        blank_t = np.zeros((seq_len,), dtype=np.int32)
        # blank_t = np.zeros((1,), dtype=np.int32)
        blank_cw = np.ones((seq_len,), dtype=np.int32)
        blank_pref = np.ones((seq_len,), dtype=np.int32) * element_hash_size
        blank_suff = np.ones((seq_len,), dtype=np.int32) * element_hash_size
        blank_target = np.zeros((seq_len,), dtype=np.int32)
        blank_mentions = np.zeros((seq_len,), dtype=np.int32)


        int_sent = np.array([wordmap.get(w, pad_id) for w in s], dtype=np.int32)
        int_repl = np.array([graphmap.get(r, rpad_id) for r in rr], dtype=np.int32)
        # int_tags = np.array([tagmap.get(t, 0) for t in tt], dtype=np.int32)
        int_cw = np.array([class_weights.get(t, 1.0) for t in tt], dtype=np.int32)
        int_pref = np.array([el_hash(w[:3], element_hash_size-1) for w in s], dtype=np.int32)
        int_suff = np.array([el_hash(w[-3:], element_hash_size-1) for w in s], dtype=np.int32)


        blank_s[0:min(int_sent.size, seq_len)] = int_sent[0:min(int_sent.size, seq_len)]
        blank_r[0:min(int_sent.size, seq_len)] = int_repl[0:min(int_sent.size, seq_len)]
        # blank_t[0:min(int_sent.size, seq_len)] = int_tags[0:min(int_sent.size, seq_len)]
        blank_cw[0:min(int_sent.size, seq_len)] = int_cw[0:min(int_sent.size, seq_len)]
        blank_pref[0:min(int_sent.size, seq_len)] = int_pref[0:min(int_sent.size, seq_len)]
        blank_suff[0:min(int_sent.size, seq_len)] = int_suff[0:min(int_sent.size, seq_len)]

        for dec, men in dec_men:
            blank_t[:] = tagmap["O"]
            blank_target[:] = 0
            blank_mentions[:] = 0
            int_dec = np.array([0 if w == "O" else 1 for w in dec], dtype=np.int32)
            int_men = np.array([0 if w == "O" else 1 for w in men], dtype=np.int32)
            int_tags = np.array([tagmap[t] if dec != "O" else tagmap["O"] for t, dec in zip(tt, dec)], dtype=np.int32)
            # assert sum(int_tags) != 0
            # blank_t[0] = np.array([tagmap[list(filter(lambda x: x != "O", (t if dec != "O" else "O" for t, dec in zip(tt, dec))))[0]]], dtype=np.int32)



            blank_target[0:min(int_dec.size, seq_len)] = int_dec[0:min(int_dec.size, seq_len)]
            blank_mentions[0:min(int_men.size, seq_len)] = int_men[0:min(int_men.size, seq_len)]
            blank_t[0:min(int_sent.size, seq_len)] = int_tags[0:min(int_sent.size, seq_len)]

            b_lens.append(len(s) if len(s) < seq_len else seq_len)
            b_sents.append(blank_s)
            b_repls.append(blank_r)
            b_tags.append(blank_t)
            b_cw.append(blank_cw)
            b_pref.append(blank_pref)
            b_suff.append(blank_suff)
            b_decls.append(blank_target)
            b_ment.append(blank_mentions)


    lens = np.array(b_lens, dtype=np.int32)
    sentences = np.stack(b_sents)
    replacements = np.stack(b_repls)
    pos_tags = np.stack(b_tags)
    cw = np.stack(b_cw)
    prefixes = np.stack(b_pref)
    suffixes = np.stack(b_suff)
    targets = np.stack(b_decls)
    mentions = np.stack(b_ment)

    batch = []
    for i in range(n_sents // batch_size):
        batch.append({"tok_ids": sentences[i * batch_size: i * batch_size + batch_size, :],
                      "graph_ids": replacements[i * batch_size: i * batch_size + batch_size, :],
                      "prefix": prefixes[i * batch_size: i * batch_size + batch_size, :],
                      "suffix": suffixes[i * batch_size: i * batch_size + batch_size, :],
                      "target": targets[i * batch_size: i * batch_size + batch_size, :],
                      "mentions": mentions[i * batch_size: i * batch_size + batch_size, :],
                      "tags": pos_tags[i * batch_size: i * batch_size + batch_size, :],
                      "class_weights": cw[i * batch_size: i * batch_size + batch_size, :],
                      "lens": lens[i * batch_size: i * batch_size + batch_size]})

    return batch


from SourceCodeTools.proc.entity.type_prediction import scorer


def main_tf_hyper_search(TRAIN_DATA, TEST_DATA,
            tokenizer_path=None, graph_emb_path=None, word_emb_path=None,
            output_dir=None, n_iter=30, max_len=500,
            suffix_prefix_dims=50, suffix_prefix_buckets=1000,
            learning_rate=0.01, learning_rate_decay=1.0, batch_size=32, finetune=False):

    train_s, train_e, train_r, train_decls_mentions = prepare_data_with_mentions(TRAIN_DATA, tokenizer_path)
    test_s, test_e, test_r, test_decls_mentions = prepare_data_with_mentions(TEST_DATA, tokenizer_path)

    cw = ClassWeightNormalizer()
    cw.init(train_e)

    t_map, inv_t_map = create_tag_map(train_e)

    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    batches = create_batches_with_mentions(batch_size, max_len, train_s, train_r, train_e, train_decls_mentions, graph_emb.ind, word_emb.ind, t_map, cw, element_hash_size=suffix_prefix_buckets)
    test_batch = create_batches_with_mentions(len(test_s), max_len, test_s, test_r, test_e, test_decls_mentions, graph_emb.ind, word_emb.ind, t_map, cw, element_hash_size=suffix_prefix_buckets)

    params = {
        "params": [
            {
                "h_sizes": [10, 10, 10],
                "dense_size": 10,
                "pos_emb_size": 10,
                "cnn_win_size": 3,
                "suffix_prefix_dims": 10,
                "suffix_prefix_buckets": 1000,
                "target_emb_dim": 5,
                "mention_emb_dim": 5
            },
            {
                "h_sizes": [20, 20],
                "dense_size": 10,
                "pos_emb_size": 20,
                "cnn_win_size": 3,
                "suffix_prefix_dims": 20,
                "suffix_prefix_buckets": 1000,
                "target_emb_dim": 5,
                "mention_emb_dim": 5
            },
            # {
            #     "h_sizes": [20, 20, 20],
            #     "dense_size": 20,
            #     "pos_emb_size": 20,
            #     "cnn_win_size": 3,
            #     "suffix_prefix_dims": 20,
            #     "suffix_prefix_buckets": 1000,
            #     "target_emb_dim": 5,
            #     "mention_emb_dim": 5
            # },
            # {
            #     "h_sizes": [40, 40, 40],
            #     "dense_size": 30,
            #     "pos_emb_size": 30,
            #     "cnn_win_size": 5,
            #     "suffix_prefix_dims": 50,
            #     "suffix_prefix_buckets": 2000,
            #     "target_emb_dim": 15,
            #     "mention_emb_dim": 15
            # },
            # {
            #     "h_sizes": [80, 80, 80],
            #     "dense_size": 40,
            #     "pos_emb_size": 50,
            #     "cnn_win_size": 7,
            #     "suffix_prefix_dims": 70,
            #     "suffix_prefix_buckets": 3000,
            #     "target_emb_dim": 25,
            #     "mention_emb_dim": 25,
            # }
        ],
        "learning_rate": [0.0001, 0.001, 0.00001],
        "learning_rate_decay": [0.998] # 0.991
    }
    from sklearn.model_selection import ParameterGrid

    for param_set_ind, params in enumerate(ParameterGrid(params)):

        print(f"\n\n{params}")
        lr = params.pop("learning_rate")
        lr_decay = params.pop("learning_rate_decay")
        params = params['params']

        ntrials = 3

        param_dir = os.path.join(output_dir, repr(param_set_ind))
        os.mkdir(param_dir)

        for trial_ind in range(ntrials):

            trial_dir = os.path.join(param_dir, repr(trial_ind))
            os.mkdir(trial_dir)

            model = TypePredictor(word_emb, graph_emb, train_embeddings=finetune,
                                  num_classes=len(t_map), seq_len=max_len, **params)

            train_losses, train_f1, test_losses, test_f1 = train(model=model, train_batches=batches, test_batches=test_batch, epochs=n_iter, learning_rate=lr,
                  scorer=lambda pred, true: scorer(pred, true, inv_t_map), learning_rate_decay=lr_decay)

            chechpoint_path = os.path.join(trial_dir, "checkpoint")
            model.save_weights(chechpoint_path)

            metadata = {
                "train_losses": train_losses,
                "train_f1": train_f1,
                "test_losses": test_losses,
                "test_f1": test_f1,
                "learning_rate": lr,
                "learning_rate_decay": lr_decay,
                "epochs": n_iter
            }

            metadata.update(params)

            with open(os.path.join(trial_dir, "params.json"), "w") as metadata_sink:
                metadata_sink.write(json.dumps(metadata, indent=4))



def main_tf(TRAIN_DATA, TEST_DATA,
            tokenizer_path=None, graph_emb_path=None, word_emb_path=None,
            output_dir=None, n_iter=30, max_len=100,
            suffix_prefix_dims=50, suffix_prefix_buckets=1000,
            learning_rate=0.01, learning_rate_decay=1.0, batch_size=32, finetune=False):

    train_s, train_e, train_r, train_decls_mentions = prepare_data_with_mentions(TRAIN_DATA, tokenizer_path)
    test_s, test_e, test_r, test_decls_mentions = prepare_data_with_mentions(TEST_DATA, tokenizer_path)

    cw = ClassWeightNormalizer()
    cw.init(train_e)

    t_map, inv_t_map = create_tag_map(train_e)

    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    batches = create_batches_with_mentions(batch_size, max_len, train_s, train_r, train_e, train_decls_mentions, graph_emb.ind, word_emb.ind, t_map, cw, element_hash_size=suffix_prefix_buckets)
    test_batch = create_batches_with_mentions(len(test_s), max_len, test_s, test_r, test_e, test_decls_mentions, graph_emb.ind, word_emb.ind, t_map, cw, element_hash_size=suffix_prefix_buckets)

    model = TypePredictor(word_emb, graph_emb, train_embeddings=finetune,
                 h_sizes=[40, 40, 40], dense_size=30, num_classes=len(t_map),
                 seq_len=max_len, pos_emb_size=30, cnn_win_size=3,
                 suffix_prefix_dims=suffix_prefix_dims, suffix_prefix_buckets=suffix_prefix_buckets)

    train(model=model, train_batches=batches, test_batches=test_batch, epochs=n_iter, learning_rate=learning_rate,
          scorer=lambda pred, true: scorer(pred, true, inv_t_map), learning_rate_decay=learning_rate_decay, finetune=finetune)

    model.save_weights(output_dir)



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
    parser.add_argument('model_output',
                        help='')

    args = parser.parse_args()

    model_path = sys.argv[1]
    data_path = sys.argv[2]

    output_dir = args.model_output
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    n_iter = 600

    allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
               'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}

    TRAIN_DATA, TEST_DATA = read_data(args.data_path, normalize=True, allowed=allowed, include_replacements=True)
    if args.hyper_search:
        main_tf_hyper_search(TRAIN_DATA, TEST_DATA, args.tokenizer,
                             graph_emb_path=args.graph_emb_path,
                             word_emb_path=args.word_emb_path,
                             output_dir=output_dir,
                             n_iter=n_iter,
                             learning_rate=args.learning_rate,
                             learning_rate_decay=args.learning_rate_decay,
                             batch_size=args.batch_size,
                             finetune=args.finetune)
    else:
        main_tf(TRAIN_DATA, TEST_DATA, args.tokenizer,
                graph_emb_path=args.graph_emb_path,
                word_emb_path=args.word_emb_path,
                output_dir=output_dir,
                n_iter=n_iter,
                learning_rate=args.learning_rate,
                learning_rate_decay=args.learning_rate_decay,
                batch_size=args.batch_size,
                finetune=args.finetune)
    # main_spacy(TRAIN_DATA, TEST_DATA, model=model_path,output_dir=output_dir, n_iter=n_iter)