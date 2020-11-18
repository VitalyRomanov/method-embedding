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

max_len = 400


class ClassWeightNormalizer:
    def __init__(self):
        self.class_counter = None

    def init(self, classes):
        import itertools
        from collections import Counter
        self.class_counter = Counter(c for c in itertools.chain.from_iterable(classes))
        total_count = sum(self.class_counter.values())
        self.class_weights = {key: total_count / val for key, val in self.class_counter.items()}

    def __getitem__(self, item):
        return self.class_weights[item]

    def get(self, item, default):
        return self.class_weights.get(item, default)


def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return {key: scorer.scores[key] for key in ['ents_p', 'ents_r', 'ents_f', 'ents_per_type']}
    return scorer.scores['ents_per_type']


# def isvalid(nlp, text, ents):
#     doc = nlp(text)
#     tags = biluo_tags_from_offsets(doc, ents)
#     if "-" in tags:
#         return False
#     else:
#         return True


def main_spacy(TRAIN_DATA, TEST_DATA, model, output_dir=None, n_iter=100):
    nlp = spacy.load(model)  # load existing spaCy model

    print("dealing with inconsistencies")
    TRAIN_DATA = deal_with_incorrect_offsets(TRAIN_DATA, nlp)
    TEST_DATA = deal_with_incorrect_offsets(TEST_DATA, nlp)
    print("done dealing with inconsistencies")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        # if model is None:
        nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print(f"{itn}:")
            print("\tLosses", losses)
            score = evaluate(nlp, TEST_DATA)
            if not os.path.isdir("models"):
                os.mkdir("models")
            nlp.to_disk(os.path.join("models", f"model_{itn}"))
            print("\t", score)


def tags_to_mask(tags):
    return list(map(lambda t: 1. if t != "O" else 0., tags))

def declarations_to_tags(doc, decls):
    declarations = []

    for decl, mentions in decls.items():
        tag_decl = biluo_tags_from_offsets(doc, [decl])
        tag_mentions = biluo_tags_from_offsets(doc, mentions)

        assert "-" not in tag_decl

        # while "-" in tag_decl:
        #     tag_decl[tag_decl.index("-")] = "O"

        # decl_mask = tags_to_mask(tag_decl)

        # assert sum(decl_mask) > 0.

        while "-" in tag_mentions:
            tag_mentions[tag_mentions.index("-")] = "O"

        # if "-" in tag_mentions:
        #     for t, tag in zip(doc, tag_mentions):
        #         print(t, tag, sep="\t")

        declarations.append((tag_decl, tag_mentions))

    return declarations



def prepare_data(sents, model_path):
    sents_w = []
    sents_t = []
    sents_r = []

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

        tokens = [t.text for t in doc]
        ents_tags = biluo_tags_from_offsets(doc, ents)
        repl_tags = biluo_tags_from_offsets(doc, repl)

        while "-" in ents_tags:
            ents_tags[ents_tags.index("-")] = "O"

        while "-" in repl_tags:
            repl_tags[repl_tags.index("-")] = "O"

        repl_tags = [try_int(t.split("-")[-1]) for t in repl_tags]

        sents_w.append(tokens)
        sents_t.append(ents_tags)
        sents_r.append(repl_tags)

    return sents_w, sents_t, sents_r


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


def create_batches(batch_size, seq_len, sents, repl, tags, graphmap, wordmap, tagmap, class_weights, element_hash_size=1000):
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


    for ind, (s, rr, tt)  in enumerate(zip(sents, repl, tags)):
        blank_s = np.ones((seq_len,), dtype=np.int32) * pad_id
        blank_r = np.ones((seq_len,), dtype=np.int32) * rpad_id
        blank_t = np.zeros((seq_len,), dtype=np.int32)
        blank_cw = np.ones((seq_len,), dtype=np.int32)
        blank_pref = np.ones((seq_len,), dtype=np.int32) * element_hash_size
        blank_suff = np.ones((seq_len,), dtype=np.int32) * element_hash_size


        int_sent = np.array([wordmap.get(w, pad_id) for w in s], dtype=np.int32)
        int_repl = np.array([graphmap.get(r, rpad_id) for r in rr], dtype=np.int32)
        int_tags = np.array([tagmap.get(t, 0) for t in tt], dtype=np.int32)
        int_cw = np.array([class_weights.get(t, 1.0) for t in tt], dtype=np.int32)
        int_pref = np.array([el_hash(w[:3], element_hash_size-1) for w in s], dtype=np.int32)
        int_suff = np.array([el_hash(w[-3:], element_hash_size-1) for w in s], dtype=np.int32)


        blank_s[0:min(int_sent.size, seq_len)] = int_sent[0:min(int_sent.size, seq_len)]
        blank_r[0:min(int_sent.size, seq_len)] = int_repl[0:min(int_sent.size, seq_len)]
        blank_t[0:min(int_sent.size, seq_len)] = int_tags[0:min(int_sent.size, seq_len)]
        blank_cw[0:min(int_sent.size, seq_len)] = int_cw[0:min(int_sent.size, seq_len)]
        blank_pref[0:min(int_sent.size, seq_len)] = int_pref[0:min(int_sent.size, seq_len)]
        blank_suff[0:min(int_sent.size, seq_len)] = int_suff[0:min(int_sent.size, seq_len)]

        # print(int_sent[0:min(int_sent.size, seq_len)].shape)

        b_lens.append(len(s) if len(s) < seq_len else seq_len)
        b_sents.append(blank_s)
        b_repls.append(blank_r)
        b_tags.append(blank_t)
        b_cw.append(blank_cw)
        b_pref.append(blank_pref)
        b_suff.append(blank_suff)

    lens = np.array(b_lens, dtype=np.int32)
    sentences = np.stack(b_sents)
    replacements = np.stack(b_repls)
    pos_tags = np.stack(b_tags)
    cw = np.stack(b_cw)
    prefixes = np.stack(b_pref)
    suffixes = np.stack(b_suff)

    batch = []
    for i in range(n_sents // batch_size):
        batch.append({"tok_ids": sentences[i * batch_size: i * batch_size + batch_size, :],
                      "graph_ids": replacements[i * batch_size: i * batch_size + batch_size, :],
                      "prefix": prefixes[i * batch_size: i * batch_size + batch_size, :],
                      "suffix": suffixes[i * batch_size: i * batch_size + batch_size, :],
                      "tags": pos_tags[i * batch_size: i * batch_size + batch_size, :],
                      "class_weights": cw[i * batch_size: i * batch_size + batch_size, :],
                      "lens": lens[i * batch_size: i * batch_size + batch_size]})

    return batch


def create_batches_with_mask(batch_size, seq_len, sents, repl, tags, unlabeled_decls, graphmap, wordmap, tagmap, class_weights=None, element_hash_size=1000):
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
    b_hide_mask = []


    for ind, (s, rr, tt, un)  in enumerate(zip(sents, repl, tags, unlabeled_decls)):
        blank_s = np.ones((seq_len,), dtype=np.int32) * pad_id
        blank_r = np.ones((seq_len,), dtype=np.int32) * rpad_id
        blank_t = np.zeros((seq_len,), dtype=np.int32)
        blank_cw = np.ones((seq_len,), dtype=np.int32)
        blank_pref = np.ones((seq_len,), dtype=np.int32) * element_hash_size
        blank_suff = np.ones((seq_len,), dtype=np.int32) * element_hash_size
        blank_hide_mask = np.ones((seq_len,), dtype=np.int32)


        int_sent = np.array([wordmap.get(w, pad_id) for w in s], dtype=np.int32)
        int_repl = np.array([graphmap.get(r, rpad_id) for r in rr], dtype=np.int32)
        int_tags = np.array([tagmap.get(t, 0) for t in tt], dtype=np.int32)
        if class_weights is not None:
            int_cw = np.array([class_weights.get(t, 1.0) for t in tt], dtype=np.int32)
        int_pref = np.array([el_hash(w[:3], element_hash_size-1) for w in s], dtype=np.int32)
        int_suff = np.array([el_hash(w[-3:], element_hash_size-1) for w in s], dtype=np.int32)
        int_hide_mask = np.array([1 if t == "O" else 0 for t in un], dtype=np.int32)


        blank_s[0:min(int_sent.size, seq_len)] = int_sent[0:min(int_sent.size, seq_len)]
        blank_r[0:min(int_sent.size, seq_len)] = int_repl[0:min(int_sent.size, seq_len)]
        blank_t[0:min(int_sent.size, seq_len)] = int_tags[0:min(int_sent.size, seq_len)]
        if class_weights is not None:
            blank_cw[0:min(int_sent.size, seq_len)] = int_cw[0:min(int_sent.size, seq_len)]
        blank_pref[0:min(int_sent.size, seq_len)] = int_pref[0:min(int_sent.size, seq_len)]
        blank_suff[0:min(int_sent.size, seq_len)] = int_suff[0:min(int_sent.size, seq_len)]
        blank_hide_mask[0:min(int_sent.size, seq_len)] = int_hide_mask[0:min(int_sent.size, seq_len)]

        # print(int_sent[0:min(int_sent.size, seq_len)].shape)

        b_lens.append(len(s) if len(s) < seq_len else seq_len)
        b_sents.append(blank_s)
        b_repls.append(blank_r)
        b_tags.append(blank_t)
        b_cw.append(blank_cw)
        b_pref.append(blank_pref)
        b_suff.append(blank_suff)
        b_hide_mask.append(blank_hide_mask)

    lens = np.array(b_lens, dtype=np.int32)
    sentences = np.stack(b_sents)
    replacements = np.stack(b_repls)
    pos_tags = np.stack(b_tags)
    cw = np.stack(b_cw)
    prefixes = np.stack(b_pref)
    suffixes = np.stack(b_suff)
    hide_mask = np.stack(b_hide_mask)

    batch = []
    for i in range(n_sents // batch_size):
        batch.append({"tok_ids": sentences[i * batch_size: i * batch_size + batch_size, :],
                      "graph_ids": replacements[i * batch_size: i * batch_size + batch_size, :],
                      "prefix": prefixes[i * batch_size: i * batch_size + batch_size, :],
                      "suffix": suffixes[i * batch_size: i * batch_size + batch_size, :],
                      "tags": pos_tags[i * batch_size: i * batch_size + batch_size, :],
                      "class_weights": cw[i * batch_size: i * batch_size + batch_size, :],
                      "hide_mask": hide_mask[i * batch_size: i * batch_size + batch_size, :],
                      "lens": lens[i * batch_size: i * batch_size + batch_size]})

    return batch


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


def main_tf_hyper_search(TRAIN_DATA, TEST_DATA,
            tokenizer_path=None, graph_emb_path=None, word_emb_path=None,
            output_dir=None, n_iter=30, max_len=100,
            suffix_prefix_dims=50, suffix_prefix_buckets=1000,
            learning_rate=0.01, learning_rate_decay=1.0, batch_size=32, finetune=False):

    train_s, train_e, train_r, train_unlabeled_decls = prepare_data_with_mentions(TRAIN_DATA, tokenizer_path)
    test_s, test_e, test_r, test_unlabeled_decls = prepare_data_with_mentions(TEST_DATA, tokenizer_path)

    cw = ClassWeightNormalizer()
    cw.init(train_e)

    t_map, inv_t_map = create_tag_map(train_e)

    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    batches = create_batches_with_mask(batch_size, max_len, train_s, train_r, train_e, train_unlabeled_decls, graph_emb.ind,
                                       word_emb.ind, t_map, element_hash_size=suffix_prefix_buckets)
    test_batch = create_batches_with_mask(len(test_s), max_len, test_s, test_r, test_e, test_unlabeled_decls, graph_emb.ind,
                                          word_emb.ind, t_map, element_hash_size=suffix_prefix_buckets)

    params = {
        "params": [
            # {
            #     "h_sizes": [10, 10, 10],
            #     "dense_size": 10,
            #     "pos_emb_size": 10,
            #     "cnn_win_size": 5,
            #     "suffix_prefix_dims": 10,
            #     "suffix_prefix_buckets": 1000,
            # },
            # {
            #     "h_sizes": [20, 20, 20],
            #     "dense_size": 20,
            #     "pos_emb_size": 20,
            #     "cnn_win_size": 5,
            #     "suffix_prefix_dims": 20,
            #     "suffix_prefix_buckets": 1000,
            # },
            {
                "h_sizes": [40, 40, 40],
                "dense_size": 30,
                "pos_emb_size": 30,
                "cnn_win_size": 5,
                "suffix_prefix_dims": 50,
                "suffix_prefix_buckets": 2000,
            },
        #     {
        #     "h_sizes": [80, 80, 80],
        #     "dense_size": 40,
        #     "pos_emb_size": 50,
        #     "cnn_win_size": 7,
        #     "suffix_prefix_dims": 70,
        #     "suffix_prefix_buckets": 3000,
        # }
        ],
        # "h_sizes": [[40, 40, 40], [20, 20, 20], [80, 80, 80]],
        # "dense_size": [20, 30, 40],
        # "pos_emb_size": [20, 30, 50],
        # "cnn_win_size": [3, 5, 7],
        # "suffix_prefix_dims": [20, 50, 70],
        # "suffix_prefix_buckets": [1000, 2000],
        "learning_rate": [0.0001],
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
                                  num_classes=len(t_map), seq_len=max_len,**params)

            train_losses, train_f1, test_losses, test_f1 = train(model=model, train_batches=batches, test_batches=test_batch, epochs=n_iter, learning_rate=lr,
                  scorer=lambda pred, true: scorer(pred, true, inv_t_map), learning_rate_decay=lr_decay, finetune=finetune)

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

            pickle.dump((t_map, inv_t_map), open(os.path.join(trial_dir, "tag_types.pkl"), "wb"))



def main_tf(TRAIN_DATA, TEST_DATA,
            tokenizer_path=None, graph_emb_path=None, word_emb_path=None,
            output_dir=None, n_iter=30, max_len=100,
            suffix_prefix_dims=50, suffix_prefix_buckets=1000,
            learning_rate=0.01, learning_rate_decay=1.0, batch_size=32, finetune=False):

    train_s, train_e, train_r, train_unlabeled_decls = prepare_data_with_mentions(TRAIN_DATA, tokenizer_path)
    test_s, test_e, test_r, test_unlabeled_decls = prepare_data_with_mentions(TEST_DATA, tokenizer_path)

    # cw = ClassWeightNormalizer()
    # cw.init(train_e)

    t_map, inv_t_map = create_tag_map(train_e)

    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    batches = create_batches_with_mask(batch_size, max_len, train_s, train_r, train_e, train_unlabeled_decls, graph_emb.ind, word_emb.ind, t_map, element_hash_size=suffix_prefix_buckets)
    test_batch = create_batches_with_mask(len(test_s), max_len, test_s, test_r, test_e, test_unlabeled_decls, graph_emb.ind, word_emb.ind, t_map, element_hash_size=suffix_prefix_buckets)

    model = TypePredictor(word_emb, graph_emb, train_embeddings=finetune,
                 h_sizes=[40, 40, 40], dense_size=30, num_classes=len(t_map),
                 seq_len=max_len, pos_emb_size=30, cnn_win_size=3,
                 suffix_prefix_dims=suffix_prefix_dims, suffix_prefix_buckets=suffix_prefix_buckets)

    train(model=model, train_batches=batches, test_batches=test_batch, epochs=n_iter, learning_rate=learning_rate,
          scorer=lambda pred, true: scorer(pred, true, inv_t_map), learning_rate_decay=learning_rate_decay, finetune=finetune)

    model.save_weights(output_dir)

    pickle.dump((t_map, inv_t_map), open("tag_types.pkl", "wb"))



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

    n_iter = 500

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