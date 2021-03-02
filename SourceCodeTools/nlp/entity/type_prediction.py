from __future__ import unicode_literals, print_function

import json
import os
import pickle
import sys
from copy import copy
from datetime import datetime
from functools import lru_cache
from math import ceil
from typing import List, Dict, Optional

import numpy as np
from spacy.gold import biluo_tags_from_offsets

from SourceCodeTools.code.ast_tools import get_declarations
from SourceCodeTools.models.ClassWeightNormalizer import ClassWeightNormalizer
from SourceCodeTools.nlp import token_hasher, create_tokenizer, tag_map_from_sentences, TagMap
from SourceCodeTools.nlp.entity import parse_biluo
from SourceCodeTools.nlp.entity.tf_models.params import cnn_params
from SourceCodeTools.nlp.entity.tf_models.tf_model import TypePredictor, train
from SourceCodeTools.nlp.entity.utils import get_unique_entities, overlap
from SourceCodeTools.nlp.entity.utils.data import read_data


def declarations_to_tags(doc, decls):
    """
    Converts the declarations and mentions of a variable into BILUO format
    :param doc: source code of a function
    :param decls: dictionary that maps from the variable declarations (first usage) to all the mentions
                    later in the function
    :return: List of tuple [(declaration_tags), (mentions_tags)]
    """
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


def try_int(val):
    try:
        return int(val)
    except:
        return val


def filter_unlabeled(entities, declarations):
    """
    Get a list of declarations that were not mentioned in `entities`
    :param entities: List of entity offsets
    :param declarations: dict, where keys are declaration offsets
    :return: list of declarations that were not mentioned in `entities`
    """
    for_mask = []
    for decl in declarations:
        for e in entities:
            if overlap(decl, e):
                break
        else:
            for_mask.append(decl)
    return for_mask


def load_pkl_emb(path):
    """

    :param path:
    :return:
    """
    embedder = pickle.load(open(path, "rb"))
    if isinstance(embedder, list):
        embedder = embedder[-1]
    return embedder


class PythonBatcher:
    def __init__(
            self, data, batch_size: int, seq_len: int,
            graphmap: Dict[str, int], wordmap: Dict[str, int], tagmap: Optional[TagMap] = None,
            mask_unlabeled_declarations=True,
            class_weights=False, element_hash_size=1000
    ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.class_weights = None
        self.mask_unlabeled_declarations = mask_unlabeled_declarations

        self.nlp = create_tokenizer("spacy")
        if tagmap is None:
            self.tagmap = tag_map_from_sentences(list(zip(*[self.prepare_sent(json.dumps(sent)) for sent in data]))[1])
        else:
            self.tagmap = tagmap

        self.graphpad = len(graphmap)
        self.wordpad = len(wordmap)
        self.tagpad = self.tagmap["O"]
        self.prefpad = element_hash_size
        self.suffpad = element_hash_size

        self.graphmap_func = lambda g: graphmap.get(g, len(graphmap))
        self.wordmap_func = lambda w: wordmap.get(w, len(wordmap))
        self.tagmap_func = lambda t: self.tagmap.get(t, len(self.tagmap))
        self.prefmap_func = lambda w: token_hasher(w[:3], element_hash_size)
        self.suffmap_func = lambda w: token_hasher(w[-3:], element_hash_size)

        self.mask_unlblpad = 1.
        if mask_unlabeled_declarations:
            self.mask_unlbl_func = lambda t: 1 if t == "O" else 0
        else:
            self.mask_unlbl_func = lambda t: 1.

        self.classwpad = 1.
        if class_weights:
            self.class_weights = ClassWeightNormalizer()
            self.class_weights.init(list(zip(*[self.prepare_sent(json.dumps(sent)) for sent in data]))[1])
            self.classw_func = lambda t: self.class_weights.get(t, self.classwpad)
        else:
            self.classw_func = lambda t: 1.

    def num_classes(self):
        return len(self.tagmap)

    @lru_cache(maxsize=200000)
    def prepare_sent(self, sent):
        sent = json.loads(sent)
        text, annotations = sent

        doc = self.nlp(text)
        ents = annotations['entities']
        repl = annotations['replacements']
        if self.mask_unlabeled_declarations:
            unlabeled_dec = filter_unlabeled(ents, get_declarations(text))

        tokens = [t.text for t in doc]
        ents_tags = biluo_tags_from_offsets(doc, ents)
        repl_tags = biluo_tags_from_offsets(doc, repl)
        if self.mask_unlabeled_declarations:
            unlabeled_dec = biluo_tags_from_offsets(doc, unlabeled_dec)

        def fix_incorrect_tags(tags):
            while "-" in tags:
                tags[tags.index("-")] = "O"

        fix_incorrect_tags(ents_tags)
        fix_incorrect_tags(repl_tags)
        if self.mask_unlabeled_declarations:
            fix_incorrect_tags(unlabeled_dec)

        assert len(tokens) == len(ents_tags) == len(repl_tags)
        if self.mask_unlabeled_declarations:
            assert len(tokens) == len(unlabeled_dec)

        # decls = declarations_to_tags(doc, decls)

        repl_tags = [try_int(t.split("-")[-1]) for t in repl_tags]

        if self.mask_unlabeled_declarations:
            return tuple(tokens), tuple(ents_tags), tuple(repl_tags), tuple(unlabeled_dec)
        else:
            return tuple(tokens), tuple(ents_tags), tuple(repl_tags)

    @lru_cache(maxsize=200000)
    def create_batches_with_mask(
            self, sent: List[str], tags: List[str], repl: List[str], unlabeled_decls: Optional[List[str]]=None
    ):

        def encode(seq, encode_func, pad):
            blank = np.ones((self.seq_len,), dtype=np.int32) * pad
            encoded = np.array([encode_func(w) for w in seq], dtype=np.int32)
            blank[0:min(encoded.size, self.seq_len)] = encoded[0:min(encoded.size, self.seq_len)]
            return blank

        # input
        pref = encode(sent, self.prefmap_func, self.prefpad)
        suff = encode(sent, self.suffmap_func, self.suffpad)
        s = encode(sent, self.wordmap_func, self.wordpad)
        r = encode(repl, self.graphmap_func, self.graphpad)  # TODO test

        # labels
        t = encode(tags, self.tagmap_func, self.tagpad)

        # mask unlabeled, feed dummy no mask provided
        hidem = encode(
            list(range(len(sent))) if unlabeled_decls is None else unlabeled_decls,
            self.mask_unlbl_func, self.mask_unlblpad
        )

        # class weights
        classw = encode(tags, self.classw_func, self.classwpad)

        assert len(s) == len(r) == len(pref) == len(suff) == len(t) == len(classw) == len(hidem)

        return {
            "tok_ids": s,
            "graph_ids": r,
            "prefix": pref,
            "suffix": suff,
            "tags": t,
            "class_weights": classw,
            "hide_mask": hidem,
            "lens": len(s) if len(s) < self.seq_len else self.seq_len
        }

    def format_batch(self, batch):
        fbatch = {
            "tok_ids": [], "graph_ids": [], "prefix": [], "suffix": [],
            "tags": [], "class_weights": [], "hide_mask": [], "lens": []
        }

        for sent in batch:
            for key, val in sent.items():
                fbatch[key].append(val)

        return {key: np.stack(val) if key != "lens" else np.array(val, dtype=np.int32) for key, val in fbatch.items()}

    def generate_batches(self):
        batch = []
        for sent in self.data:
            batch.append(self.create_batches_with_mask(*self.prepare_sent(json.dumps(sent))))
            if len(batch) == self.batch_size:
                yield self.format_batch(batch)
                batch = []
        yield self.format_batch(batch)

    def __iter__(self):
        return self.generate_batches()

    def __len__(self):
        return int(ceil(len(self.data) / self.batch_size))


def scorer(pred, labels, tagmap, eps=1e-8):
    """
    Compute f1 score, precision, and recall from BILUO labels
    :param pred: predicted BILUO labels
    :param labels: ground truth BILUO labels
    :param tagmap:
    :param eps:
    :return:
    """
    # TODO
    # the scores can be underestimated because ground truth does not contain all possible labels
    # this results in higher reported false alarm rate
    pred_biluo = [tagmap.inverse(p) for p in pred]
    labels_biluo = [tagmap.inverse(p) for p in labels]

    pred_spans = set(parse_biluo(pred_biluo))
    true_spans = set(parse_biluo(labels_biluo))

    tp = len(pred_spans.intersection(true_spans))
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return precision, recall, f1


def write_config(trial_dir, params, extra_params=None):
    config_path = os.path.join(trial_dir, "model_config.conf")

    import configparser

    params = copy(params)
    if extra_params is not None:
        params.update(extra_params)

    config = configparser.ConfigParser()
    config['DEFAULT'] = params

    with open(config_path, 'w') as configfile:
        config.write(configfile)


def train_model(
        train_data, test_data, params,
        graph_emb_path=None, word_emb_path=None,
        output_dir=None, epochs=30, batch_size=32, seq_len=100, finetune=False, trials=1
):

    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    suffix_prefix_buckets = params.pop("suffix_prefix_buckets")

    train_batcher = PythonBatcher(
        train_data, batch_size, seq_len=seq_len, graphmap=graph_emb.ind, wordmap=word_emb.ind, tagmap=None,
        class_weights=False, element_hash_size=suffix_prefix_buckets
    )
    test_batcher = PythonBatcher(
        test_data, batch_size, seq_len=seq_len, graphmap=graph_emb.ind, wordmap=word_emb.ind,
        tagmap=train_batcher.tagmap,  # use the same mapping
        class_weights=False, element_hash_size=suffix_prefix_buckets  # class_weights are not used for testing
    )

    print(f"\n\n{params}")
    lr = params.pop("learning_rate")
    lr_decay = params.pop("learning_rate_decay")

    param_dir = os.path.join(output_dir, str(datetime.now()))
    os.mkdir(param_dir)

    for trial_ind in range(trials):
        trial_dir = os.path.join(param_dir, repr(trial_ind))
        os.mkdir(trial_dir)

        model = TypePredictor(
            word_emb, graph_emb, train_embeddings=finetune, suffix_prefix_buckets=suffix_prefix_buckets,
            num_classes=train_batcher.num_classes(), seq_len=seq_len, **params
        )

        train_losses, train_f1, test_losses, test_f1 = train(
            model=model, train_batches=train_batcher, test_batches=test_batcher,
            epochs=epochs, learning_rate=lr, scorer=lambda pred, true: scorer(pred, true, train_batcher.tagmap),
            learning_rate_decay=lr_decay, finetune=finetune
        )

        checkpoint_path = os.path.join(trial_dir, "checkpoint")
        model.save_weights(checkpoint_path)

        metadata = {
            "train_losses": train_losses,
            "train_f1": train_f1,
            "test_losses": test_losses,
            "test_f1": test_f1,
            "learning_rate": lr,
            "learning_rate_decay": lr_decay,
            "epochs": epochs,
            "suffix_prefix_buckets": suffix_prefix_buckets,
            "seq_len": seq_len
        }

        # write_config(trial_dir, params, extra_params={"suffix_prefix_buckets": suffix_prefix_buckets, "seq_len": seq_len})

        metadata.update(params)

        with open(os.path.join(trial_dir, "params.json"), "w") as metadata_sink:
            metadata_sink.write(json.dumps(metadata, indent=4))

        pickle.dump(train_batcher.tagmap, open(os.path.join(trial_dir, "tag_types.pkl"), "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
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
    parser.add_argument('--max_seq_len', dest='max_seq_len', default=100, type=int,
                        help='')
    # parser.add_argument('--pretrain_phase', dest='pretrain_phase', default=20, type=int,
    #                     help='')
    parser.add_argument('--epochs', dest='epochs', default=500, type=int,
                        help='')
    parser.add_argument('--trials', dest='trials', default=1, type=int,
                        help='')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('model_output',
                        help='')

    args = parser.parse_args()

    output_dir = args.model_output
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
    #            'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}

    train_data, test_data = read_data(
        open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True, include_only="entities",
        min_entity_count=5
    )

    unique_entities = get_unique_entities(train_data, field="entities")

    for params in cnn_params:
        train_model(
            train_data, test_data, params, graph_emb_path=args.graph_emb_path, word_emb_path=args.word_emb_path,
            output_dir=output_dir, epochs=args.epochs, batch_size=args.batch_size,
            finetune=args.finetune, trials=args.trials, seq_len=args.max_seq_len,
        )
