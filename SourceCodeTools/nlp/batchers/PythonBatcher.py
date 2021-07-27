import json
import os
import shelve
import tempfile
from functools import lru_cache
from math import ceil
from typing import Dict, Optional, List

from spacy.gold import biluo_tags_from_offsets

from SourceCodeTools.code.ast_tools import get_declarations
from SourceCodeTools.models.ClassWeightNormalizer import ClassWeightNormalizer
from SourceCodeTools.nlp import create_tokenizer, tag_map_from_sentences, TagMap, token_hasher, try_int
from SourceCodeTools.nlp.entity import fix_incorrect_tags
from SourceCodeTools.nlp.entity.utils import overlap

import numpy as np

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


class PythonBatcher:
    def __init__(
            self, data, batch_size: int, seq_len: int,
            wordmap: Dict[str, int], *, graphmap: Optional[Dict[str, int]], tagmap: Optional[TagMap] = None,
            mask_unlabeled_declarations=True,
            class_weights=False, element_hash_size=1000
    ):

        self.create_cache()

        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.class_weights = None
        self.mask_unlabeled_declarations = mask_unlabeled_declarations

        self.nlp = create_tokenizer("spacy")
        if tagmap is None:
            self.tagmap = tag_map_from_sentences(list(zip(*[self.prepare_sent(sent) for sent in data]))[1])
        else:
            self.tagmap = tagmap

        self.graphpad = len(graphmap) if graphmap is not None else None
        self.wordpad = len(wordmap)
        self.tagpad = self.tagmap["O"]
        self.prefpad = element_hash_size
        self.suffpad = element_hash_size

        self.graphmap_func = lambda g: graphmap.get(g, len(graphmap)) if graphmap is not None else None
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
            self.class_weights.init(list(zip(*[self.prepare_sent(sent) for sent in data]))[1])
            self.classw_func = lambda t: self.class_weights.get(t, self.classwpad)
        else:
            self.classw_func = lambda t: 1.

    def __del__(self):
        self.sent_cache.close()
        self.batch_cache.close()

    def create_cache(self):
        # self.tmp_dir = tempfile.TemporaryDirectory()
        # self.sent_cache = shelve.open(os.path.join(self.tmp_dir.name, "sent_cache.db"))
        # self.batch_cache = shelve.open(os.path.join(self.tmp_dir.name, "batch_cache.db"))
        self.sent_cache = shelve.open("sent_cache.db")
        self.batch_cache = shelve.open("batch_cache.db")

    def num_classes(self):
        return len(self.tagmap)

    # @lru_cache(maxsize=200000)
    def prepare_sent(self, sent):

        sent_json = json.dumps(sent)
        if sent_json in self.sent_cache:
            return self.sent_cache[sent_json]

        # sent = json.loads(sent)
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
            output = tuple(tokens), tuple(ents_tags), tuple(repl_tags), tuple(unlabeled_dec)
        else:
            output = tuple(tokens), tuple(ents_tags), tuple(repl_tags)

        self.sent_cache[sent_json] = output
        return output

    # @lru_cache(maxsize=200000)
    def create_batches_with_mask(
            self, sent: List[str], tags: List[str], repl: List[str], unlabeled_decls: Optional[List[str]]=None
    ):

        input_json = json.dumps((sent, tags, repl, unlabeled_decls))
        if input_json in self.batch_cache:
            return self.batch_cache[input_json]

        def encode(seq, encode_func, pad):
            blank = np.ones((self.seq_len,), dtype=np.int32) * pad
            encoded = np.array([encode_func(w) for w in seq], dtype=np.int32)
            blank[0:min(encoded.size, self.seq_len)] = encoded[0:min(encoded.size, self.seq_len)]
            return blank

        # input
        pref = encode(sent, self.prefmap_func, self.prefpad)
        suff = encode(sent, self.suffmap_func, self.suffpad)
        s = encode(sent, self.wordmap_func, self.wordpad)
        r = encode(repl, self.graphmap_func, self.graphpad) if self.graphmap_func is not None else None

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

        output = {
            "tok_ids": s,
            # "graph_ids": r,
            "prefix": pref,
            "suffix": suff,
            "tags": t,
            "class_weights": classw,
            "hide_mask": hidem,
            "lens": len(s) if len(s) < self.seq_len else self.seq_len
        }

        if r is not None:
            output["graph_ids"] = r

        self.batch_cache[input_json] = output
        return output

    def format_batch(self, batch):
        fbatch = {
            "tok_ids": [], "graph_ids": [], "prefix": [], "suffix": [],
            "tags": [], "class_weights": [], "hide_mask": [], "lens": []
        }

        for sent in batch:
            for key, val in sent.items():
                fbatch[key].append(val)

        if len(fbatch["graph_ids"]) == 0:
            fbatch.pop("graph_ids")

        return {key: np.stack(val) if key != "lens" else np.array(val, dtype=np.int32) for key, val in fbatch.items()}

    def generate_batches(self):
        batch = []
        for sent in self.data:
            batch.append(self.create_batches_with_mask(*self.prepare_sent(sent)))
            if len(batch) >= self.batch_size:
                yield self.format_batch(batch)
                batch = []
        yield self.format_batch(batch)

    def __iter__(self):
        return self.generate_batches()

    def __len__(self):
        return int(ceil(len(self.data) / self.batch_size))