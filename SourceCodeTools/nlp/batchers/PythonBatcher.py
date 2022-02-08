import json
import os
import shelve
import shutil
import tempfile
from collections import defaultdict
from time import time
from math import ceil
from typing import Dict, Optional, List

import spacy

from SourceCodeTools.code.ast.ast_tools import get_declarations
from SourceCodeTools.models.ClassWeightNormalizer import ClassWeightNormalizer
from SourceCodeTools.nlp import create_tokenizer, tag_map_from_sentences, TagMap, token_hasher, try_int
from SourceCodeTools.nlp.entity import fix_incorrect_tags
from SourceCodeTools.code.annotator_utils import adjust_offsets, biluo_tags_from_offsets
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


def print_token_tag(doc, tags):
    for t, tag in zip(doc, tags):
        print(t, "\t", tag)


class PythonBatcher:
    def __init__(
            self, data, batch_size: int, seq_len: int,
            wordmap: Dict[str, int], *, graphmap: Optional[Dict[str, int]], tagmap: Optional[TagMap] = None,
            mask_unlabeled_declarations=True,
            class_weights=False, element_hash_size=1000, len_sort=True, tokenizer="spacy", no_localization=False
    ):

        self.create_cache()

        self.data = sorted(data, key=lambda x: len(x[0])) if len_sort else data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.class_weights = None
        self.mask_unlabeled_declarations = mask_unlabeled_declarations
        self.tokenizer = tokenizer
        if tokenizer == "codebert":
            self.vocab = spacy.blank("en").vocab
        self.no_localization = no_localization

        self.nlp = create_tokenizer(tokenizer)
        if tagmap is None:
            self.tagmap = tag_map_from_sentences(list(zip(*[self.prepare_sent(sent) for sent in data]))[1])
        else:
            self.tagmap = tagmap

        self.graphpad = len(graphmap) if graphmap is not None else None
        self.wordpad = len(wordmap)
        self.tagpad = self.tagmap["O"]
        self.prefpad = element_hash_size
        self.suffpad = element_hash_size

        self.graphmap_func = (lambda g: graphmap.get(g, len(graphmap))) if graphmap is not None else None
        self.wordmap_func = lambda w: wordmap.get(w, len(wordmap))
        self.tagmap_func = lambda t: self.tagmap.get(t, self.tagmap["O"])
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

        from shutil import rmtree
        rmtree(self.tmp_dir, ignore_errors=True)

    def create_cache(self):
        char_ranges = [chr(i) for i in range(ord("a"), ord("a")+26)] + [chr(i) for i in range(ord("A"), ord("A")+26)] + [chr(i) for i in range(ord("0"), ord("0")+10)]
        from random import sample
        rnd_name = "".join(sample(char_ranges, k=10)) + str(int(time() * 1e6))
        time()

        self.tmp_dir = os.path.join(tempfile.gettempdir(), rnd_name)
        if os.path.isdir(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.mkdir(self.tmp_dir)

        self.sent_cache = shelve.open(os.path.join(self.tmp_dir, "sent_cache"))
        self.batch_cache = shelve.open(os.path.join(self.tmp_dir, "batch_cache"))

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

        if self.tokenizer == "codebert":
            backup_tokens = doc
            fixed_spaces = [False]
            fixed_words = ["<s>"]

            for ind, t in enumerate(doc):
                if len(t.text) > 1:
                    fixed_words.append(t.text.strip("Ġ"))
                else:
                    fixed_words.append(t.text)
                if ind != 0:
                    fixed_spaces.append(t.text.startswith("Ġ") and len(t.text) > 1)
            fixed_spaces.append(False)
            fixed_spaces.append(False)
            fixed_words.append("</s>")

            assert len(fixed_spaces) == len(fixed_words)

            from spacy.tokens import Doc
            doc = Doc(self.vocab, fixed_words, fixed_spaces)

            assert len(doc) - 2 == len(backup_tokens)
            assert len(doc.text) - 7 == len(backup_tokens.text)
            ents = adjust_offsets(ents, -3)
            repl = adjust_offsets(repl, -3)
            if self.mask_unlabeled_declarations:
                unlabeled_dec = adjust_offsets(unlabeled_dec, -3)

        ents_tags = biluo_tags_from_offsets(doc, ents, self.no_localization)
        repl_tags = biluo_tags_from_offsets(doc, repl, self.no_localization)
        if self.mask_unlabeled_declarations:
            unlabeled_dec = biluo_tags_from_offsets(doc, unlabeled_dec, self.no_localization)

        fix_incorrect_tags(ents_tags)
        fix_incorrect_tags(repl_tags)
        if self.mask_unlabeled_declarations:
            fix_incorrect_tags(unlabeled_dec)

        if self.tokenizer == "codebert":
            tokens = ["<s>"] + [t.text for t in backup_tokens] + ["</s>"]

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
        ).astype(np.bool)

        # class weights
        classw = encode(tags, self.classw_func, self.classwpad)

        assert len(s) == len(pref) == len(suff) == len(t) == len(classw) == len(hidem)
        if r is not None:
            assert len(r) == len(s)

        no_localization_mask = np.array([tag != self.tagpad for tag in t]).astype(np.bool)

        output = {
            "tok_ids": s,
            "replacements": repl,
            # "graph_ids": r,
            "prefix": pref,
            "suffix": suff,
            "tags": t,
            "class_weights": classw,
            "hide_mask": hidem,
            "no_loc_mask": no_localization_mask,
            "lens": len(sent) if len(sent) < self.seq_len else self.seq_len
        }

        if r is not None:
            output["graph_ids"] = r

        self.batch_cache[input_json] = output
        return output

    def format_batch(self, batch):
        # fbatch = {
        #     "tok_ids": [], "graph_ids": [], "prefix": [], "suffix": [],
        #     "tags": [], "class_weights": [], "hide_mask": [], "lens": [], "replacements": []
        # }
        fbatch = defaultdict(list)

        for sent in batch:
            for key, val in sent.items():
                fbatch[key].append(val)

        if len(fbatch["graph_ids"]) == 0:
            fbatch.pop("graph_ids")

        max_len = max(fbatch["lens"])

        return {
            key: np.stack(val)[:,:max_len] if key != "lens" and key != "replacements"
            else (np.array(val, dtype=np.int32) if key != "replacements" else np.array(val)) for key, val in fbatch.items()}

    def generate_batches(self):
        batch = []
        for sent in self.data:
            batch.append(self.create_batches_with_mask(*self.prepare_sent(sent)))
            if len(batch) >= self.batch_size:
                yield self.format_batch(batch)
                batch = []
        if len(batch) > 0:
            yield self.format_batch(batch)
        # yield self.format_batch(batch)

    def __iter__(self):
        return self.generate_batches()

    def __len__(self):
        return int(ceil(len(self.data) / self.batch_size))