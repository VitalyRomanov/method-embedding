import json
from functools import lru_cache
from typing import Dict, Optional, List

import numpy as np
from spacy.gold import biluo_tags_from_offsets

from SourceCodeTools.code.ast.ast_tools import get_declarations
from SourceCodeTools.nlp import TagMap, try_int
from SourceCodeTools.nlp.batchers import PythonBatcher
from SourceCodeTools.nlp.entity import fix_incorrect_tags
from SourceCodeTools.nlp.entity.utils import overlap


def filter_declarations(entities, declarations):
    valid = {}

    for decl in declarations:
        for e in entities:
            if overlap(decl, e):
                valid[decl] = declarations[decl]

    return valid


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

        fix_incorrect_tags(tag_mentions)

        # if "-" in tag_mentions:
        #     for t, tag in zip(doc, tag_mentions):
        #         print(t, tag, sep="\t")

        declarations.append((tag_decl, tag_mentions))

    return declarations


class PythonBatcherMentions(PythonBatcher):
    def __init__(
            self, data, batch_size: int, seq_len: int,
            graphmap: Dict[str, int], wordmap: Dict[str, int], tagmap: Optional[TagMap] = None,
            mask_unlabeled_declarations=True,
            class_weights=False, element_hash_size=1000, **kwargs
    ):
        super(PythonBatcherMentions, self).__init__(
            data, batch_size, seq_len=seq_len,
            graphmap=graphmap, wordmap=wordmap, tagmap=tagmap, mask_unlabeled_declarations=mask_unlabeled_declarations,
            class_weights=class_weights, element_hash_size=element_hash_size
        )

        self.declpad = 0
        self.mentpad = 0

        self.declmap_func = lambda d: 0 if d == "O" else 1
        self.mentmap_func = lambda d: 0 if d == "O" else 1
        self.tagmap_func = lambda t_dec: tagmap[t_dec[0]] if t_dec[1] != "O" else tagmap["O"]

    @lru_cache(maxsize=200000)
    def prepare_sent(self, sent):
        sent = json.loads(sent)
        text, annotations = sent

        doc = self.nlp(text)
        ents = annotations['entities']
        repl = annotations['replacements']
        decls = filter_declarations(ents, get_declarations(text))

        tokens = [t.text for t in doc]
        ents_tags = biluo_tags_from_offsets(doc, ents)
        repl_tags = biluo_tags_from_offsets(doc, repl)
        decls = declarations_to_tags(doc, decls)

        fix_incorrect_tags(ents_tags)
        fix_incorrect_tags(repl_tags)

        assert len(tokens) == len(ents_tags) == len(repl_tags)

        repl_tags = [try_int(t.split("-")[-1]) for t in repl_tags]

        return tokens, ents_tags, repl_tags, decls

    @lru_cache(maxsize=200000)
    def create_batches_with_mask(
            self, sent: List[str], tags: List[str], repl: List[str], decls_mentions: Optional[List[str]] = None
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

        # class weights
        classw = encode(tags, self.classw_func, self.classwpad)

        # declarations, mentions, labels
        for dec, men in decls_mentions:
            targ = encode(dec, self.declmap_func, self.declpad)
            ment = encode(dec, self.mentmap_func, self.mentpad)
            t = encode(zip(tags, dec), self.tagmap_func, self.tagpad)

            assert len(s) == len(r) == len(pref) == len(suff) == len(t) == len(classw) == len(targ) == len(ment)

            yield {
                "tok_ids": s,
                "graph_ids": r,
                "prefix": pref,
                "suffix": suff,
                "target": targ,
                "mentions": ment,
                "tags": t,
                "class_weights": classw,
                "lens": len(s) if len(s) < self.seq_len else self.seq_len
            }