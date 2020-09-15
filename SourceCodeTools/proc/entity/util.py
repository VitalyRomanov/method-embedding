from spacy.tokenizer import Tokenizer
import re, json

from spacy.gold import biluo_tags_from_offsets, offsets_from_biluo_tags

import hashlib

def el_hash(el, buckets):
    return int(hashlib.md5(el.encode('utf-8')).hexdigest(), 16) % buckets

def custom_tokenizer(nlp):
    prefix_re = re.compile(r'''^[\[\(\{"':\.!@~,=+-/\*]''')
    suffix_re = re.compile(r'''[\]\)\}"':\.\!~,=+-/\*]$''')
    infix_re = re.compile(r'''[\[\]\(\)\{\},=+-/@\*\"\.!:~]''')
    return Tokenizer(nlp.vocab,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                )

def inject_tokenizer(nlp):
    nlp.tokenizer = custom_tokenizer(nlp)
    return nlp

def normalize_entities(entities):
    norm = lambda x: x.split("[")[0].split(".")[-1]

    return [(e[0], e[1], norm(e[2])) for e in entities]

def overlap(p1, p2):
    if (p2[1] - p1[0]) * (p2[0] - p1[1]) <= 0:
        return True
    else:
        return False

def resolve_repeats(entities):
    ents = []

    for e in entities:
        overlaps = False
        for ee in ents:
            if overlap((e[0], e[1]), (ee[0], ee[1])):
                overlaps = True
        if overlaps is False:
            ents.append(e)

    return ents

def filter_allowed(ents, allowed=None):
    if allowed is None:
        return ents
    else:
        # replace entitties not in the allowed list with "Other"
        return [e if e[2] in allowed else (e[0], e[1], "Other") for e in ents]


def read_data(data_path, normalize=False, include_replacements=False, allowed=None):
    import random

    TRAIN_DATA = []
    entities_in_train = set()

    with open(data_path, "r") as data:
        for line in data:
            entry = json.loads(line)
            TRAIN_DATA.append([entry['text'], {'entities': filter_allowed(entry['ents'], allowed=allowed)}])

            if len(TRAIN_DATA[-1][1]['entities']) == 0:
                TRAIN_DATA.pop(-1)
                continue

            if include_replacements:
                if "replacements" in entry:
                    TRAIN_DATA[-1][1]['replacements'] = resolve_repeats(entry['replacements'])
            if normalize:
                TRAIN_DATA[-1][1]['entities'] = normalize_entities(TRAIN_DATA[-1][1]['entities'])
            TRAIN_DATA[-1][1]['entities'] = [(int(e[0]), int(e[1]), e[2]) for e in TRAIN_DATA[-1][1]['entities']]

            for e in TRAIN_DATA[-1][1]['entities']:
                entities_in_train.add(e[2])

    random.shuffle(TRAIN_DATA)
    train_size = int(len(TRAIN_DATA) * 0.8)
    train, test = TRAIN_DATA[:train_size], TRAIN_DATA[train_size:]

    entities_in_test = set()
    for tr in test:
        evict = set()

        for ind, e in enumerate(tr[1]['entities']):
            if e[2] not in entities_in_train:
                evict.add(ind)

        tr[1]['entities'] = [e for ind, e in enumerate(tr[1]['entities']) if ind not in evict]

        for e in tr[1]['entities']:
            entities_in_test.add(e[2])

    test = [sent for sent in test if len(sent[1]['entities']) > 0]

    for tr in test:
        assert len(tr[1]['entities']) > 0

    assert len(entities_in_test - entities_in_train) == 0

    return train, test


def deal_with_incorrect_offsets(sents, nlp):

    for ind in range(len(sents)):
        doc = nlp(sents[ind][0])

        tags = biluo_tags_from_offsets(doc, sents[ind][1]['entities'])

        sents[ind][1]['entities'] = offsets_from_biluo_tags(doc, tags)

        if "replacements" in sents[ind][1]:
            tags = biluo_tags_from_offsets(doc, sents[ind][1]['replacements'])
            while "-" in tags:
                tags[tags.index("-")] = "O"
            sents[ind][1]['replacements'] = offsets_from_biluo_tags(doc, tags)

    return sents


def read_data_classes(data_path):
    import random

    TRAIN_DATA = []
    with open(data_path, "r") as data:
        for line in data:
            entry = json.loads(line)
            if entry['cats']:
                TRAIN_DATA.append([entry['text'].split("\n")[0], {entry['cats'][0]['returns']: 1.0}])
    random.shuffle(TRAIN_DATA)
    train_size = int(len(TRAIN_DATA) * 0.8)
    train, test = TRAIN_DATA[:train_size], TRAIN_DATA[train_size:]
    return train, test