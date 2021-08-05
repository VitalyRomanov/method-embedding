import logging
from collections import Counter
from typing import Iterable

from spacy.tokenizer import Tokenizer
import re, json

from spacy.gold import biluo_tags_from_offsets, offsets_from_biluo_tags


# def custom_tokenizer(nlp):
#     prefix_re = re.compile(r"^[^\w\s]")
#     infix_re = re.compile(r"[^\w\s]")
#     suffix_re = re.compile(r"[^\w\s]$")
#     return Tokenizer(
#         nlp.vocab, prefix_search=prefix_re.search,
#         suffix_search=suffix_re.search, infix_finditer=infix_re.finditer
#     )
#
#
# def inject_tokenizer(nlp):
#     nlp.tokenizer = custom_tokenizer(nlp)
#     return nlp


def normalize_entities(entities):
    norm = lambda x: x.strip("\"").strip("'").split("[")[0].split(".")[-1]

    if len(entities) == 0:
        return entities

    if type(entities[0]) is not str:
        # normalize entities with spans
        return [(e[0], e[1], norm(e[2])) for e in entities]
    else:
        # normalize categories
        return [norm(e) for e in entities]


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


def format_like(entity, new_entity):
    if type(entity) is not str:
        return entity[0], entity[1], new_entity
    else:
        return new_entity


def filter_allowed(ents, allowed=None, behaviour=None):

    assert behaviour in {"rename", "drop"}, f"Parameters `behaviour` should be either `rename` or `drop`"

    if allowed is None:
        return ents
    else:
        # replace entities not in the allowed list with "Other"
        filtered_entities = []
        for entity in ents:
            ent_name = get_entity_name(entity)
            if ent_name in allowed:
                filtered_entities.append(entity)
            else:
                if behaviour == "rename":
                    filtered_entities.append(format_like(entity, "Other"))
        return filtered_entities


def get_entity_name(entity):
    if type(entity) is str:
        # this is category name
        return entity
    else:
        # this is NER entity
        return entity[2]


def format_record(entry, field, allowed):
    source_field = "ents" if field == "entities" else "cats"
    return [
        entry['text'], {
            field: filter_allowed(entry[source_field], allowed=allowed, behaviour="rename")
        }
    ]


def filter_infrequent(train_data, entities_in_dataset, field, min_entity_count, behaviour):

    if min_entity_count is None:
        return

    allowed = set(entity for entity, count in entities_in_dataset.items() if count >= min_entity_count)

    filter_entities(train_data, field=field, allowed=allowed, behaviour=behaviour)


def get_unique_entities(data, field):
    entities_in_train = set()
    for _, annotations in data:
        for entity in annotations[field]:
            entities_in_train.add(get_entity_name(entity))
    return entities_in_train


def filter_entities(data, field, allowed, behaviour):
    unique_entities = set()
    for record in data:
        record[1][field] = filter_allowed(record[1][field], allowed=allowed, behaviour=behaviour)

        for entity in record[1][field]:
            unique_entities.add(get_entity_name(entity))

    test = [sent for sent in data if len(sent[1][field]) > 0]

    for record in test:
        assert len(record[1][field]) > 0

    if behaviour == "drop":
        assert len(unique_entities - allowed) == 0
    else:
        _l = len(unique_entities - allowed)
        assert _l <= 1
        if _l == 1:
            assert unique_entities - allowed == {"Other"}


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