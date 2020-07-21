from spacy.tokenizer import Tokenizer
import re, json

def custom_tokenizer(nlp):
    prefix_re = re.compile(r'''[\[*]''')
    suffix_re = re.compile(r'''[\]]''')
    infix_re = re.compile(r'''[\[\]\(\),=*]''')
    return Tokenizer(nlp.vocab,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                )

def inject_tokenizer(nlp):
    nlp.tokenizer = custom_tokenizer(nlp)
    return nlp

def read_data(data_path):
    import random

    TRAIN_DATA = []
    with open(data_path, "r") as data:
        for line in data:
            entry = json.loads(line)
            TRAIN_DATA.append([entry['text'], {'entities': entry['ents']}])
            TRAIN_DATA[-1][1]['entities'] = [(int(e[0]), int(e[1]), e[2]) for e in TRAIN_DATA[-1][1]['entities']]
    random.shuffle(TRAIN_DATA)
    train_size = int(len(TRAIN_DATA) * 0.8)
    train, test = TRAIN_DATA[:train_size], TRAIN_DATA[train_size:]
    return train, test

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