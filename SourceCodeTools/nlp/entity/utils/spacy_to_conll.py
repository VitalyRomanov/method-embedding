import pickle, sys, json, re
import spacy
from spacy.gold import biluo_tags_from_offsets

from SourceCodeTools.nlp import create_tokenizer

nlp = create_tokenizer("spacy")

TRAIN_DATA = []
with open(sys.argv[1], "r") as data:
    for line in data:
        entry = json.loads(line)
        TRAIN_DATA.append([entry['text'], {'entities': entry['ents']}])
        TRAIN_DATA[-1][1]['entities'] = [(int(e[0]), int(e[1]), e[2]) for e in TRAIN_DATA[-1][1]['entities']]


for text, ent in TRAIN_DATA:
    doc = nlp(text)
    entities = ent['entities']
    tags = biluo_tags_from_offsets(doc, entities)
    for token, tag in zip(doc, tags):
        print(token.text, tag, sep="\t")
    print("\t")
    # TODO
    # filter valid
    # if text.startswith("def format_percentiles("):
    #     print("-" in tags)
    #     print(tags)
    #     print(entities)
    #
    #     # for t in doc:
    #     #
    #     #     if t.is_space: continue
    #     #     print(t.text, tags[t.i])
    #     #     if t.text == '.':
    #     #         print()

