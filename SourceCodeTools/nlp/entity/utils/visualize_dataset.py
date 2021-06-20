import os.path

import spacy
import sys
import json
from spacy.gold import biluo_tags_from_offsets
from spacy.tokenizer import Tokenizer
import re

from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.entity import parse_biluo
from SourceCodeTools.nlp.entity.entity_render import render_annotations

annotations_path = sys.argv[1]
output_path = os.path.dirname(annotations_path)

data = []
references = []
results = []

nlp = create_tokenizer("spacy")

with open(annotations_path) as annotations:
    for line in annotations:
        entry = json.loads(line.strip())

        doc = nlp(entry['text'])
        tags_r = parse_biluo(biluo_tags_from_offsets(doc, entry['replacements']))
        tags_e = parse_biluo(biluo_tags_from_offsets(doc, entry['ents']))

        data.append([entry['text']])
        references.append(tags_r)
        results.append(tags_e)

html = render_annotations(zip(data, references, results))
open(os.path.join(output_path, "annotations.html"), "w").write(html)