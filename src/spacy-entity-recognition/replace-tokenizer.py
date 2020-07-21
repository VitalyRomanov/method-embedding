import spacy
import sys
from util import inject_tokenizer

nlp = spacy.load(sys.argv[1])
nlp = inject_tokenizer(nlp)
nlp.to_disk(sys.argv[2])

