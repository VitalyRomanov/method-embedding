from spacy.tokenizer import Tokenizer
import re

def custom_tokenizer(nlp):
    prefix_re = re.compile(r'''[\[*]''')
    suffix_re = re.compile(r'''[\]]''')
    infix_re = re.compile(r'''[\[\]\(\),=*]''')
    return Tokenizer(nlp.vocab,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                )
