from SourceCodeTools.nlp.tokenizers import token_hasher, create_tokenizer

from SourceCodeTools.nlp.TagMap import *


def try_int(val):
    try:
        return int(val)
    except:
        return val