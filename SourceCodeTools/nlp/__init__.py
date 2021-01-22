import hashlib


def token_hasher(token: str, buckets: int):
    return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % buckets


def _custom_tokenizer(nlp):
    import re
    from spacy.tokenizer import Tokenizer

    prefix_re = re.compile(r"^[^\w\s]")
    infix_re = re.compile(r"[^\w\s]")
    suffix_re = re.compile(r"[^\w\s]$")
    return Tokenizer(
        nlp.vocab, prefix_search=prefix_re.search,
        suffix_search=suffix_re.search, infix_finditer=infix_re.finditer
    )


def _inject_tokenizer(nlp):
    nlp.tokenizer = _custom_tokenizer(nlp)
    return nlp


def create_tokenizer(type, bpe_path=None, regex=None):
    if type == "spacy":
        import spacy
        return _inject_tokenizer(spacy.blank("en"))
    elif type == "regex":
        from nltk import RegexpTokenizer
        if regex is None:
            regex = "[\w]+|[^\w\s]|[0-9]+"
        _tokenizer = RegexpTokenizer(regex)

        def default_tokenizer(text):
            return _tokenizer.tokenize(text)

        return default_tokenizer
    elif type == "bpe":
        if bpe_path is None:
            raise Exception("Specify path for bpe tokenizer model")

        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer

        return make_tokenizer(load_bpe_model(bpe_path))
    else:
        raise Exception("Supported tokenizer types: spacy, regex, bpe")
