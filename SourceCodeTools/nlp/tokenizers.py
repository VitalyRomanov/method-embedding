import hashlib

from SourceCodeTools.nlp.spacy_tools.SpacyPythonBpeTokenizer import SpacyPythonBpe


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
    """
    Create tokenizer instance. Usage

    ```
    tok = create_tokenizer("spacy") # create spacy doc
    tokens = tok("string for tokenization")

    ...

    tok = create_tokenizer("bpe") # create list of tokens
    tokens = tok("string for tokenization")
    ```

    :param type: tokenizer type is one of [spacy|spacy_bpe|regex|bpe]. Spacy creates a blank english tokenizer with additional
        tokenization rules. Regex tokenizer is a simple tokenizer from nltk that uses regular expression
        `[\w]+|[^\w\s]|[0-9]+`. BPE tokenizer is an instance of sentencepiece model (requires pretrained model).
    :param bpe_path: path for pretrained BPE model. Used for BPE tokenizer
    :param regex: Override regular expression for Regex tokenizer.
    :return: Returns spacy pipeline (nlp) or a tokenize function.
    """
    if type == "spacy":
        import spacy
        return _inject_tokenizer(spacy.blank("en"))
    elif type == "spacy_bpe":
        import spacy
        nlp = spacy.blank("en")

        if bpe_path is None:
            raise Exception("Specify path for bpe tokenizer model")

        nlp.tokenizer = SpacyPythonBpe(bpe_path, nlp.vocab)
        return nlp
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

# import tokenize
# from io import BytesIO
# tokenize.tokenize(BytesIO(s.encode('utf-8')).readline)