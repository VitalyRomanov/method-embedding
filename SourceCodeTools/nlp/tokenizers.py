import hashlib
import logging
from functools import lru_cache

import spacy
from spacy.tokens import Doc

from SourceCodeTools.nlp.spacy_tools.SpacyPythonBpeTokenizer import SpacyPythonBpe
from SourceCodeTools.nlp.string_tools import get_byte_to_char_map, get_byte_to_char_map2


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


def recover_offset_char_spans(tokens, offsets):
    if isinstance(tokens, AdapterDoc):
        return tokens.token_offsets_to_char_offsets(offsets)
    else:
        raise NotImplementedError()


class AdapterDoc:
    def __init__(
            self, tokens,
            # plain_tokens=None,
            original_text=None,
            reverse_tokenization_fn=None,
            tokens_as_bytes=False,
            start_token_adjustment_amount=0,
            tokens_for_alignment=None,
    ):
        """
        A simple wrapper for tokens that also stores additional data such as character span adjustment and
        tokens compatible with `biluo_tags_from_offsets`
        """
        self.tokens = tokens
        # self.adjustment_amount = 0
        # self.tokens_for_biluo_alignment = None
        # self._replacements = replacements
        self._original_text = original_text
        self._reverse_tokenization_fn = reverse_tokenization_fn
        # self._plain_tokens = plain_tokens  # without service tokens such as <s>
        self._tokens_for_alignment = tokens_for_alignment  # without service tokens such as <s>
        self._tokens_as_bytes = tokens_as_bytes
        self._start_token_adjustment_amount = start_token_adjustment_amount

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return "".join(self.tokens)

    def __len__(self):
        return len(self.tokens)

    @property
    def requires_offset_adjustment(self):
        return self._start_token_adjustment_amount > 0 or self._tokens_as_bytes

    @property
    def text(self):
        if self._reverse_tokenization_fn is None:
            r = "".join(self.tokens)
            # if self._replacements is not None:
            #     for rc, c in self._replacements.items():
            #         r = r.replace(rc, c)
        else:
            r = self._reverse_tokenization_fn(self.tokens)
        return r

    def adjust_offsets(self, offsets):
        if self._tokens_as_bytes:
            _b2c = get_byte_to_char_map(self._original_text)
            _c2b = dict(zip(_b2c.values(), _b2c.keys()))

            offsets = []
            for o in offsets:
                try:
                    offsets.append((_c2b[o[0]], _c2b[o[1]], o[2]))
                except KeyError:
                    continue
            # encountered an error which resulted in offset span being outside the length of the original string.
            # not sure about the cause, could be python bug?. let's simply skip such offsets for now.
            # offsets = [(_c2b[o[0]], _c2b[o[1]], o[2]) for o in offsets]

        if self._start_token_adjustment_amount != 0:
            offsets = [(o[0] + self._start_token_adjustment_amount, o[1] + self._start_token_adjustment_amount, o[2]) for o in offsets]

        return offsets

    def token_offsets_to_char_offsets(self, offsets):
        if len(offsets) == 0:
            return []

        sorted_offsets = sorted(offsets, key=lambda x: x[0])
        offsets = []
        current_offset = sorted_offsets.pop(0)
        start_found = False
        start_char = None
        position = 0

        cum_chars = 0
        num_tokens = len(self.tokens)
        while position < num_tokens:
            current_token_len = len(self.tokens[position])
            if start_found is False and position == current_offset[0]:
                start_char = cum_chars
                start_found = True
                position += 1
                cum_chars += current_token_len
            elif start_found is True and position == current_offset[1]:
                end_char = cum_chars
                start_found = False
                offsets.append((start_char, end_char, current_offset[2]))
                if len(sorted_offsets) == 0:
                    break
                current_offset = sorted_offsets.pop(0)
            else:
                position += 1
                cum_chars += current_token_len

        if self._start_token_adjustment_amount != 0:
            offsets = ((o[0] - self._start_token_adjustment_amount, o[1] - self._start_token_adjustment_amount, o[2]) for o in offsets)

        if self._tokens_as_bytes:
            _b2c = get_byte_to_char_map2(self._original_text)
            _c2b = dict(zip(_b2c.values(), _b2c.keys()))

            offsets = ((_b2c[o[0]], _b2c[o[1]], o[2]) for o in offsets)

        # def verify_offset(offset):
        #     token = self._original_text[offset[0]: offset[1]]
        #     original_len = len(token)
        #     if token.strip() != "":
        #         lstrip_len = original_len - len(token.lstrip())
        #         rstrip_len = original_len - len(token.rstrip())
        #         return (offset[0] + lstrip_len, offset[1] - rstrip_len, offset[2])
        #     else:
        #         return offset
        #
        # offsets = [verify_offset(offset) for offset in offsets]

        return list(offsets)

    def get_original_spans_for_tokens(self):

        token_spans = []

        assert self._start_token_adjustment_amount != 0
        last_token = len(self.tokens) - 1
        for ind, token in enumerate(self.tokens):
            if ind > 0 and ind < last_token:
                token_spans.append((ind, ind + 1, None))

        offsets = self.token_offsets_to_char_offsets(token_spans)

        offsets = [
            offset for offset in offsets if (
                (offset[0] < offset[1]) and (offset[0] >= 0) and (offset[0]) < len(self._original_text)
            )
        ]

        return offsets

    def get_tokens_for_alignment(self):
        return self._tokens_for_alignment


class CodebertAdapter:
    """
    This tokenizer returns tokens in a format that can be used with `biluo_tags_from_offsets`
    """
    def __init__(self):
        from transformers import RobertaTokenizer

        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.nlp = spacy.blank("en")
        self.regex_tok = create_tokenizer("regex")

    def primary_tokenization(self, text):
        return self.tokenizer.tokenize(text)

    def secondary_tokenization(self, tokens):
        # secondary tokenizer performs subtokenization
        # example:
        # "(arg1" -> "(", "arg1"
        new_tokens = []
        for token in tokens:
            new_tokens.extend(self.regex_tok(token))
        return new_tokens

    def __call__(self, text):
        """
        Tokenization function. Example:
            original string: 'a + b'
            codebert tokenized: '<s>', 'a', 'Ġ+', 'Ġb', '</s>'
        """
        # TODO
        # there is a bug when there is " <s> " in the input it is stripped of surrounding spaces "<s>"
        tokens = self.primary_tokenization(text)
        tokens = self.secondary_tokenization(tokens)
        doc = Doc(self.nlp.vocab, tokens, spaces=[False] * len(tokens))

        backup_tokens = doc
        fixed_spaces = [False]
        fixed_words = ["<s>"]  # add additional tokens for codebert to avoid adding them later.

        for ind, t in enumerate(doc):
            if len(t.text) > 1:
                fixed_words.append(t.text.strip("Ġ"))
            else:
                fixed_words.append(t.text)
            if ind != 0:
                fixed_spaces.append(t.text.startswith("Ġ") and len(t.text) > 1)
        fixed_spaces.append(False)
        fixed_spaces.append(False)
        fixed_words.append("</s>")

        assert len(fixed_spaces) == len(fixed_words)

        doc = Doc(self.nlp.vocab, fixed_words, fixed_spaces)

        assert len(doc) - 2 == len(backup_tokens)
        assert len(doc.text) - 7 == len(backup_tokens.text)

        final_doc = AdapterDoc(
            ["<s>"] + [t.text for t in backup_tokens] + ["</s>"],
            original_text=text,
            reverse_tokenization_fn=self.tokenizer.convert_tokens_to_string,
            tokens_as_bytes=True,
            start_token_adjustment_amount=3,
            tokens_for_alignment=doc
        )
        # final_doc.adjustment_amount = -3
        # final_doc.tokens_for_biluo_alignment = doc

        return final_doc


@lru_cache
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
            regex = "[\w0-9]+|[^\w\s]|[0-9]+"
        _tokenizer = RegexpTokenizer(regex)

        def default_tokenizer(text):
            return _tokenizer.tokenize(text)

        return default_tokenizer
    elif type == "bpe":
        if bpe_path is None:
            raise Exception("Specify path for bpe tokenizer model")

        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer

        return make_tokenizer(load_bpe_model(bpe_path))
    elif type == "codebert":
        from transformers import RobertaTokenizer
        import spacy
        from spacy.tokens import Doc

        adapter = CodebertAdapter()

        def tokenize(text):
            return adapter(text)

        return tokenize
    else:
        raise Exception("Supported tokenizer types: spacy, regex, bpe")


def codebert_to_spacy(tokens):
    backup_tokens = tokens
    fixed_spaces = [False]
    fixed_words = ["<s>"]

    for ind, t in enumerate(tokens):
        if len(t.text) > 1:
            fixed_words.append(t.text.strip("Ġ"))
        else:
            fixed_words.append(t.text)
        if ind != 0:
            fixed_spaces.append(t.text.startswith("Ġ") and len(t.text) > 1)
    fixed_spaces.append(False)
    fixed_spaces.append(False)
    fixed_words.append("</s>")

    assert len(fixed_spaces) == len(fixed_words)

    from spacy.tokens import Doc
    import spacy
    doc = Doc(spacy.blank("en").vocab, fixed_words, fixed_spaces)

    assert len(doc) - 2 == len(backup_tokens)
    assert len(doc.text) - 7 == len(backup_tokens.text)

    adjustment = -3
    # spans = [adjust_offsets(sp, -3) for sp in spans]
    return doc, adjustment


# import tokenize
# from io import BytesIO
# tokenize.tokenize(BytesIO(s.encode('utf-8')).readline)