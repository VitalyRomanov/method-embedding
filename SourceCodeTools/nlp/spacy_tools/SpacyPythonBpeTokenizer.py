from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
import tokenize
from io import BytesIO


class SpacyPythonBpe(Tokenizer):
    def __init__(self, bpe_model, vocab, *args, **kwargs):
        super(SpacyPythonBpe, self).__init__(vocab, *args, **kwargs)

        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer

        self.bpe_tokenizer = make_tokenizer(load_bpe_model(bpe_model))
        self.__vocab = vocab

    def __call__(self, text):
        # https://github.com/explosion/spaCy/blob/master/spacy/tokens/doc.pyx#104
        python_tokens = list(tokenize.tokenize(BytesIO(text.encode('utf-8')).readline))
        tokens = []
        spaces = []
        in_python = []
        indents = []
        for ind, token in enumerate(python_tokens):

            if ind == 0:
                tokens.append(token.string)
                in_python.append(True)
                continue

            cur_tok_start = token.start
            prev_tok_end = python_tokens[ind-1].end

            if token.type == 6:  # dedent
                indents.pop(-1)
                continue

            if token.type == 5:  # indent
                indents.append(token.string)
                tokens.append(token.string)
                in_python.append(True)

            else:
                if prev_tok_end[0] != cur_tok_start[0]:  # different lines
                    if token.type != 61:  # NL
                        if len(indents) > 0:
                            tokens.append(indents[-1])
                            spaces.append(False)
                            in_python.append(False)
                    else:
                        tokens.append(token.line)
                        spaces.append(False)
                        in_python.append(False)
                        continue

                tokens.append(token.string)
                in_python.append(True)

            if prev_tok_end[0] != cur_tok_start[0]:  # different lines
                spaces.append(False)
            elif prev_tok_end[1] == cur_tok_start[1]:  # no space
                spaces.append(False)
            elif cur_tok_start[1] - prev_tok_end[1] == 1:
                assert token.line.encode("utf-8")[prev_tok_end[1]:cur_tok_start[1]].decode("utf-8") == " "
                spaces.append(True)
            elif cur_tok_start[1] - prev_tok_end[1] > 1:
                whitespace = token.line.encode("utf-8")[prev_tok_end[1]:cur_tok_start[1]].decode("utf-8")
                spaces.append(False)
                tokens.append(whitespace)
                in_python.append(False)
                spaces.append(False)
            else:
                raise NotImplementedError()
        spaces.append(False)

        # tokens = self.bpe_tokenizer(text)
        # spaces = []
        # for ind, token in enumerate(tokens):
        #     if ind == 0:
        #         pass
        #     else:
        #         if token.startswith("▁"):
        #             spaces.append(True)
        #         else:
        #             spaces.append(False)
        # spaces.append(False)
        # tokens = [token.replace("▁", "") for token in tokens]
        tokens, spaces = zip(*[(token, space) for token, space in zip(tokens, spaces) if token not in {"", "utf-8"}])
        doc = Doc(self.__vocab, words=tokens, spaces=spaces)
        return doc


def test_SpacyPythonBpe():
    from SourceCodeTools.nlp import create_tokenizer
    nlp = create_tokenizer("spacy_bpe", bpe_path="/Users/LTV/Dropbox (Personal)/sentencepiece_bpe.model")

    # code = """    def method2(self) :
    #     variable1 = self.field
    #     variable2 = str(variable1)
    #     return variable2"""

    code = """    def method2(self) :
 
        variable1 = self.field
        variable2 = str(variable1)
        return variable2"""

    doc = nlp(code)

    assert str(doc) == code

    print(doc)