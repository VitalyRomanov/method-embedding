import pandas as pd
import ast, astor
import sys, os
import json
import spacy
import re

from spacy.gold import biluo_tags_from_offsets
from spacy.tokenizer import Tokenizer


def custom_tokenizer(nlp):
    prefix_re = re.compile(r'''[\[*]''')
    suffix_re = re.compile(r'''[\]]''')
    infix_re = re.compile(r'''[\[\]\(\),=*]''')
    return Tokenizer(nlp.vocab,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                )

nlp = spacy.blank("en")
nlp.tokenizer = custom_tokenizer(nlp)
tokenizer = nlp.Defaults.create_tokenizer(nlp)

allowed = {['str','bool','Optional','None','int','Any','Union','List','Dict','Callable','ndarray','FrameOrSeries','bytes','DataFrame','Matcher','float','Tuple','bool_t','Description','Type']}

def preprocess(ent):
    return ent.strip("\"").split("[")[0].split(".")[-1]

def inspect_fdef(node):
    if node.returns is not None:
        return [{"name": "returns", "line": node.returns.lineno, "col_offset": node.returns.col_offset, "end_col_offset": node.returns.end_col_offset}]
    else:
        return []

def inspect_arg(node):
    if node.annotation is not None:
        return [{"name": "annotation", "line": node.annotation.lineno, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset}]
    else:
        return []

def inspect_ann(node):
    if node.annotation is not None:
        return [{"name": "annotation", "line": node.annotation.lineno, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset}]
    else:
        return []

def get_cum_lens(body):
    body_lines = body.split("\n")
    cum_lens = [0]
    for ind, line in enumerate(body_lines):
        cum_lens.append(len(line) + cum_lens[-1] + 1)
    return cum_lens


def get_initial_labels(body_):
    try:
        root = ast.parse(body_)
    except:
        return None

    positions = []
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            positions.extend(inspect_fdef(node))
        elif isinstance(node, ast.arg):
            positions.extend(inspect_arg(node))
        elif isinstance(node, ast.AnnAssign):
            positions.extend(inspect_ann(node))

    if positions:
        df = pd.DataFrame(positions)
        df.sort_values(by=['line', 'end_col_offset'], ascending=[True, False], inplace=True)
        return df
    else:
        return None

def strip_docstring(body):
    # TODO
    # this does not seem to be working well
    root = ast.parse(body)
    main_doc = None
    ranges = []
    for node in ast.walk(root):
        try:
            docstring = ast.get_docstring(node)
        except:
            continue
        else:
            if docstring is not None:
                ranges.append((node.body[0].lineno-1, node.body[0].end_lineno))
                # if main_doc is None:
                #     main_doc = docstring
                # node.body = node.body[1:]

    ranges = sorted(ranges, key=lambda x: x[0], reverse=True)

    body_lines = body.split("\n")

    new_body = []
    new_doc = []

    block = False
    for line in body_lines:
        if not block and line.lstrip().startswith('"""') or line.lstrip().startswith("'''"):
            block = True
            continue

        if block:
            new_doc.append(line)
        else:
            new_body.append(line)

        if block and line.rstrip().endswith('"""') or line.rstrip().endswith("'''"):
            block = False
            continue

    new_body = "\n".join(new_body)
    new_doc = "\n".join(new_doc)

    return new_body, new_doc
    # return astor.to_source(root), main_doc

def isvalid(text, ents):
    doc = nlp(text)
    tags = biluo_tags_from_offsets(doc, ents)
    if "-" in tags:
        return False
    else:
        return True

def process_body(body, remove_docstring=True):
    body_ = body.strip()

    entry = {"ents": [], "cats": []}

    if remove_docstring:
        only_body, doc = strip_docstring(body_)
        entry['text'] = only_body
        entry['docstring'] = doc

    body_ = only_body

    initial_labels = get_initial_labels(body_)

    if initial_labels is not None:

        body_lines = body_.split("\n")



        for ind, row in initial_labels.iterrows():
            line = row.line - 1

            # TODO
            # multiline annotations are not parsed correctly

            annotation = body_lines[line][row.col_offset: row.end_col_offset]
            annotation = preprocess(annotation)
            if annotation not in allowed:
                continue
            tail = body_lines[line][row.end_col_offset:]
            head = body_lines[line][:row.col_offset].rstrip()
            before_contraction = len(body_lines[line])

            if row['name'] == "returns":
                assert head.endswith("->")
                head = head[:-2]
                if line == 0: # only use labels for the main and not nested functions
                    entry["cats"].append({"returns": annotation})

            elif row['name'] == "annotation":

                assert head.endswith(':')
                head = head[:-1]
                contraction = before_contraction - len(head) - len(tail)

                for i in range(len(entry["ents"])):
                    tline, start, end, ann = entry["ents"][i]
                    if tline == line:
                        entry["ents"][i] = (tline, start - contraction, end - contraction, ann)
                entry["ents"].append((line, int(row.var_col_offset), len(head), annotation))
            else:
                raise Exception("wtf")

            new_line = head + tail
            body_lines[line] = new_line

        entry['text'] = "\n".join(body_lines)

        cum_lens = get_cum_lens(entry['text'])

        entry["ents"] = [(cum_lens[line] + start, cum_lens[line] + end, annotation) for
                         ind, (line, start, end, annotation) in enumerate(entry["ents"])]

        entry['original'] = body

        if isvalid(entry['text'], entry["ents"]):
            return entry
        else:
            return None
    else:
        return None


def main(args):
    bodies_path = args[1]
    bodies = pd.read_csv(bodies_path)

    body_field = bodies.columns[1]

    data = []

    for ind, body in enumerate(bodies[body_field]):
        # b = """def cosine(w: float, A: float = 1, phi: float = 0, offset: float = 0) -> \"partial[Callable[[], None]]\":\n    ''' Return a driver function that can advance a sequence of cosine values.\n\n    .. code-block:: none\n\n        value = A * cos(w*i + phi) + offset\n\n    Args:\n        w (float) : a frequency for the cosine driver\n        A (float) : an amplitude for the cosine driver\n        phi (float) : a phase offset to start the cosine driver with\n        offset (float) : a global offset to add to the driver values\n\n    '''\n    from math import cos\n    def f(i: float) -> float:\n        return A * cos(w*i + phi) + offset\n    return partial(force, sequence=_advance(f))"""
        # entry = process_body(b)
        entry = process_body(body)
        if entry is not None:
            data.append(entry)

    format = "jsonl"
    if format == "jsonl":
        with open(args[2], "a") as sink:
            for entry in data:
                sink.write(f"{json.dumps(entry)}\n")
    elif format == "csv":
        if os.path.isfile(args[2]):
            header = False
        else:
            header = True
        pd.DataFrame(data).to_csv(args[2], index=False, header=header)



if __name__ == "__main__":
    main(sys.argv)