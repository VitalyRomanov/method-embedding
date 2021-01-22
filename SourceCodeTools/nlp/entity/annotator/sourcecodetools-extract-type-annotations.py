import ast
import json
import os
import sys
from ast import literal_eval

import pandas as pd
import spacy
from spacy.gold import biluo_tags_from_offsets

from SourceCodeTools.nlp.entity.util import inject_tokenizer
from SourceCodeTools.nlp.entity.annotator.annotator_utils import to_offsets, overlap

nlp = inject_tokenizer(spacy.blank("en"))

allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
           'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}


def preprocess(ent):
    return ent
    # return ent.strip("\"").split("[")[0].split(".")[-1]


def inspect_fdef(node):
    if node.returns is not None:
        return [{"name": "returns", "line": node.returns.lineno - 1, "end_line": node.returns.end_lineno - 1,
                 "col_offset": node.returns.col_offset, "end_col_offset": node.returns.end_col_offset}]
    else:
        return []


def inspect_arg(node):
    return inspect_ann(node)


def isint(val):
    try:
        int(val)
        return True
    except:
        return False

def inspect_ann(node):
    if node.annotation is not None:
        return [{"name": "annotation", "line": node.annotation.lineno - 1, "end_line": node.annotation.end_lineno - 1,
                 "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset,
                 "var_line": node.lineno - 1, "var_end_line": node.end_lineno - 1, "var_col_offset": node.col_offset,
                 "var_end_col_offset": node.end_col_offset}]
    else:
        return []


def resolve_collisions(entities_1, entities_2):
    evict = set()

    for ind_1, e1 in enumerate(entities_1):
        for ind_2, e2 in enumerate(entities_2):
            if overlap(e1, e2):
                evict.add(ind_2)

    return [r for ind, r in enumerate(entities_2) if ind not in evict]


def correct_entities(entities, removed_offsets):
    offsets_sorted = sorted(removed_offsets, key=lambda x: x[0], reverse=True)

    offset_lens = [offset[1] - offset[0] for offset in offsets_sorted]

    for_correction = entities

    for offset_len, offset in zip(offset_lens, offsets_sorted):
        new_entities = []
        for entity in for_correction:
            if offset[0] <= entity[0] and offset[1] <= entity[0]:
                if len(entity) == 2:
                    new_entities.append((entity[0] - offset_len, entity[1] - offset_len))
                elif len(entity) == 3:
                    new_entities.append((entity[0] - offset_len, entity[1] - offset_len, entity[2]))
                else:
                    raise Exception("Invalid entity size")
            elif offset[0] >= entity[1] and offset[1] >= entity[1]:
                new_entities.append(entity)
            elif offset[0] <= entity[1] <= offset[1] or offset[0] <= entity[0] <= offset[1]:
                pass  # likely to be a type annotation being removed
            else:
                raise Exception("Invalid data?")

        for_correction = new_entities

    return for_correction


def get_docstring(body):

    body_lines = body.split("\n")

    docstring_ranges = []

    for node in ast.walk(ast.parse(body)):
        try:
            docstring = ast.get_docstring(node)
        except:
            continue
        else:
            if docstring is not None:
                docstring_ranges.append((node.body[0].lineno - 1, node.body[0].end_lineno - 1, #first line, last line
                                         0, len(body_lines[node.body[0].end_lineno - 1]), "docstring")) # beginning of first line, end of last line

    # do not need to use as_bytes here because column offsets are created with len(), not with ast package
    return to_offsets(body, docstring_ranges)

    # block_symbol = ""
    # block = False
    #
    # docstrings = []
    # cdocstring = None
    #
    # for line_no, line in enumerate(body_lines):
    #     if not block and (line.lstrip().startswith('"""') or line.lstrip().startswith("'''")):
    #         block = True
    #         block_symbol = line.lstrip()[:3]
    #         cdocstring = (line_no, 0)  # line and col_offset
    #
    #     if block and line.rstrip().endswith(block_symbol):
    #         if line_no == cdocstring[0] and len(line.strip()) >= 6 or \
    #                 line_no != cdocstring[0]:
    #             block = False
    #             block_symbol = ""
    #             docstrings.append((cdocstring[0], line_no, cdocstring[1], len(line), "docstring"))
    #
    # return to_offsets(body, docstrings)


def remove_offsets(body, entities, offsets):
    cuts = []

    new_body = body

    offsets_sorted = sorted(offsets, key=lambda x: x[0], reverse=True)

    for offset in offsets_sorted:
        cuts.append(new_body[offset[0]: offset[1]])
        new_body = new_body[:offset[0]] + new_body[offset[1]:]

    new_entities = correct_entities(entities, removed_offsets=offsets_sorted)

    return new_body, new_entities, cuts


def unpack_returns(body, labels):
    returns = []

    for ind, row in labels.iterrows():
        if row['name'] == "returns":
            returns.append((row['line'], row['end_line'], row['col_offset'], row['end_col_offset'], "returns"))

    # most likely do not need to use as_bytes here, because non-unicode usually appear in strings
    # but type annotations usually appear in the end of signature and in the beginnig of a line
    return_offsets = to_offsets(body, returns, as_bytes=True)

    cuts = []
    ret = []

    for offset in return_offsets:
        beginning = offset[0]
        end = offset[1]

        head = body[:offset[0]]
        orig_len = len(head)
        head = head.rstrip()
        head = head.rstrip("\\")
        head = head.rstrip()
        stripped_len = len(head)

        fannsymbol = "->"
        assert head.endswith(fannsymbol)
        beginning = beginning - (orig_len - stripped_len) - len(fannsymbol)
        cuts.append((beginning, end))
        ret.append(preprocess(body[offset[0]: offset[1]]))

    return ret, cuts


def unpack_annotations(body, labels):
    variables = []
    annotations = []

    for ind, row in labels.iterrows():
        if row['name'] == "annotation":
            variables.append((
                row['var_line'], row['var_end_line'], row['var_col_offset'], row['var_end_col_offset'],
                'variable'))
            annotations.append(
                (row['line'], row['end_line'], row['col_offset'], row['end_col_offset'], 'annotation '))

    # most likely do not need to use as_bytes here, because non-unicode usually appear in strings
    # but type annotations usually appear in the end of signature and in the beginnig of a line
    variables = to_offsets(body, variables, as_bytes=True)
    annotations = to_offsets(body, annotations, as_bytes=True)

    cuts = []
    vars = []

    for offset_ann, offset_var in zip(annotations, variables):
        beginning = offset_ann[0]
        end = offset_ann[1]

        head = body[:offset_ann[0]]
        orig_len = len(head)
        head = head.rstrip()
        stripped_len = len(head)

        annsymbol = ":"
        assert head.endswith(annsymbol)
        beginning = beginning - (orig_len - stripped_len) - len(annsymbol)
        cuts.append((beginning, end))

        assert offset_var[0] != len(head)
        vars.append((offset_var[0], beginning, preprocess(body[offset_ann[0]: offset_ann[1]])))

    return vars, cuts


def process_body(body, replacements):
    # do not need these two lines anymore
    # replacements = [(r[0], r[0], r[1], r[2], r[3]) for r in replacements]
    # replacements = to_offsets(body, replacements)

    entry = {"ents": [],
             "cats": [],
             "replacements": [],
             "text": None,
             "docstrings": []}

    body_ = body.lstrip()
    initial_strip = body[:len(body) - len(body_)]

    replacements = correct_entities(replacements, [(0, len(initial_strip))])

    docsting_offsets = get_docstring(body_)

    body_, replacements, docstrings = remove_offsets(body_, replacements, docsting_offsets)
    entry['docstrings'].extend(docstrings)

    initial_labels = get_initial_labels(body_)

    if initial_labels is None:
        return None

    returns, return_cuts = unpack_returns(body_, initial_labels)
    annotations, annotation_cuts = unpack_annotations(body_, initial_labels)

    body_, replacements_annotations, _ = remove_offsets(body_, replacements + annotations,
                                                        return_cuts + annotation_cuts)

    entry['replacements'].extend(list(filter(lambda x: isint(x[2]), replacements_annotations)))
    entry['ents'].extend(list(filter(lambda x: not isint(x[2]), replacements_annotations)))
    entry['cats'].extend(returns)
    entry['text'] = body_

    assert isvalid(body_, entry['replacements'])
    assert isvalid(body_, entry['ents'])

    return entry


def get_initial_labels(body_):
    try:
        root = ast.parse(body_)
    except SyntaxError as e:
        # print(e)
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
        df = pd.DataFrame(
            positions)
        df = df.astype({col: "Int32" for col in df.columns if col != "name"})
        df.sort_values(by=['line', 'end_col_offset'], ascending=[True, False], inplace=True)

        return df
    else:
        return None


def isvalid(text, ents):
    doc = nlp(text)
    tags = biluo_tags_from_offsets(doc, ents)
    # for t, tag in zip(doc, tags):
    #     print(tag, t.text, sep="\t")
    if "-" in tags:
        return False
    else:
        return True


# def overlap(p1, p2):
#     if (p2[1] - p1[0]) * (p2[0] - p1[1]) <= 0:
#         return True
#     else:
#         return False


def to_global_ids(entry, id_map, local_names, global_names):
    replacements = []
    for r in entry['replacements']:
        # id_ = int(r[2].split("_")[-1])
        id_ = r[2]
        assert local_names[id_] == global_names[id_map[id_]], f"{local_names[id_]} != {global_names[id_map[id_]]}"
        # assert local_names[id_][0].lower() == local_names[id_][0], f"{local_names[id_]}"
        replacements.append((r[0], r[1], str(id_map[id_])))

    entry['replacements'] = replacements
    return entry


def main(args):
    bodies_path = args[1]
    bodies = pd.read_csv(bodies_path)
    id2global = pd.read_csv(args[3])
    id_maps = dict(zip(id2global['id'], id2global['global_id']))

    global_names = pd.read_csv(args[4])
    local_names = pd.read_csv(args[5])
    global_names = dict(zip(global_names['id'].tolist(), global_names['serialized_name'].tolist()))
    local_names = dict(zip(local_names['id'].tolist(), local_names['serialized_name'].tolist()))

    # body_field = bodies.columns[1]

    data = []

    for ind, (_, row) in enumerate(bodies.iterrows()):
        body = row['body']
        replacements = literal_eval(row['replacement_list'])
        entry = process_body(body, replacements)
        if entry is not None:
            entry = to_global_ids(entry, id_maps, local_names, global_names)
            data.append(entry)
        print(f"{ind}/{len(bodies)}", end="\r")

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
