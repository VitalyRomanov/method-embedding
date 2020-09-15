import pandas as pd
import ast  # , astor
import sys, os
import json
import spacy
import re
from SourceCodeTools.proc.entity.util import inject_tokenizer
from ast import literal_eval

from spacy.gold import biluo_tags_from_offsets

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
    # if node.annotation is not None:
    #     return [{"name": "annotation", "line": node.annotation.lineno - 1, "end_line": node.annotation.end_lineno - 1,
    #              "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset,
    #              "var_line": node.lineno - 1, "var_end_line": node.end_lineno - 1, "var_col_offset": node.col_offset,
    #              "var_end_col_offset": node.end_col_offset}]
    # else:
    #     return []


def inspect_ann(node):
    if node.annotation is not None:
        return [{"name": "annotation", "line": node.annotation.lineno - 1, "end_line": node.annotation.end_lineno - 1,
                 "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset,
                 "var_line": node.lineno - 1, "var_end_line": node.end_lineno - 1, "var_col_offset": node.col_offset,
                 "var_end_col_offset": node.end_col_offset}]
    else:
        return []


def get_cum_lens(body):
    """
    Calculate the cummulative lengths of each line with respect to the beginning of
    the function's body.
    """
    body_lines = body.split("\n")
    cum_lens = [0]
    for ind, line in enumerate(body_lines):
        cum_lens.append(len(line) + cum_lens[-1] + 1)
    return cum_lens


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
                new_entities.append((entity))
            elif offset[0] <= entity[1] <= offset[1] or offset[0] <= entity[0] <= offset[1]:
                pass  # likely to be a type annotation being removed
            else:
                raise Exception("Invalid data?")

        for_correction = new_entities

    return for_correction


def to_offsets(body, entities):
    """
    Transform entity annotation format from (line, end_line, col, end_col)
    to (char_ind, end_char_ind).
    """
    cum_lens = get_cum_lens(body)

    repl = [(cum_lens[line] + start, cum_lens[end_line] + end, annotation) for
            ind, (line, end_line, start, end, annotation) in enumerate(entities)]

    return repl


def get_docstring(body):
    body_lines = body.split("\n")

    block_symbol = ""
    block = False

    docstrings = []
    cdocstring = None

    for line_no, line in enumerate(body_lines):
        if not block and (line.lstrip().startswith('"""') or line.lstrip().startswith("'''")):
            block = True
            block_symbol = line.lstrip()[:3]
            cdocstring = (line_no, 0)  # line and col_offset

        if block and line.rstrip().endswith(block_symbol):
            if line_no == cdocstring[0] and len(line.strip()) >= 6 or \
                    line_no != cdocstring[0]:
                block = False
                block_symbol = ""
                docstrings.append((cdocstring[0], line_no, cdocstring[1], len(line), "docstring"))

    return to_offsets(body, docstrings)


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

    return_offsets = to_offsets(body, returns)

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

    variables = to_offsets(body, variables)
    annotations = to_offsets(body, annotations)

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

    replacements = [(r[0], r[0], r[1], r[2], r[3]) for r in replacements]
    replacements = to_offsets(body, replacements)

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

    if initial_labels is None: return None

    returns, return_cuts = unpack_returns(body_, initial_labels)
    annotations, annotation_cuts = unpack_annotations(body_, initial_labels)

    # if body_.startswith("def __init__(\n        self,\n        expr,\n        engine: str = \"numexpr\",\n        parser: str = \"pandas\",\n        env: Optional[Scope] = None,\n        level: int = 0,\n    )"):
    #     print()

    body_, replacements_annotations, _ = remove_offsets(body_, replacements + annotations,
                                                             return_cuts + annotation_cuts)

    entry['replacements'].extend(replacements_annotations[:-len(annotations)])
    entry['ents'].extend(replacements_annotations[-len(annotations):])
    entry['cats'].extend(returns)
    entry['text'] = body_

    assert isvalid(body_, entry['replacements'])
    assert isvalid(body_, entry['ents'])

    return entry


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
        df = pd.DataFrame(
            positions)
        df = df.astype({col: "Int32" for col in df.columns if col != "name"})
        df.sort_values(by=['line', 'end_col_offset'], ascending=[True, False], inplace=True)

        return df
    else:
        return None


# def strip_docstring(body):  # remove first docstring (docstring of the main function (not nested functions)
#     # # TODO
#     # # this does not seem to be working well
#     # root = ast.parse(body)
#     # main_doc = None
#     # ranges = []
#     # for node in ast.walk(root):
#     #     try:
#     #         docstring = ast.get_docstring(node)
#     #     except:
#     #         continue
#     #     else:
#     #         if docstring is not None:
#     #             ranges.append((node.body[0].lineno-1, node.body[0].end_lineno))
#     #             # if main_doc is None:
#     #             #     main_doc = docstring
#     #             # node.body = node.body[1:]
#     #
#     # ranges = sorted(ranges, key=lambda x: x[0], reverse=True)
#
#     body_lines = body.split("\n")
#
#     new_body = []
#     new_doc = []
#
#     block = False
#     no_removals = False
#
#     docstring_starts = 0
#     doc_len_lines = 0
#
#     block_symbol = ""
#
#     for ind, line in enumerate(body_lines):
#         if not block and (line.lstrip().startswith('"""') or line.lstrip().startswith("'''")):
#             block = True
#             block_symbol = line.lstrip()[:3]
#             # continue
#
#         if block is False or no_removals is True:
#             new_body.append(line)
#         else:
#             new_doc.append(line)
#             docstring_starts = ind
#             # doc_len_lines = 1
#
#         if block and line.rstrip().endswith(block_symbol) or line.rstrip().endswith(block_symbol):
#             if len(new_doc) == 1 and len(line.strip()) >= 6 or \
#                     len(new_doc) > 1:
#                 block = False
#                 no_removals = True
#                 block_symbol = ""
#                 # continue
#
#     doc_len_lines += len(new_doc)
#
#     new_body = "\n".join(new_body)
#     new_doc = "\n".join(new_doc)
#
#     return new_body, new_doc, docstring_starts, doc_len_lines
#     # return astor.to_source(root), main_doc


def isvalid(text, ents):
    doc = nlp(text)
    tags = biluo_tags_from_offsets(doc, ents)
    for t, tag in zip(doc, tags):
        print(tag, t.text, sep="\t")
    if "-" in tags:
        return False
    else:
        return True


def overlap(p1, p2):
    if (p2[1] - p1[0]) * (p2[0] - p1[1]) <= 0:
        return True
    else:
        return False


# def label_replacements_overap(labels, replacements):
#     evict = set()
#     for ind, row in labels.iterrows():
#         for ind, r in enumerate(replacements):
#             if r[0] == line - 1:
#                 if overlap((row.col_offset, row.end_col_offset), (r[1], r[2])):
#                     evict.add(ind)
#
#     return [r for ind, r in enumerate(replacements) if ind not in evict]


# def adjust_replacements(replacements, doc_start, doc_len):
#     repl = []
#
#     for r in replacements:
#         line = r[0]
#         if line > doc_start:
#             line -= doc_len
#
#         repl.append((line, r[1], r[2], r[3]))
#
#     return repl


# def account_for_strip(replacements, first_line_strip):
#     repl = []
#     for r in replacements:
#         line = r[0]
#         col_s = r[1]
#         col_e = r[2]
#
#         if line == 0:
#             col_s -= first_line_strip
#             col_e -= first_line_strip
#
#         repl.append((line, col_s, col_e, r[3]))
#
#     return repl


# def prepare_replacements(replacements):
#     return pd.DataFrame(
#         [{"line": r[0], "col_offset": r[1], "end_col_offset": r[2], "ann": r[3]} for r in replacements]).sort_values(
#         by=['line', 'end_col_offset'], ascending=[True, False])


# def adjust_contraction(entry, line, head, contraction):
#     for i in range(len(entry["ents"])):
#         tline, start, end, ann = entry["ents"][i]
#         if tline == line:
#             entry["ents"][i] = (tline, start - contraction, end - contraction, ann)
#
#     for i in range(len(entry['replacements'])):
#         tline, start, end, ann = entry["replacements"][i]
#         if tline == line:
#             if start >= len(head):
#                 entry["replacements"][i] = (tline, start - contraction, end - contraction, ann)
#
#     return entry


# def process_body(body, replacements, remove_docstring=True):
#     body_ = body.strip()
#
#     replacements = account_for_strip(replacements, first_line_strip=len(body) - len(body.lstrip()))
#
#     entry = {"ents": [], "cats": [], "replacements": []}
#
#     if remove_docstring:
#         try:
#             only_body, doc, doc_start, doc_len = strip_docstring(body_)
#             o_r = replacements
#             replacements = adjust_replacements(replacements, doc_start, doc_len)
#         except UnicodeDecodeError:
#             return None
#         except SyntaxError:
#             return None
#
#         entry['text'] = only_body
#         entry['docstring'] = doc
#
#         body_ = only_body
#
#     initial_labels = get_initial_labels(body_)
#
#     # wrap everything in dataframe and sort
#     # replacements = prepare_replacements(replacements)
#
#     if initial_labels is not None:
#
#         body_lines = body_.split("\n")
#
#         replacements = label_replacements_overap(initial_labels, replacements)
#
#         entry['replacements'].extend(replacements)
#
#         # initial labels are sorted
#         #
#         for ind, row in initial_labels.iterrows():
#             line = row.line - 1
#
#             # TODO
#             # multiline annotations are not parsed correctly
#
#             annotation = body_lines[line][row.col_offset: row.end_col_offset]
#             annotation = preprocess(annotation)
#             tail = body_lines[line][row.end_col_offset:]
#             head = body_lines[line][:row.col_offset].rstrip()
#             before_contraction = len(body_lines[line])
#
#             if row['name'] == "returns":
#                 try:
#                     assert head.endswith("->")
#                 except AssertionError:
#                     return None
#                 head = head[:-2]
#                 if line == 0:  # only use labels for the main and not nested functions
#                     entry["cats"].append({"returns": annotation})
#
#                 contraction = before_contraction - len(head) - len(tail)
#
#                 entry = adjust_contraction(entry, line, head, contraction)
#
#                 # for i in range(len(entry["ents"])):
#                 #     tline, start, end, ann = entry["ents"][i]
#                 #     if tline == line:
#                 #         entry["ents"][i] = (tline, start - contraction, end - contraction, ann)
#                 #
#                 # for i in range(len(entry['replacements'])):
#                 #     tline, start, end, ann = entry["replacements"][i]
#                 #     if tline == line:
#                 #         if start >= len(head):
#                 #             entry["replacements"][i] = (tline, start - contraction, end - contraction, ann)
#
#             elif row['name'] == "annotation":
#
#                 try:
#                     assert head.endswith(':')
#                 except AssertionError:
#                     return None
#                 head = head[:-1]
#                 contraction = before_contraction - len(head) - len(tail)
#
#                 entry = adjust_contraction(entry, line, head, contraction)
#
#                 # for i in range(len(entry["ents"])):
#                 #     tline, start, end, ann = entry["ents"][i]
#                 #     if tline == line:
#                 #         entry["ents"][i] = (tline, start - contraction, end - contraction, ann)
#                 #
#                 # for i in range(len(entry['replacements'])):
#                 #     tline, start, end, ann = entry["replacements"][i]
#                 #     if tline == line:
#                 #         if start >= len(head):
#                 #             entry["replacements"][i] = (tline, start - contraction, end - contraction, ann)
#
#                 assert int(row.var_col_offset) != len(head)
#
#                 entry["ents"].append((line, int(row.var_col_offset), len(head), annotation))
#             else:
#                 raise Exception("wtf")
#
#             new_line = head + tail
#             body_lines[line] = new_line
#
#         entry['text'] = "\n".join(body_lines)
#
#         cum_lens = get_cum_lens(entry['text'])
#
#         ents_before = [body_lines[line][start: end] for
#                        ind, (line, start, end, annotation) in enumerate(entry["ents"])]
#
#         entry["ents"] = [(cum_lens[line] + start, cum_lens[line] + end, annotation) for
#                          ind, (line, start, end, annotation) in enumerate(entry["ents"])]
#
#         entry["replacements"] = [(cum_lens[line] + start, cum_lens[line] + end, annotation) for
#                                  ind, (line, start, end, annotation) in enumerate(entry["replacements"])]
#
#         ents_after = [entry['text'][start: end] for
#                       ind, (start, end, annotation) in enumerate(entry["ents"])]
#
#         assert ents_before == ents_after
#
#         entry['original'] = body
#
#         # entry["ents"] = list(map(lambda x: x if x[2] in allowed else (x[0], x[1], "Other"), entry["ents"]))
#         # entry["cats"] = list(filter(lambda x: x["returns"] if x["returns"] in allowed else "Other", entry['cats']))
#
#         if not entry["ents"]:  # and not entry["cats"]:
#             return None  # in case all entities were filtered
#
#         # assert isvalid(entry['text'], entry["ents"])
#         assert isvalid(entry['text'], entry["replacements"])
#         return entry
#
#         # if isvalid(entry['text'], entry["ents"]):
#         #     return entry
#         # else:
#         #     return None
#     else:
#         return None


def to_global_ids(entry, id_map, local_names, global_names):
    replacements = []
    for r in entry['replacements']:
        id_ = int(r[2].split("_")[-1])
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
