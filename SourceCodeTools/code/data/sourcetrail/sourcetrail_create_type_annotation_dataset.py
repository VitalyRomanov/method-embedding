import ast
import json
import logging
import os

import pandas as pd
from tqdm import tqdm

from SourceCodeTools.code.data.sourcetrail.file_utils import unpersist, unpersist_if_present
from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.entity.annotator.annotator_utils import to_offsets, adjust_offsets2, \
    resolve_self_collisions2
from SourceCodeTools.nlp.spacy_tools import isvalid

# allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
#            'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}


def preprocess(ent):
    return ent
    # return ent.strip("\"").split("[")[0].split(".")[-1]


def inspect_fdef(node):
    """
    Extract return type annotations
    :param node: ast node
    :return:
    """
    if node.returns is not None:
        return [{"name": "returns", "line": node.returns.lineno - 1, "end_line": node.returns.end_lineno - 1,
                 "col_offset": node.returns.col_offset, "end_col_offset": node.returns.end_col_offset}]
    else:
        return []


def inspect_arg(node):
    """
    Extract variable type annotation
    :param node: ast node
    :return:
    """
    return inspect_ann(node)


def isint(val):
    try:
        int(val)
        return True
    except ValueError:
        return False


def inspect_ann(node):
    """
    Extract variable type annotation
    :param node: ast node
    :return:
    """
    if node.annotation is not None:
        return [{"name": "annotation", "line": node.annotation.lineno - 1, "end_line": node.annotation.end_lineno - 1,
                 "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset,
                 "var_line": node.lineno - 1, "var_end_line": node.end_lineno - 1, "var_col_offset": node.col_offset,
                 "var_end_col_offset": node.end_col_offset}]
    else:
        return []


def correct_entities(entities, removed_offsets):
    """
    Update offsets based on the information about removed spans
    :param entities: list of entities in format (start, end, entity)
    :param removed_offsets: list of removed spans in format (start, end)
    :return:
    """
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


def get_docstring(body: str):
    """
    Get docstring ranges
    :param body:
    :return:
    """
    body_lines = body.split("\n")

    docstring_ranges = []

    for node in ast.walk(ast.parse(body)):
        try:
            docstring = ast.get_docstring(node)
        except:  # syntax error?
            continue
        else:
            if docstring is not None:
                docstring_ranges.append(
                    (
                        node.body[0].lineno - 1, node.body[0].end_lineno - 1,  # first line, last line
                        0, len(body_lines[node.body[0].end_lineno - 1]),  # beginning of first line, end of last line
                        "docstring"
                    )
                )

    # as bytes is not needed because the offsets are created using len and not ast package
    return to_offsets(body, docstring_ranges, as_bytes=False)


def remove_offsets(body: str, entities, offsets):
    """
    Remove offsets from body, adjust entities to match trimmed body
    :param body:
    :param entities: list of entities in format (start, end, entity)
    :param offsets: list of removed spans in format (start, end)
    :return:
    """
    cuts = []

    new_body = body

    offsets_sorted = sorted(offsets, key=lambda x: x[0], reverse=True)

    for offset in offsets_sorted:
        cuts.append(new_body[offset[0]: offset[1]])
        new_body = new_body[:offset[0]] + new_body[offset[1]:]

    new_entities = correct_entities(entities, removed_offsets=offsets_sorted)

    return new_body, new_entities, cuts


def unpack_returns(body: str, labels: pd.DataFrame):
    """
    Use information from ast package to strip return type annotation from function body
    :param body:
    :param labels: DataFrame with information about return type annotation
    :return: Trimmed body and list of return types (normally one).
    """
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
    """
    Use information from ast package to strip type annotation from function body
    :param body:
    :param labels: DataFrame with information about type annotations
    :return: Trimmed body and list of annotations.
    """
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


def process_body(nlp, body: str, replacements=None):
    """
    Extract annotation information, strip documentation and type annotations.
    :param nlp: Spacy tokenizer
    :param body: Function body
    :param replacements: Optional. Additional replacements that need to be adjusted for modified function
    :return: Entry with modified function body. Returns None if not annotations in the function
    """

    if replacements is None:
        replacements = []

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

    replacements_annotations = adjust_offsets2(replacements_annotations, len(initial_strip))
    body_ = initial_strip + body_

    entry['replacements'].extend(list(filter(lambda x: isint(x[2]), replacements_annotations)))
    entry['ents'].extend(list(filter(lambda x: not isint(x[2]), replacements_annotations)))
    entry['cats'].extend(returns)
    entry['text'] = body_

    entry['replacements'] = resolve_self_collisions2(entry['replacements'])

    assert isvalid(nlp, body_, entry['replacements'])
    assert isvalid(nlp, body_, entry['ents'])

    return entry


def get_initial_labels(body_):
    """
    Walk ast to find type annotations
    :param body_:
    :return: DataFrame with type annotation information
    """
    try:
        root = ast.parse(body_)
    except SyntaxError:
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


def to_global_ids(entry, id_map, global_names=None, local_names=None):
    """
    Map local ids to ids in a global source code graph
    :param entry:
    :param id_map: mapping from local ids to global ids
    :param global_names: Optional, used for verification
    :param local_names: Optional, used for verification
    :return: entry with global replacement ids
    """
    global_replacements = []
    for r in entry['replacements']:
        id_ = r[2]
        if global_names is not None and local_names is not None:
            assert local_names[id_] == global_names[id_map[id_]], f"{local_names[id_]} != {global_names[id_map[id_]]}"

        # cast id into string to make format compatible with spacy's NER classifier
        global_replacements.append((r[0], r[1], str(id_map[id_])))

    entry['replacements'] = global_replacements
    return entry


def offsets_for_func(offsets, body, func_id):
    def in_mention(id_, mentions):
        for mention in mentions:
            if mention[2] == id_:
                return True
        return False

    def get_correct_mention(mentions, id_):
        for mention in mentions:
            if mention[2] == id_:
                return mention
        raise Exception("Mention should have been found")

    in_mention_ = lambda mention: in_mention(func_id, mention)
    body_offsets = offsets.query("mentioned_in.map(@in_mention)", local_dict={"in_mention": in_mention_})

    if len(body_offsets) == 0:
        return []

    start_, end_, id_ = get_correct_mention(mentions=body_offsets.iloc[0, 4], id_=func_id)

    body_offsets = [tuple(offset) for offset in body_offsets[["start", "end", "node_id"]].values.tolist()]

    initial_indent = len(body) - len(body.lstrip())
    body_offsets = adjust_offsets2(body_offsets, amount=-(start_-initial_indent))
    body_offsets = list(set(body_offsets))

    return body_offsets


def load_names(nodes_path):
    if nodes_path is not None:
        nodes = unpersist(nodes_path)
        names = dict(zip(nodes['id'].tolist(), nodes['serialized_name'].tolist()))
    else:
        names = None
    return names


def process_package(working_directory, global_names=None):
    """
    Find functions with annotations, extract annotation information, strip documentation and type annotations.
    :param working_directory: location of package related files
    :param global_names: optional, mapping from global node ids to names
    :return: list of entries in spacy compatible format
    """
    bodies = unpersist_if_present(os.path.join(working_directory, "source_graph_bodies.bz2"))
    if bodies is None:
        return []

    offsets_path = os.path.join(working_directory, "offsets.bz2")

    # offsets store information about spans for nodes referenced in the source code
    if os.path.isfile(offsets_path):
        offsets = unpersist(offsets_path)
    else:
        logging.warning(f"No file with offsets: {offsets_path}")
        offsets = None

    def load_local2global(working_directory):
        local2global = unpersist(os.path.join(working_directory, "local2global_with_ast.bz2"))
        id_maps = dict(zip(local2global['id'], local2global['global_id']))
        return id_maps

    id_maps = load_local2global(working_directory)

    local_names = load_names(os.path.join(working_directory, "nodes_with_ast.bz2"))

    nlp = create_tokenizer("spacy")

    data = []

    for ind, (_, row) in tqdm(
            enumerate(bodies.iterrows()), total=len(bodies),
            leave=True, desc=os.path.basename(working_directory)
    ):
        body = row['body']

        if offsets is not None:
            graph_node_spans = offsets_for_func(offsets, body, row["id"])
        else:
            graph_node_spans = []

        entry = process_body(nlp, body, replacements=graph_node_spans)

        if entry is not None:
            entry = to_global_ids(entry, id_maps, global_names, local_names)
            data.append(entry)

    return data


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("packages", type=str, help="")
    parser.add_argument("output_dataset", type=str, help="")
    parser.add_argument("--format", "-f", dest="format", default="jsonl", help="jsonl|csv")
    parser.add_argument("--global_nodes", "-g", dest="global_nodes", default=None)

    args = parser.parse_args()

    global_names = load_names(args.global_nodes)

    data = []

    for package in os.listdir(args.packages):
        pkg_path = os.path.join(args.packages, package)
        if not os.path.isdir(pkg_path):
            continue

        data.extend(process_package(working_directory=pkg_path, global_names=global_names))

    if args.format == "jsonl":  # jsonl format is used by spacy
        with open(args.output_dataset, "w") as sink:
            for entry in data:
                sink.write(f"{json.dumps(entry)}\n")
    elif args.format == "csv":
        if os.path.isfile(args.output_dataset):
            header = False
        else:
            header = True
        pd.DataFrame(data).to_csv(args.output_dataset, index=False, header=header)


if __name__ == "__main__":
    main()
