import ast
import json
import logging
import os
from bisect import bisect_left
from os.path import join

import pandas as pd

from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.code.annotator_utils import to_offsets, adjust_offsets2, \
    resolve_self_collisions2
from SourceCodeTools.nlp.spacy_tools import isvalid


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
        return [{
            "name": "returns",
            "line": node.returns.lineno - 1,
            "end_line": node.returns.end_lineno - 1,
            "col_offset": node.returns.col_offset,
            "end_col_offset": node.returns.end_col_offset,
            "var_line": node.lineno - 1,
            "var_end_line": node.end_lineno - 1,
            "var_col_offset": node.col_offset,
            "var_end_col_offset": node.end_col_offset
        }]
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
            if offset[0] <= entity[0] and offset[1] <= entity[0]:  # removed span is to the left of the entity
                if len(entity) == 2:
                    new_entities.append((entity[0] - offset_len, entity[1] - offset_len))
                elif len(entity) == 3:
                    new_entities.append((entity[0] - offset_len, entity[1] - offset_len, entity[2]))
                else:
                    raise Exception("Invalid entity size")
            elif offset[0] >= entity[1] and offset[1] >= entity[1]:
                # removed span is to the right of the entity
                new_entities.append(entity)
            elif offset[0] <= entity[0] <= offset[1] and offset[0] <= entity[1] <= offset[1]:
                # removed span covers the entity
                pass
            elif offset[0] <= entity[0] <= offset[1] and entity[0] <= offset[1] <= entity[1]:
                # removed span overlaps on the left
                if len(entity) == 3:
                    new_entities.append((entity[0] - offset_len + offset[1] - entity[1], entity[1] - offset_len, entity[2]))
                elif len(entity) == 2:
                    new_entities.append((entity[0] - offset_len + offset[1] - entity[1], entity[1] - offset_len))
                else:
                    raise Exception("Invalid entity size")
            elif entity[0] <= offset[0] <= entity[1] and entity[0] <= entity[1] <= offset[1]:
                # removed span overlaps on the right
                if len(entity) == 3:
                    new_entities.append((entity[0], entity[1] - offset_len + offset[1] - entity[1], entity[2]))
                elif len(entity) == 2:
                    new_entities.append((entity[0], entity[1] - offset_len + offset[1] - entity[1]))
                else:
                    raise Exception("Invalid entity size")
            elif entity[0] <= offset[0] <= entity[1] and entity[0] <= offset[1] <= entity[1]:
                # removed span is inside the entity
                if len(entity) == 3:
                    new_entities.append((entity[0], entity[1] - offset_len, entity[2]))
                elif len(entity) == 2:
                    new_entities.append((entity[0], entity[1] - offset_len))
                else:
                    raise Exception("Invalid entity size")
            else:
                logging.warning(f"Encountered invalid offset: {entity}")
                # raise Exception("Invalid data?")

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


def remove_offsets(body: str, entities, default_values, return_offsets, offsets_to_cut):
    """
    Remove offsets from body, adjust entities to match trimmed body
    :param body:
    :param entities: list of entities in format (start, end, entity)
    :param default_values:
    :param return_offsets:
    :param offsets_to_cut:
    :return:
    """
    cuts = []

    new_body = body

    offsets_sorted = sorted(offsets_to_cut, key=lambda x: x[0], reverse=True)

    for offset in offsets_sorted:
        cuts.append(new_body[offset[0]: offset[1]])
        new_body = new_body[:offset[0]] + new_body[offset[1]:]

    new_entities = correct_entities(entities, removed_offsets=offsets_sorted)
    default_values = correct_entities(default_values, removed_offsets=offsets_sorted)
    return_offsets = correct_entities(return_offsets, removed_offsets=offsets_sorted)

    return new_body, new_entities, default_values, return_offsets, cuts


def unpack_returns(body: str, labels: pd.DataFrame):
    """
    Use information from ast package to strip return type annotation from function body
    :param body:
    :param labels: DataFrame with information about return type annotation
    :return: Trimmed body and list of return types (normally one).
    """
    if labels is None:
        return [], []

    returns = []
    variables = []

    for ind, row in labels.iterrows():
        if row['name'] == "returns":
            variables.append((
                row['var_line'], row['var_end_line'], row['var_col_offset'], row['var_end_col_offset'],
                'function'))
            returns.append((row['line'], row['end_line'], row['col_offset'], row['end_col_offset'], "returns"))

    # most likely do not need to use as_bytes here, because non-unicode usually appear in strings
    # but type annotations usually appear in the end of signature and in the beginning of a line
    return_offsets = to_offsets(body, returns, as_bytes=True)
    function_offsets = to_offsets(body, variables, as_bytes=True)

    cuts = []
    ret = []

    for offset, fn_offset in zip(return_offsets, function_offsets):
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
        ret.append((fn_offset[0], fn_offset[1], preprocess(body[offset[0]: offset[1]])))

    return ret, cuts


def ast_node_to_offset(node, body):
    return to_offsets(
        body,
        [(node.lineno - 1, node.end_lineno - 1, node.col_offset, node.end_col_offset, "node")],
        as_bytes=True
    )[0]


def get_defaults_spans(body):
    root = ast.parse(body)
    defaults_offsets = []
    defaults = {}
    for node in ast.walk(root):
        if isinstance(node, ast.arguments):
            without_default = len(node.args) - len(node.defaults)
            for ind, i in enumerate(range(without_default, len(node.args))):
                arg = ast_node_to_offset(node.args[i], body)
                default_value = ast_node_to_offset(node.defaults[ind], body)
                defaults[arg] = default_value
                defaults_offsets.append(default_value)

            for i in range(len(node.kwonlyargs)):
                arg = ast_node_to_offset(node.kwonlyargs[i], body)
                if node.kw_defaults[i] is not None:
                    default_value = ast_node_to_offset(node.kw_defaults[i], body)
                    defaults[arg] = default_value
                    defaults_offsets.append(default_value)

    default_values = []
    for v_off, d_off in defaults.items():
        default_values.append(
            (v_off[0], v_off[1], body[d_off[0]: d_off[1]])
        )

    extended = []
    for start, end, label in defaults_offsets:
        while body[start] != "=":
            start -= 1
        extended.append((start, end))

    return extended, default_values


def unpack_annotations(body, labels, remove_default=False):
    """
    Use information from ast package to strip type annotation from function body
    :param body:
    :param labels: DataFrame with information about type annotations
    :param remove_default:
    :return: Trimmed body and list of annotations.
    """
    if labels is None:
        return [], [], []

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
    # but type annotations usually appear in the end of signature and in the beginning of a line
    variables = to_offsets(body, variables, as_bytes=True)
    annotations = to_offsets(body, annotations, as_bytes=True)
    defaults_spans, default_values = get_defaults_spans(body)

    cuts = []
    vars = []

    for offset_ann, offset_var in zip(annotations, variables):
        beginning = offset_ann[0]
        end = offset_ann[1]

        head = body[:offset_ann[0]]
        orig_len = len(head)
        head = head.rstrip(" \\\n\t")  # include character \ since it can be used to indicate line break
        stripped_len = len(head)

        annsymbol = ":"
        if head.endswith(annsymbol):
            assert head.endswith(annsymbol)
            beginning = beginning - (orig_len - stripped_len) - len(annsymbol)
            # Workaround if there is a space before annotation. Example: "groupby : List[str]"
            while beginning > 0 and body[beginning - 1] == " ":
                beginning -= 1
            cuts.append((beginning, end))

            assert offset_var[0] != len(head)
            vars.append((offset_var[0], beginning, preprocess(body[offset_ann[0]: offset_ann[1]])))
        else:
            logging.info(f"Failed to remove annotation")

    if remove_default:
        cuts.extend(defaults_spans)

    return vars, cuts, default_values


def body_valid(body):
    try:
        ast.parse(body)
        return True
    except SyntaxError:
        return False


def process_body(
        nlp, body: str, replacements=None, require_labels=False, remove_default=False, remove_docstring=False,
        keep_return_offsets=False
):
    """
    Extract annotation information, strip documentation and type annotations.
    :param nlp: Spacy tokenizer
    :param body: Function body
    :param replacements: Optional. Additional replacements that need to be adjusted for modified function
    :param require_labels:
    :param remove_default:
    :param remove_docstring:
    :param keep_return_offsets:
    :return: Entry with modified function body. Returns None if not annotations in the function
    """

    if replacements is None:
        replacements = []

    entry = {
        "ents": [],
        "cats": [],
        "replacements": [],
        "defaults": [],
        "text": None,
        "docstrings": []
    }

    body_ = body.lstrip()
    initial_strip = body[:len(body) - len(body_)]

    replacements = correct_entities(replacements, [(0, len(initial_strip))])

    if body_valid(body_) is False:
        return None

    docstring_offsets = get_docstring(body_)

    if remove_docstring:
        body_, replacements, _, _, docstrings = remove_offsets(
            body_,
            entities=replacements,
            default_values=[],
            return_offsets=[],
            offsets_to_cut=docstring_offsets
        )
        entry['docstrings'].extend(docstrings)

    was_valid = body_valid(body_)
    initial_labels = get_initial_labels(body_)

    if require_labels and initial_labels is None:
        return None

    returns, return_cuts = unpack_returns(body_, initial_labels)
    annotations, annotation_cuts, default_values = unpack_annotations(body_, initial_labels, remove_default=remove_default)

    body_, replacements_annotations, default_values, returns, _ = remove_offsets(
        body_,
        entities=replacements + annotations, default_values=default_values,
        return_offsets=returns,
        offsets_to_cut=return_cuts + annotation_cuts
    )

    if not keep_return_offsets:
        returns = [return_[2] for return_ in returns]

    is_valid = body_valid(body_)
    if was_valid != is_valid:
        print("Failed processing")
        return None
        # raise Exception()

    replacements_annotations = adjust_offsets2(replacements_annotations, len(initial_strip))
    body_ = initial_strip + body_

    entry['replacements'].extend(list(filter(lambda x: isint(x[2]), replacements_annotations)))
    entry['ents'].extend(list(filter(lambda x: not isint(x[2]), replacements_annotations)))
    entry['cats'].extend(returns)
    entry['text'] = body_
    entry["defaults"] = default_values

    entry['replacements'] = resolve_self_collisions2(entry['replacements'])

    # assert isvalid(nlp, body_, entry['replacements'])
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

        global_replacements.append((r[0], r[1], id_map[id_]))

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

    body_offsets = offsets.query(
        "mentioned_in.map(@in_mention)",
        local_dict={"in_mention": lambda mention: in_mention(func_id, mention)}
    )

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


def process_package(working_directory, global_names=None, require_labels=False, remove_default=False):
    """
    Find functions with annotations, extract annotation information, strip documentation and type annotations.
    :param working_directory: location of package related files
    :param global_names: optional, mapping from global node ids to names
    :param remove_default:
    :param require_labels:
    :return: list of entries in spacy compatible format
    """
    # bodies = unpersist_if_present(os.path.join(working_directory, "source_graph_bodies.bz2"))
    # if bodies is None:
    #     return []

    # offsets_path = os.path.join(working_directory, "offsets.bz2")

    # # offsets store information about spans for nodes referenced in the source code
    # if os.path.isfile(offsets_path):
    #     offsets = unpersist(offsets_path)
    # else:
    #     logging.warning(f"No file with offsets: {offsets_path}")
    #     offsets = None

    if not os.path.isfile(join(working_directory, "has_annotations")):
        return []

    def load_local2global(working_directory):
        local2global = unpersist(os.path.join(working_directory, "local2global_with_ast.bz2"))
        id_maps = dict(zip(local2global['id'], local2global['global_id']))
        return id_maps

    id_maps = load_local2global(working_directory)

    local_names = load_names(os.path.join(working_directory, "nodes_with_ast.bz2"))

    node_maps = get_node_maps(unpersist(join(working_directory, "nodes_with_ast.bz2")))
    filecontent = get_filecontent_maps(unpersist(join(working_directory, "filecontent_with_package.bz2")))
    offsets = group_offsets(unpersist(join(working_directory, "offsets.bz2")))

    data = []
    nlp = create_tokenizer("spacy")

    for ind, function_data in enumerate(iterate_functions(offsets, node_maps, filecontent)):
        try:
            entry = process_body(
                nlp, function_data.pop("body"), replacements=function_data.pop("offsets"),
                require_labels=require_labels, remove_default=remove_default, remove_docstring=True
            )
        except Exception as e:
            logging.warning("Error during processing")
            print(working_directory)
            print(e)
            continue

        if entry is not None:
            entry = to_global_ids(entry, id_maps, global_names, local_names)
            entry.update(function_data)
            data.append(entry)

    return data


def iterate_functions(offsets, nodes, filecontent):

    allowed_entity_types = {"class_method", "function"}

    for package_id in offsets:
        content = filecontent[package_id]

        lines = []
        for ind, line in enumerate(content.split("\n")):
            length = len(line)
            if ind != 0:
                length += lines[ind - 1] + 1
            lines.append(length)

        # entry is a function or a class
        for (entity_start, entity_end, entity_node_id), entity_offsets in offsets[package_id].items():
            if nodes[entity_node_id][1] in allowed_entity_types:
                body = content[entity_start: entity_end]
                adjusted_entity_offsets = adjust_offsets2(entity_offsets, -entity_start)

                yield {
                    "body": body,
                    "offsets": adjusted_entity_offsets,
                    "package": package_id[0],
                    "file_id": package_id[1],
                    "line_start_python": bisect_left(lines, entity_start),
                    "line_end_python": bisect_left(lines, entity_end-1) + 1
                }


def get_node_maps(nodes):
    return dict(zip(nodes["id"], zip(nodes["serialized_name"], nodes["type"])))


def get_filecontent_maps(filecontent):
    return dict(zip(zip(filecontent["package"], filecontent["id"]), filecontent["content"]))


def group_offsets(offsets):
    """
    :param offsets: Dataframe with offsets
    :return: offsets grouped first by package name and file id, and then by the entity in which they occur.
    """
    # This function will process all function that have graph annotations. If there are no
    # annotations - the function is not processed.
    offsets_grouped = {}

    for file_id, start, end, node_id, mentioned_in, package in offsets.values:
        package_id = (package, file_id)
        if package_id not in offsets_grouped:
            offsets_grouped[package_id] = {}

        offset_ent = (start, end, node_id)

        for e in mentioned_in:
            e = tuple(e)
            if e not in offsets_grouped[package_id]:
                offsets_grouped[package_id][e] = []

            offsets_grouped[package_id][e].append(offset_ent)

    return offsets_grouped


def create_from_dataset(**kwargs):
    remove_default = kwargs["remove_default"]

    node_maps = get_node_maps(unpersist(join(kwargs["dataset_path"], "common_nodes.json")))
    filecontent = get_filecontent_maps(unpersist(join(kwargs["dataset_path"], "common_filecontent.json")))
    offsets = group_offsets(unpersist(join(kwargs["dataset_path"], "common_offsets.json")))

    data = []
    nlp = create_tokenizer("spacy")

    for ind, function_data in enumerate(iterate_functions(offsets, node_maps, filecontent)):
        entry = process_body(
            nlp, function_data.pop("body"), replacements=function_data.pop("offsets"),
            require_labels=kwargs["require_labels"], remove_default=remove_default, remove_docstring=True
        )

        if entry is not None:
            entry.update(function_data)
            data.append(entry)

    store(data, **kwargs)


def create_from_environments(**kwargs):
    global_names = load_names(kwargs["global_nodes"])

    remove_default = kwargs["remove_default"]

    data = []

    for package in os.listdir(kwargs["dataset_path"]):
        pkg_path = os.path.join(kwargs["dataset_path"], package)
        if not os.path.isdir(pkg_path):
            continue

        data.extend(
            process_package(
                working_directory=pkg_path, global_names=global_names, require_labels=kwargs["require_labels"],
                remove_default=remove_default
            )
        )

    store(data, **kwargs)


def store(data, **kwargs):
    if kwargs["format"] == "json":  # json format is used by spacy
        with open(kwargs["output_path"], "w") as sink:
            for entry in data:
                sink.write(f"{json.dumps(entry)}\n")
    elif kwargs["format"] == "csv":
        if os.path.isfile(kwargs["output_path"]):
            header = False
        else:
            header = True
        pd.DataFrame(data).to_csv(kwargs["output_path"], index=False, header=header)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("dataset_format", type=str)
    parser.add_argument("dataset_path", type=str, help="")
    parser.add_argument("output_path", type=str, help="")
    parser.add_argument("--format", "-f", dest="format", default="json", help="json|csv")
    parser.add_argument("--remove_default", action="store_true", default=False)
    parser.add_argument("--global_nodes", "-g", dest="global_nodes", default=None)
    parser.add_argument("--require_labels", action="store_true", default=False)
    args = parser.parse_args()

    if args.dataset_format == "envs":
        create_from_environments(**args.__dict__)
    elif args.dataset_format == "dataset":
        create_from_dataset(**args.__dict__)
    else:
        raise ValueError("Supported dataset formats are: envs|dataset")
    # remove_default = False
