import ast
import sys
from typing import Tuple, List, Optional

from SourceCodeTools.code.data.sourcetrail.common import *


pd.options.mode.chained_assignment = None  # default='warn'


def overlap(range: Tuple[int, int], ranges: List[Tuple[int, int]]) -> bool:
    for r in ranges:
        if (r[0] - range[0]) * (r[1] - range[1]) <= 0:
            return True
    return False


def isnamechar(char: str) -> bool:
    return "A" <= char <= "Z" or \
           "a" <= char <= "z" or \
           char == "." or \
           char == "_" or \
           "0" <= char <= "9"


def extend_range(start: int, end: int, line: str) -> Optional[Tuple[int, int]]:
    # assume only the following symbols are possible in names: A-Z a-z 0-9 . _
    # if start - 1 > 0 and line[start - 1] == "!":
    #     # used in f-strings
    #     # f"[{attr_selector!r}]"
    #     return None
    return start, end
    # if start - 1 > 0 and isnamechar(line[start-1]):
    #     return extend_range(start - 1, end, line)
    # elif start - 1 > 0 and line[start - 1] == "!":
    #     # used in f-strings
    #     # f"[{attr_selector!r}]"
    #     return None
    # else:
    #     if start - 1 > 0 and line[start] == "." and not isnamechar(line[start - 1]):
    #         return start + 1, end
    #     return start, end


def get_docstring_ast(body):
    try:
        # this does not work with java, docstring formats are different
        ast_parse = ast.parse(body.strip())
        function_definitions = [node for node in ast_parse.body if isinstance(node, ast.FunctionDef)]
        return ast.get_docstring(function_definitions[0])
    except:
        return ""


def get_function_body(file_content, file_id, start, end) -> str:
    source_lines = file_content.query(f"id == {file_id}").iloc[0]['content'].split("\n")

    body_lines = source_lines[start: end]

    body_num_lines = len(body_lines)

    trim = 0

    if body_num_lines > 2: # assume one line function, or signaure and return statement
        initial_indent = len(body_lines[0]) - len(body_lines[0].lstrip())

        while trim < body_num_lines:
            cline = body_lines[body_num_lines - trim - 1]
            if body_num_lines - 1 > 1:
                if cline.strip() == "":
                    trim += 1
                elif len(cline) - len(cline.lstrip()) <= initial_indent:
                    trim += 1
                else:
                    break
            else:
                break

        if trim == body_num_lines:
            trim = 0

    return "\n".join(source_lines[start: end - trim])


def has_valid_syntax(function_body):
    try:
        ast.parse(function_body.lstrip())
        return True
    except SyntaxError:
        return False


def get_range_for_replacement(occurrence, start_col, end_col, line, nodeid2name):
    extended_range = extend_range(start_col, end_col, line)

    if extended_range is None:
        return None, None

    st_id = occurrence.element_id

    name = occurrence.serialized_name
    if not isinstance(name, str):  # happens when id refers to an edge, not a node
        st_id = occurrence.target_node_id

    node_name = nodeid2name[st_id]

    if node_name == UNRESOLVED_SYMBOL:
        # this is an unresolved symbol, avoid
        return None, None
    else:
        name = f"srctrlnd_{st_id}"  # sourcetrailnode
        return extended_range, name


def process_body(body, local_occurrences, nodeid2name, f_id, f_start):
    body_lines = body.split("\n")

    local_occurrences = sort_occurrences(local_occurrences)

    list_of_replacements = []

    for occ_ind, occurrence in local_occurrences.iterrows():
        if occurrence.start_line == occurrence.end_line:

            curr_line = occurrence.start_line - 1 - f_start

            if curr_line >= len(body_lines):
                continue

            start_col = occurrence.start_column - 1
            end_col = occurrence.end_column

            extended_range, sourcetrail_name = get_range_for_replacement(
                occurrence, start_col, end_col, body_lines[curr_line], nodeid2name
            )

            if extended_range is not None:
                occ_col_start, occ_col_end = extended_range

                list_of_replacements.append((
                    curr_line, occ_col_start, occ_col_end, sourcetrail_name
                ))

    return {
        "id": f_id,
        "body": body,
        "docstring": get_docstring_ast(body),
        "replacement_list": list_of_replacements,
    }


def process_bodies(nodes, edges, source_location, occurrence, file_content, lang):

    occurrence_groups = get_occurrence_groups(nodes, edges, source_location, occurrence)

    bodies = []

    nodeid2name = dict(zip(nodes['id'], nodes['serialized_name']))

    for group_ind, (file_id, occurrences) in custom_tqdm(
            enumerate(occurrence_groups), message="Processing function bodies", total=len(occurrence_groups)
    ):

        function_definitions = get_function_definitions(occurrences)

        if len(function_definitions):
            for ind, f_def in function_definitions.iterrows():
                f_start = f_def.start_line
                f_end = f_def.end_line

                local_occurrences = get_occurrences_from_range(occurrences, start=f_start, end=f_end)

                # move to zero-index
                f_start -= 1
                f_end -= 1

                body = get_function_body(file_content, file_id, f_start, f_end)

                if not has_valid_syntax(body):
                    continue

                processed = process_body(body, local_occurrences, nodeid2name, f_def.element_id, f_start)

                if processed is not None:
                    bodies.append(processed)

    if len(bodies) > 0:
        bodies_processed = pd.DataFrame(bodies)
        return bodies_processed
    else:
        return None


if __name__ == "__main__":
    working_directory = sys.argv[1]
    try:
        lang = sys.argv[2]
    except:
        lang = "python"

    source_location = read_source_location(working_directory)
    occurrence = read_occurrence(working_directory)
    nodes = read_nodes(working_directory)
    edges = read_edges(working_directory)
    file_content = read_filecontent(working_directory)
    bodies_processed = process_bodies(nodes, edges, source_location, occurrence, file_content, lang)

    if bodies_processed is not None:
        write_processed_bodies(bodies_processed, working_directory)