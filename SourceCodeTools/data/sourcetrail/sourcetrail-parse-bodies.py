import ast
import random
import string
import sys
from typing import Tuple, List

from SourceCodeTools.data.sourcetrail.common import *

columns = [
    "id",
    "body",
    "body_normalized",
    "body_with_random_replacements",
    "docstring",
    "replacement_list",
    "random_2_original",
    "random_2_srctrl"
]


class RandomReplacementException(Exception):
    def __init__(self, message):
        super(RandomReplacementException, self).__init__(message)


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


def extend_range(start: int, end: int, line: str) -> Tuple[int, int]:
    # assume only the following symbols are possible in names: A-Z a-z 0-9 . _
    if start - 1 > 0 and isnamechar(line[start-1]):
        return extend_range(start - 1, end, line)
    else:
        if start - 1 > 0 and line[start] == "." and not isnamechar(line[start - 1]):
            return start + 1, end
        return start, end


def do_replacement(string_: str, start: int, end: int, substitution: str):
    return string_[:start] + substitution + \
                             string_[end:]


def get_docstring_ast(body):
    try:
        # this does not work with java, docstring formats are different
        ast_parse = ast.parse(body.strip())
        function_definitions = [node for node in ast_parse.body if isinstance(node, ast.FunctionDef)]
        return ast.get_docstring(function_definitions[0])
    except:
        return ""


def get_random_string(str_len):
    return "".join(random.choices(string.ascii_letters, k=str_len))


def generate_random_replacement(len: int, source: List[str]):
    attempts_left = 30
    secondary_attempts = 10
    replacement = get_random_string(len)
    body = "\n".join(source)
    while body.find(replacement) != -1:
        attempts_left -= 1
        replacement = get_random_string(len)
        if attempts_left <= 0:
            secondary_attempts -= 1
            replacement = "".join(random.choices("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧЪЫЬЭЮЯабвгдежзиклмнопрстуфхцчъыьэюя", k=len))
            if secondary_attempts == 0:
                raise RandomReplacementException("Could not replace with random name")

    return replacement


# def get_occurrence_groups(working_directory):
#     source_location = read_source_location(working_directory)
#     occurrence = read_occurrence(working_directory)
#     nodes = read_nodes(working_directory)
#     edges = read_edges(working_directory)
#
#     # merge nodes and edges, some references in code point to edges, not to nodes
#     node_edge = pd.concat([nodes, edges], sort=False).astype({"target_node_id": "Int32", "source_node_id": "Int32"})
#     assert len(node_edge["id"].unique()) == len(node_edge), f"{len(node_edge['id'].unique())} != {len(node_edge)}"
#
#     # rename columns
#     source_location.rename(columns={'id': 'source_location_id', 'type': 'occ_type'}, inplace=True)
#     node_edge.rename(columns={'id': 'element_id'}, inplace=True)
#
#     # join tables
#     occurrences = occurrence.merge(source_location, on='source_location_id', )
#     nodes = node_edge.merge(occurrences, on='element_id')
#     occurrence_groups = nodes.groupby("file_node_id")
#
#     return occurrence_groups


# def get_source_lines(file_content, file_id) -> List[str]:
#     file_content.query(f"id == {file_id}").iloc[0]['content'].split("\n")
#     return


def get_function_body(file_content, file_id, start, end) -> str:
    source_lines = file_content.query(f"id == {file_id}").iloc[0]['content'].split("\n")
    return "\n".join(source_lines[start: end + 1])


def has_valid_syntax(function_body):
    try:
        ast.parse(function_body.lstrip())
        return True
    except SyntaxError as e:
        return False


def get_range_for_replacement(occurrence, start_col, end_col, line, nodes):
    occ_col_start, occ_col_end = extend_range(start_col, end_col, line)
    extended_range = (occ_col_start, occ_col_end)

    st_id = occurrence.element_id

    name = occurrence.serialized_name
    if not isinstance(name, str):  # happens when id refers to an edge, not a node
        st_id = occurrence.target_node_id
        # name = node_edge.query(f"element_id == {int(row_elem.target_node_id)}").iloc[0].serialized_name
        # if not isinstance(name, str):
        #     name = "empty_name"

    node_info = nodes.query(f"id == {st_id}")
    assert len(node_info) == 1

    if node_info.iloc[0].type == UNRESOLVED_SYMBOL:
        # this is an unresolved symbol, avoid
        return None, None
    else:
        name = f"srctrlnd_{st_id}"  # sourcetrailnode
        return extended_range, name


def get_occurrence_string(line, col_start, col_end):
    return line[col_start: col_end]


def main(args):

    working_directory = args[1]
    try:
        lang = args[2]
    except:
        lang = "python"

    nodes = read_nodes(working_directory)
    file_content = read_filecontent(working_directory)

    occurrence_groups = get_occurrence_groups(working_directory)

    bodies = []

    for group_ind, (file_id, occurrences) in enumerate(occurrence_groups):

        function_definitions = get_function_definitions(occurrences)

        if len(function_definitions):
            for ind, f_def in function_definitions.iterrows():
                f_start = f_def.start_line
                f_end = f_def.end_line

                local_occurrences = get_occurrences_from_range(occurrences, start=f_start, end=f_end)

                # list of lines of code
                # source_lines = get_source_lines(file_content, file_id)

                # move to zero-index
                f_start -= 1
                # if lang != "java":
                f_end -= 1

                # if lang == "python":
                #     # assert that f_end is indeed the end of function
                #     assert len(sources[f_end - 1]) - len(sources[f_end - 1].lstrip()) != \
                #            len(sources[f_end]) - len(sources[f_end].lstrip())

                body = get_function_body(file_content, file_id, f_start, f_end)

                if not has_valid_syntax(body):
                    continue

                body_normalized = body.split("\n")
                body_with_random_replacements = body.split("\n")
                random_2_original = {}
                random_2_srctrl = {}

                # assert source_lines[f_start] == body_with_random_replacements[0]

                local_occurrences = sort_occurrences(local_occurrences)

                prev_line = 0
                replaced_ranges = []
                list_of_replacements = []

                for occ_ind, occurrence in local_occurrences.iterrows():
                    if occurrence.start_line == occurrence.end_line:

                        curr_line = occurrence.start_line - 1 - f_start
                        if prev_line != curr_line:
                            replaced_ranges = []
                            # assert body_with_random_replacements[curr_line - f_start] == source_lines[curr_line]

                        # line = body_normalized[curr_line]

                        start_col = occurrence.start_column - 1
                        end_col = occurrence.end_column

                        # this is a hack for java, some annotations in java have a large span
                        # e.g. some annotations cover the entire signature declaration
                        if lang == 'java':
                            if " " in body_normalized[curr_line][start_col: end_col]:
                                continue

                        if not overlap((start_col, end_col), replaced_ranges):
                            extended_range, sourcetrail_name = get_range_for_replacement(
                                occurrence, start_col, end_col, body_normalized[curr_line], nodes
                            )

                            if extended_range is not None:

                                replaced_ranges.append(extended_range)
                                occ_col_start, occ_col_end = extended_range

                                # this is a hack for java
                                # remove special symbols so that code can later be parsed by ast parser
                                # name = name.replace("___", "__stspace__")
                                # name = name.replace(")", "__strrbr__")
                                # name = name.replace("(", "__stlrbr__")
                                # name = name.replace(">", "__strtbr__")
                                # name = name.replace("<", "__stltbr__")
                                # name = name.replace("?", "__qmark__")
                                # name = name.replace("@", "__stat__")
                                # name = name.replace('.', '____')

                                list_of_replacements.append((
                                    curr_line, occ_col_start, occ_col_end, sourcetrail_name
                                ))

                                random_name = generate_random_replacement(
                                    len=occ_col_end - occ_col_start, source=body_with_random_replacements
                                )
                                random_2_original[random_name] = get_occurrence_string(
                                    body_with_random_replacements[curr_line], occ_col_start, occ_col_end
                                )
                                body_with_random_replacements[curr_line] = do_replacement(
                                    body_with_random_replacements[curr_line], occ_col_start, occ_col_end, random_name
                                )
                                random_2_srctrl[random_name] = sourcetrail_name

                                body_normalized[curr_line] = do_replacement(
                                    body_normalized[curr_line], occ_col_start, occ_col_end, sourcetrail_name
                                )

                        prev_line = curr_line

                norm_body = "\n".join(body_normalized)
                body_with_random_replacements = "\n".join(body_with_random_replacements)
                bodies.append({
                    "id": f_def.element_id,
                    "body": body,
                    "body_normalized": norm_body,
                    "body_with_random_replacements": body_with_random_replacements,
                    "docstring": get_docstring_ast(body),
                    "replacement_list": list_of_replacements,
                    "random_2_original": random_2_original,
                    "random_2_srctrl": random_2_srctrl
                })

        print(f"\r{group_ind}/{len(occurrence_groups)}", end="")

    print(" " * 30, end="\r")

    if bodies:
        bodies_processed = pd.DataFrame(bodies, columns=columns)
    else:
        bodies_processed = pd.DataFrame(bodies)
        # .astype({
        # "id": int,
        # "body": str,
        # "body_normalized": str,
        # "body_with_random_replacements": str,
        # "docstring": str,
        # "replacement_list": list_of_replacements,
        # "random_2_original": random_2_original,
        # "random_2_srctrl": random_2_srctrl
    # })

    write_processed_bodies(bodies_processed, working_directory)
    # bodies_processed_path = os.path.join(working_directory, "source-graph-bodies.bz2")
    # persist(bodies_processed, bodies_processed_path)
    # write_parquet(bodies_processed, path=bodies_processed_path)
    # if len(bodies) != 0:
    #     pd.DataFrame(bodies).to_csv(source_graph_docstring_path, index=False, quoting=QUOTE_NONNUMERIC)
    # else:
    #     with open(source_graph_docstring_path, "w") as sink:
    #         sink.write("id,body,docstring,normalized_body,replacement_list,random_replacements,random_2_original,random_2_srctrl\n")


if __name__ == "__main__":
    replacement_attempts = 100

    while replacement_attempts >= 0:
        try:
            main(sys.argv)
        except RandomReplacementException:
            if replacement_attempts == 0:
                raise Exception("Could not replace with random names")
            replacement_attempts -= 1
        else:
            break
