from csv import QUOTE_NONNUMERIC
import pandas as pd
import sys, os
import ast
import string, random

from typing import Tuple, List
from pprint import pprint

DEFINITION_TYPE = 1
UNRESOLVED_SYMBOL = 1

# from node_name_serializer import serialize_node_name
# from nltk import RegexpTokenizer
#
# tokenizer = RegexpTokenizer(
#             "[A-Za-z_0-9]+|[^\w\s]"
#         )

pd.options.mode.chained_assignment = None  # default='warn'

working_directory = sys.argv[1]
try:
    lang = sys.argv[2]
except:
    lang = "python"

source_location_path = os.path.join(working_directory, "source_location.csv")
occurrence_path = os.path.join(working_directory, "occurrence.csv")
node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
edge_path = os.path.join(working_directory, "edges.csv")
filecontent_path = os.path.join(working_directory, "filecontent.csv")

source_location = pd.read_csv(source_location_path, sep=",", dtype={'id': int, 'file_node_id': int, 'start_line': int, 'start_column': int, 'end_line': int, 'end_column': int, 'type': int})
occurrence = pd.read_csv(occurrence_path, sep=",", dtype={'element_id': int, 'source_location_id': int})
node = pd.read_csv(node_path, sep=",", dtype={"id": int, "type": int, "serialized_name": str})
edge = pd.read_csv(edge_path, sep=",", dtype={'id': int, 'type': int, 'source_node_id': int, 'target_node_id': int})
filecontent = pd.read_csv(filecontent_path, sep=",", dtype={'id': int, 'content': str})

# merge nodes and edges, some references in code point to edges, not to nodes
node_edge = pd.concat([node, edge], sort=False).astype({"target_node_id": "Int32", "source_node_id": "Int32"})
assert len(node_edge["id"].unique()) == len(node_edge), f"{len(node_edge['id'].unique())} != {len(node_edge)}"

# rename columns
source_location.rename(columns={'id':'source_location_id', 'type':'occ_type'}, inplace=True)
node_edge.rename(columns={'id':'element_id'}, inplace=True)

# join tables
occurrences = occurrence.merge(source_location, on='source_location_id',)
nodes = node_edge.merge(occurrences, on='element_id')
occurrence_group = nodes.groupby("file_node_id")


def overlap(range: Tuple[int, int], ranges: List[Tuple[int, int]]) -> bool:
    for r in ranges:
        if (r[0] - range[0]) * (r[1] - range[1]) <= 0:
            return True
    return False


def isnamechar(char: str) -> bool:
    return char >= "A" and char <= "Z" or \
        char >= "a" and char <= "z" or \
        char == "." or \
        char == "_" or \
        char >= "0" and char <= "9"


def extend_range(start: int, end: int, line: str) -> Tuple[int, int]:
    # assume only the following symbols are possible in names: A-Z a-z 0-9 . _
    if start - 1 > 0 and isnamechar(line[start-1]):
        return extend_range(start - 1, end, line)
    else:
        if start - 1 > 0 and line[start] == "." and not isnamechar(line[start - 1]):
        # if start - 1 > 0 and line[start] == "." and ( line[start - 1] in [')', ']', '}', '"', '\''] or
        #         line[0: start].isspace()):
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

def generate_random_remplacement(len: int, source: List[str]):
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
                raise Exception("Could not replace with random name")

    return replacement



bodies = []


for occ_ind, (group_id, group) in enumerate(occurrence_group):

    definitions = group.query(f"occ_type == {DEFINITION_TYPE} and (type == 4096 or type == 8192)")

    if len(definitions):
        for ind, row in definitions.iterrows():
            f_start = row.start_line
            f_end = row.end_line

            elements: pd.DataFrame = group.query(f"start_line >= {f_start} and end_line <= {f_end} and occ_type != {DEFINITION_TYPE} and start_line == end_line")

            # list of lines of code
            sources: List[str] = filecontent.query(f"id == {group_id}").iloc[0]['content'].split("\n")

            # move to zero-index
            f_start -= 1
            # if lang != "java":
            f_end -= 1

            # if lang == "python":
            #     # assert that f_end is indeed the end of function
            #     assert len(sources[f_end - 1]) - len(sources[f_end - 1].lstrip()) != \
            #            len(sources[f_end]) - len(sources[f_end].lstrip())

            body: str = "\n".join(sources[f_start: f_end + 1])
            try:
                ast.parse(body.lstrip())
            except SyntaxError as e:
                continue

            bodies.append({"id": row.element_id, "body": body, "docstring": get_docstring_ast(body)})
            body_with_random_replacements = bodies[-1]['body'].split("\n")
            random_2_original = {}
            random_2_srctrl = {}

            assert sources[f_start] == body_with_random_replacements[0]

            elements.sort_values(by=["start_line", "end_column"], inplace=True, ascending=[True, False])

            prev_line = 0
            replaced_ranges = []
            list_of_replacements = []

            for ind, row_elem in elements.iterrows():
                if row_elem.start_line == row_elem.end_line:

                    curr_line = row_elem.start_line - 1
                    if prev_line != curr_line:
                        replaced_ranges = []
                        assert body_with_random_replacements[curr_line - f_start] == sources[curr_line]

                    line = sources[curr_line]

                    start_c = row_elem.start_column - 1
                    end_c = row_elem.end_column

                    # this is a hack for java, some annotations in java have a large span
                    # e.g. some annotations cover the entire signature declaration
                    if lang == 'java':
                        if " " in sources[curr_line][start_c: end_c]:
                            continue

                    if not overlap((start_c, end_c), replaced_ranges):
                        e_start, e_end = extend_range(start_c, end_c, line)
                        replaced_ranges.append((e_start, e_end))
                        st_id = row_elem.element_id

                        name = row_elem.serialized_name
                        if not isinstance(name, str): # happens when id refers to an edge, not a node
                            st_id = row_elem.target_node_id
                            # name = node_edge.query(f"element_id == {int(row_elem.target_node_id)}").iloc[0].serialized_name
                            # if not isinstance(name, str):
                            #     name = "empty_name"

                        node_info = node.query(f"id == {st_id}")
                        assert len(node_info) == 1

                        if node_info.iloc[0].type == UNRESOLVED_SYMBOL:
                            # this is an unresolved symbol, avoid
                            replaced_ranges.pop(-1)
                        else:
                            name = f"srctrlnd_{st_id}" # sourcetrailnode

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
                                curr_line - f_start, e_start, e_end, name
                            ))


                            random_name = generate_random_remplacement(len=e_end - e_start, source=body_with_random_replacements)
                            random_2_original[random_name] = body_with_random_replacements[curr_line - f_start][
                                                             e_start:e_end]
                            body_with_random_replacements[curr_line - f_start] = do_replacement(body_with_random_replacements[curr_line - f_start], e_start, e_end, random_name)
                            random_2_srctrl[random_name] = name


                            sources[curr_line] = do_replacement(sources[curr_line], e_start, e_end, name)
                            # sources[curr_line] = sources[curr_line][:e_start] + name + \
                            #                          sources[curr_line][e_end:]
                    prev_line = curr_line

            norm_body = "\n".join(sources[f_start: f_end + 1])
            body_with_random_replacements = "\n".join(body_with_random_replacements)
            bodies[-1]["normalized_body"] = norm_body
            bodies[-1]["replacement_list"] = repr(list_of_replacements)
            bodies[-1]["random_replacements"] = body_with_random_replacements
            bodies[-1]["random_2_original"] = random_2_original
            bodies[-1]["random_2_srctrl"] = random_2_srctrl

            # try:
            #     ast.parse(norm_body.lstrip())
            # except SyntaxError as e:
            #     print(e)
            #     pass

            # for line in sources[row.start_line - 1: row.end_line - 1]:
            #     for token in tokenizer.tokenize(line):
            #         if token.startswith("srstrlnd_"):
            #             if len(token.split("_")) != 2:
            #                 print(elements)
            #                 print()
            #                 print(body)
            #                 print()
            #                 print(norm_body)
            #                 raise  Exception()

            # pprint(bodies[-1])
    print(f"\r{occ_ind}/{len(occurrence_group)}", end="")

print(" " * 30, end="\r")

source_graph_docstring_path = os.path.join(working_directory, "source-graph-bodies.csv")
if len(bodies) != 0:
    pd.DataFrame(bodies).to_csv(source_graph_docstring_path, index=False, quoting=QUOTE_NONNUMERIC)
else:
    with open(source_graph_docstring_path, "w") as sink:
        sink.write("id,body,docstring,normalized_body,replacement_list\n")