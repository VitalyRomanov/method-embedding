# from python_ast import AstGraphGenerator
import sys, os
import pandas as pd
# import numpy as np
import ast
from pprint import pprint
# from node_name_serializer import serialize_node_name
# from nltk import RegexpTokenizer
#
# tokenizer = RegexpTokenizer(
#             "[A-Za-z_0-9]+|[^\w\s]"
#         )

pd.options.mode.chained_assignment = None  # default='warn'

working_directory = sys.argv[1]

source_location_path = os.path.join(working_directory, "source_location.csv")
occurrence_path = os.path.join(working_directory, "occurrence.csv")
node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
edge_path = os.path.join(working_directory, "edges.csv")
filecontent_path = os.path.join(working_directory, "filecontent.csv")

print("Reading data...", end ="")
source_location = pd.read_csv(source_location_path, sep=",")
occurrence = pd.read_csv(occurrence_path, sep=",")
node = pd.read_csv(node_path, sep=",")
edge = pd.read_csv(edge_path, sep=",")
filecontent = pd.read_csv(filecontent_path, sep=",")

node_edge = pd.concat([node, edge], sort=False).astype({"target_node_id": "Int32", "source_node_id": "Int32"})

print("ok", end ="\n")

assert len(node_edge["id"].unique()) == len(node_edge), f"{len(node_edge['id'].unique())} != {len(node_edge)}"

source_location.rename(columns={'id':'source_location_id', 'type':'occ_type'}, inplace=True)
node_edge.rename(columns={'id':'element_id'}, inplace=True)

occurrences = occurrence.merge(source_location, on='source_location_id',)

nodes = node_edge.merge(occurrences, on='element_id')

occurrence_group = nodes.groupby("file_node_id")

DEFINITION_TYPE = 1

def overlap(range, ranges):
    for r in ranges:
        if (r[0] - range[0]) * (r[1] - range[1]) <= 0:
            return True
    return False

def extend_range(start, end, line):
    # assume only the following symbols are possible in names: A-Z a-z 0-9 . _
    if start - 1 > 0 and ( \
            line[start - 1] >= "A" and line[start - 1] <= "Z" or \
            line[start - 1] >= "a" and line[start - 1] <= "z" or \
            line[start - 1] == "." or \
            line[start - 1] == "_" or \
            line[start - 1] >= "0" and line[start - 1] <= "9"):
        return extend_range(start - 1, end, line)
    else:
        return start, end

def get_docstring_ast(body):
    try:
        # this does not work with java, docstring formats are different
        ast_parse = ast.parse(body.strip())
        function_definitions = [node for node in ast_parse.body if isinstance(node, ast.FunctionDef)]
        return ast.get_docstring(function_definitions[0])
    except:
        return ""

bodies = []

for occ_ind, (group_id, group) in enumerate(occurrence_group):


    definitions = group.query(f"occ_type == {DEFINITION_TYPE} and (type == 4096 or type == 8192)")

    if len(definitions):
        # print("\n\n\n")
        # print(definitions)
        for ind, row in definitions.iterrows():
            elements = group.query(f"start_line >= {row.start_line} and end_line <= {row.end_line} and occ_type != {DEFINITION_TYPE} and start_line == end_line")

            sources = filecontent.query(f"id == {group_id}").iloc[0]['content'].split("\n")

            body = "\n".join(sources[row.start_line - 1: row.end_line - 1])
            bodies.append({"id": row.element_id, "body": body, "docstring": get_docstring_ast(body)})
            # print(body)

            elements.sort_values(by=["start_line", "end_column"], inplace=True, ascending=[True, False])

            # print(elements)

            # for start_line_g, sl_grp in elements.groupby("start_line"):

                # valid_elements = sl_grp.query("start_line == end_line")
                # valid_elements.sort_values(by="end_column", inplace=True, ascending=False)

                # TODO:
                #  sorting does not help with java!!! see hack below
                #          element_id  type                                    serialized_name  source_node_id  ...  start_column  end_line  end_column  occ_type
                # 169736           34  8192  edu.stanford.nlp.coref.hybrid.ChineseCorefBenc...             NaN  ...             3        33          75         4
                # 2516264         152     2                                                NaN            34.0  ...            67        33          75         0
                # 2516263         150     2                                                NaN            34.0  ...            38        33          44         0
                # 169734           34  8192  edu.stanford.nlp.coref.hybrid.ChineseCorefBenc...             NaN  ...            25        33          36         0
                # 2516257         149     2                                                NaN            34.0  ...            18        33          23         0
                #
                # [5 rows x 12 columns]
                #   private static String runCorefTest(boolean deleteOnExit) throws Exception {

                # elements.sort_values(by="start_line", inplace=True)
                # print(valid_elements)

            prev_line = 0
            curr_line = 1

            for ind, row_elem in elements.iterrows():
                if row_elem.start_line == row_elem.end_line:

                    curr_line = row_elem.start_line
                    if prev_line != curr_line:
                        replaced_ranges = []

                    line = sources[curr_line - 1]

                    start_c = row_elem.start_column - 1
                    end_c = row_elem.end_column

                    # this is a hack for java, some annotations in java have a large span
                    if " " in sources[curr_line - 1][start_c: end_c]:
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

                        name = f"srstrlnd_{st_id}" # sourcetrailnode

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

                        sources[curr_line - 1] = sources[curr_line - 1][:e_start] + name + \
                                                 sources[curr_line - 1][e_end:]
                    prev_line = curr_line

            norm_body = "\n".join(sources[row.start_line - 1: row.end_line - 1])
            bodies[-1]["normalized_body"] = norm_body

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
pd.DataFrame(bodies).to_csv(source_graph_docstring_path, index=False)