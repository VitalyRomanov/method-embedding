# from python_ast import AstGraphGenerator
import sys, os
import pandas as pd
# import numpy as np
import ast
from pprint import pprint

working_directory = sys.argv[1]

source_location_path = os.path.join(working_directory, "source_location.csv")
occurrence_path = os.path.join(working_directory, "occurrence.csv")
node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
edge_path = os.path.join(working_directory, "edges.csv")
filecontent_path = os.path.join(working_directory, "filecontent.csv")

source_location = pd.read_csv(source_location_path, sep=",")
occurrence = pd.read_csv(occurrence_path, sep=",")
node = pd.read_csv(node_path, sep=",")
edge = pd.read_csv(edge_path, sep=",")
filecontent = pd.read_csv(filecontent_path, sep=",")

node_edge = pd.concat([node, edge], sort=False)

assert len(node_edge["id"].unique()) == len(node_edge)

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
    if start - 1 > 0 and ( \
            line[start - 1] >= "A" and line[start - 1] <= "Z" or \
            line[start - 1] >= "a" and line[start - 1] <= "z" or \
            line[start - 1] == "."):
        return extend_range(start - 1, end, line)
    else:
        return start, end

def get_docstring_ast(body):
    ast_parse = ast.parse(body.strip())
    function_definitions = [node for node in ast_parse.body if isinstance(node, ast.FunctionDef)]
    return ast.get_docstring(function_definitions[0])

bodies = []

for group_id, group in occurrence_group:


    definitions = group.query(f"occ_type == {DEFINITION_TYPE} and (type == 4096 or type == 8192)")

    if len(definitions):
        # print("\n\n\n")
        # print(definitions)
        for ind, row in definitions.iterrows():
            elements = group.query(f"start_line >= {row.start_line} and end_line <= {row.end_line} and occ_type != {DEFINITION_TYPE}")

            sources = filecontent.query(f"id == {group_id}").iloc[0]['content'].split("\n")

            body = "\n".join(sources[row.start_line - 1: row.end_line - 1])
            bodies.append({"id": row.element_id, "body": body, "docstring": get_docstring_ast(body)})
            # print(body)

            for start_line_g, sl_grp in elements.groupby("start_line"):

                valid_elements = sl_grp.query("start_line == end_line")
                valid_elements.sort_values(by="end_column", inplace=True, ascending=False)
                # elements.sort_values(by="start_line", inplace=True)
                # print(valid_elements)

                prev_line = 0
                curr_line = 1


                for ind, row_elem in valid_elements.iterrows():
                    if row_elem.start_line == row_elem.end_line:

                        curr_line = row_elem.start_line
                        if prev_line != curr_line:
                            replaced_ranges = []

                        line = sources[curr_line - 1]

                        start_c = row_elem.start_column - 1
                        end_c = row_elem.end_column

                        if not overlap((start_c, end_c), replaced_ranges):
                            e_start, e_end = extend_range(start_c, end_c, line)
                            replaced_ranges.append((e_start, e_end))
                            st_id = row_elem.element_id
                            # name = row_elem.serialized_name
                            # if not isinstance(name, str):
                            #     name = node_edge.query(f"element_id == {int(row_elem.target_node_id)}").iloc[0].serialized_name
                            #     if not isinstance(name, str):
                            #         name = "empty_name"
                            #
                            # print(curr_line, start_c, end_c, line[e_start: e_end], name)
                            # sources[curr_line - 1] = sources[curr_line - 1].replace(line[e_start: e_end], name)
                            # print(curr_line, start_c, end_c, line[e_start: e_end], f"sourcetrail_node__{st_id}")
                            sources[curr_line - 1] = sources[curr_line - 1][:e_start] + f"sourcetrail_node__{st_id}" + sources[curr_line - 1][e_end:]
                        prev_line = curr_line

            body = "\n".join(sources[row.start_line - 1: row.end_line - 1])
            # print(body)
            bodies[-1]["normalized_body"] = body
            # pprint(bodies[-1])
            # print("\n\n\n")

source_graph_docstring_path = os.path.join(working_directory, "source-graph-bodies.csv")
pd.DataFrame(bodies).to_csv(source_graph_docstring_path, index=False)