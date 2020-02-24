#%%
import pandas as pd
import os 
import json

working_directory = "/Volumes/External/datasets/Code/source-graphs/python-source-graph"

source_location_path = os.path.join(working_directory, "source_location.csv")
occurrence_path = os.path.join(working_directory, "occurrence.csv")
node_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")
filecontent_path = os.path.join(working_directory, "filecontent.csv")

# TODO:
# 1. Account for missing files

#%%

source_location = pd.read_csv(source_location_path, sep=",")
occurrence = pd.read_csv(occurrence_path, sep=",")
node = pd.read_csv(node_path, sep=",")

#%%
source_location.rename(columns={'id':'source_location_id'}, inplace=True)
node.rename(columns={'id':'element_id'}, inplace=True)

DEFINITION_TYPE = 1

definition_nodes = occurrence.merge( 
                            source_location[source_location.type == DEFINITION_TYPE], 
                            on='source_location_id',)


function_definitions = node.query('type == 4096 or type == 8192')[['element_id']].merge(
    definition_nodes, on='element_id', 
)

#%%

definition_index = dict()

for row_ind, row in function_definitions.iterrows():
    if row.file_node_id not in definition_index:
        definition_index[row.file_node_id] = dict()

    definition_index[row.file_node_id][row.element_id] = (row.start_line, row.end_line)


#%%
import re 

def get_docstring(body):
    docstring = []

    for ind, line in enumerate(body[1:]):
        match = re.search("^\s+#", line)
        if ind == 0 and match:
            docstring.append(line[match.span()[1]:].strip().lstrip())
        elif ind != 0 and match and len(docstring):
            docstring.append(line[match.span()[1]:].strip().lstrip())
        else: 
            break

    if not docstring:
        found_start = False
        finished = False
        for ind, line in enumerate(body[1:]):
            match = re.search("^\s+\"\"\"", line)
            if ind == 0 and match:
                found_start = True
                docstring.append(line[match.span()[1]:].strip().lstrip())
            elif ind != 0 and found_start:
                end_match = re.search("\"\"\"", line)
                if end_match:
                    docstring.append(line[:end_match.span()[0]].strip().lstrip())
                    finished = True
                    break
                else:
                    docstring.append(line.strip().lstrip())
            else: 
                break

        if not finished:
            docstring = []

    return "\n".join(docstring)

count = 0

docstrings = []

for chunk in pd.read_csv(filecontent_path, chunksize=100, sep=",", header=0):
    for row_ind, row in chunk.iterrows():
        # print(row)
        if functions := definition_index.get(row.id, 0):
            for func_id, func_range in functions.items():
                body = row.content.split("\n")[func_range[0]-1: func_range[1]-1]
                docstring = get_docstring(body)
                if docstring:
                    docstrings.append({
                        "id": int(func_id),
                        "docstring": docstring
                    })
                    # print(row.id, func_id, func_range)
                    # print(docstring)



source_graph_docstring_path = os.path.join(working_directory, "source-graph-docstring.jsonl")
with open(source_graph_docstring_path, "w") as source_graph_docstring:
    for ds in docstrings:
        source_graph_docstring.write("%s\n" % json.dumps(ds))
    






# %%
