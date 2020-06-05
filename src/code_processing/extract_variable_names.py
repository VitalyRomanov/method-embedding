#%%
import os
import ast
import sys
import pandas

working_directory = sys.argv[1]

# bodies_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/function_bodies/functions.csv" # sys.argv[1]
# nodes_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/normalized_sourcetrail_nodes.csv" # sys.argv[2]

bodies_path = os.path.join(working_directory, "source-graph-functions.csv")
nodes_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")

#%%

id_offset = pandas.read_csv(nodes_path)["id"].max() + 1

# %%

bodies = pandas.read_csv(bodies_path)
bodies.dropna(axis=0, inplace=True)

variable_names = dict()
func_var_pairs = []

for ind, row in bodies.iterrows():
    try:
        tree = ast.parse(row['content'].strip())
    except SyntaxError:
        continue
    variables = [n.id for n in ast.walk(tree) if type(n).__name__ == "Name"]
    for v in variables:
        if v not in variable_names:
            variable_names[v] = id_offset
            id_offset += 1

        # func_var_pairs.append((row['id'], variable_names[v]))
        func_var_pairs.append((row['id'], v))

        # print(f"{row['id']},{variable_names[v]}")

#%% 

# with open(os.path.join(working_directory, "source-graph-variable-name-ids.csv"), "w") as vnames:
#     vnames.write(f"id,name\n")
#     for item, value in variable_names.items():
#         vnames.write(f"{value},{item}\n")

#%%

with open(os.path.join(working_directory, "source-graph-function-variable-pairs.csv"), "w") as fvpairs:
    fvpairs.write("src,dst\n")
    for func, var in func_var_pairs:
        fvpairs.write(f"{func},{var}\n")

# %%
