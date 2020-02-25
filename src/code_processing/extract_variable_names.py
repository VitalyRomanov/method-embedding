#%%
 
import ast
import sys
import pandas

bodies_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/function_bodies/functions.csv" # sys.argv[1]
nodes_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/normalized_sourcetrail_nodes.csv" # sys.argv[2]

#%%

id_offset = pandas.read_csv(nodes_path)["id"].max() + 1

# %%

bodies = pandas.read_csv(bodies_path)
bodies.dropna(axis=0, inplace=True)

variable_names = dict()
func_var_pairs = []

for ind, row in bodies.iterrows():
    try:
        tree = ast.parse(row['content'])
    except SyntaxError:
        continue
    variables = [n.id for n in ast.walk(tree) if type(n).__name__ == "Name"]
    for v in variables:
        if v not in variable_names:
            variable_names[v] = id_offset
            id_offset += 1

        func_var_pairs.append((row['id'], variable_names[v]))

        # print(f"{row['id']},{variable_names[v]}")

#%% 

with open("variable_name_ids.txt", "w") as vnames:
    vnames.write(f"id,name\n")
    for item, value in variable_names.items():
        vnames.write(f"{value},{item}\n")

#%%

with open("function_variable_pairs.txt", "w") as fvpairs:
    for func, var in func_var_pairs:
        fvpairs.write(f"{func},{var}\n")

# %%
