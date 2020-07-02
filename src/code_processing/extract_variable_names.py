import os, sys, ast, pandas, javac_parser
from collections import Counter

lang = sys.argv[1]
working_directory = sys.argv[2]

if lang == "py":
    pass
elif lang == "java":
    pass
    java = javac_parser.Java()
else:
    raise ValueError("Valid languages: py, java")

bodies_path = os.path.join(working_directory, "source-graph-bodies.csv")
nodes_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")

id_offset = pandas.read_csv(nodes_path)["id"].max() + 1

bodies = pandas.read_csv(bodies_path)[['id', 'body']]
bodies.dropna(axis=0, inplace=True)


if lang == "java":
    nodes = pandas.read_csv(nodes_path)
    names = nodes['serialized_name'].apply(lambda x: x.split("___")[0].split("."))
    not_local = set()
    for name in names:
        for n in name:
            not_local.add(n)

variable_names = dict()
func_var_pairs = []

for body_ind, (ind, row) in enumerate(bodies.iterrows()):
    variables = []
    try:
        if lang == "py":
            tree = ast.parse(row['body'].strip())
            variables.extend([n.id for n in ast.walk(tree) if type(n).__name__ == "Name"])
        elif lang == "java":
            lines = row['body'].strip() #.split("\n")
            tokens = java.lex(lines)
            variables = [name for type, name, _, _, _ in tokens if type == "IDENTIFIER" and name not in not_local]
        else: continue
    except SyntaxError: # thrown by ast
        continue

    for v in set(variables):
        if v not in variable_names:
            variable_names[v] = id_offset
            id_offset += 1

        # func_var_pairs.append((row['id'], variable_names[v]))
        func_var_pairs.append((row['id'], v))

        # print(f"{row['id']},{variable_names[v]}")
    print(f"\r{body_ind}/{len(bodies)}", end="")
print(" " * 30, end ="\r")

#%% 

# with open(os.path.join(working_directory, "source-graph-variable-name-ids.csv"), "w") as vnames:
#     vnames.write(f"id,name\n")
#     for item, value in variable_names.items():
#         vnames.write(f"{value},{item}\n")

#%%

counter = Counter(map(lambda x: x[1], func_var_pairs))

with open(os.path.join(working_directory, "source-graph-function-variable-pairs.csv"), "w") as fvpairs:
    fvpairs.write("src,dst\n")
    for func, var in func_var_pairs:
        if counter[var] > 1:
            fvpairs.write(f"{func},{var}\n")

# TODO:
# string names should be embedded with char ngrams
