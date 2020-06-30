#%%
import os
import ast
import sys
import pandas

lang = sys.argv[1]
working_directory = sys.argv[2]

if lang == "py":
    import ast
elif lang == "java":
    import javalang
else:
    raise ValueError("Valid languages: py, java")

# bodies_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/function_bodies/functions.csv" # sys.argv[1]
# nodes_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/normalized_sourcetrail_nodes.csv" # sys.argv[2]

bodies_path = os.path.join(working_directory, "source-graph-bodies.csv")
nodes_path = os.path.join(working_directory, "normalized_sourcetrail_nodes.csv")

#%%

id_offset = pandas.read_csv(nodes_path)["id"].max() + 1

# %%

bodies = pandas.read_csv(bodies_path)[['id', 'body']]
bodies.dropna(axis=0, inplace=True)

variable_names = dict()
func_var_pairs = []

for body_ind, (ind, row) in enumerate(bodies.iterrows()):
    try:
        if lang == "py":
            tree = ast.parse(row['body'].strip())
        elif lang == "java":
            tokens = javalang.tokenizer.tokenize(row['body'].strip().replace("\n", " "))
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse_expression()
        else: continue
    except SyntaxError: # thrown by ast
        continue
    except javalang.parser.JavaSyntaxError: # thrown by javalang
        continue
    except TypeError: # thrown by javalang
        pass
    except StopIteration: # thrown by javalang
        pass

    if lang == "py":
        variables = [n.id for n in ast.walk(tree) if type(n).__name__ == "Name"]
    elif lang == "java":
        variables = [node.member for _, node in tree if type(node) is javalang.tree.MemberReference]

    # print(variables)
    # print(tree)
    # print(row['body'].strip())
    #
    # import sys; sys.exit()

    for v in variables:
        if v not in variable_names:
            variable_names[v] = id_offset
            id_offset += 1

        # func_var_pairs.append((row['id'], variable_names[v]))
        func_var_pairs.append((row['id'], v))

        # print(f"{row['id']},{variable_names[v]}")
    if body_ind % 1000 == 0:
        print(f"\r{body_ind}/{len(bodies)}", end="")
print(" " * 30, end ="\r")

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
