import ast
import pandas
import sys
from collections import Counter

from SourceCodeTools.data.sourcetrail.file_utils import *

pandas.options.mode.chained_assignment = None

def main():
    lang = sys.argv[1]
    working_directory = sys.argv[2]

    if lang == "py":
        pass
    elif lang == "java":
        pass
        import javac_parser
        java = javac_parser.Java()
    else:
        raise ValueError("Valid languages: py, java")

    nodes = read_nodes(working_directory)
    bodies = read_processed_bodies(working_directory)

    id_offset = nodes["id"].max() + 1
    bodies = bodies[['id', 'body']].dropna(axis=0)


    if lang == "java":
        nodes = read_nodes(working_directory)
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

            func_var_pairs.append((row['id'], v))

        print(f"\r{body_ind}/{len(bodies)}", end="")
    print(" " * 30, end ="\r")

    if func_var_pairs:
        counter = Counter(map(lambda x: x[1], func_var_pairs))
        pp = []
        for func, var in func_var_pairs:
            if counter[var] > 1:
                pp.append({
                    'src': func,
                    'dst': var
                })
        pairs = pd.DataFrame(pp)
        persist(pairs, os.path.join(working_directory, filenames["function-variable-pairs"]))


if __name__ == "__main__":
    main()
