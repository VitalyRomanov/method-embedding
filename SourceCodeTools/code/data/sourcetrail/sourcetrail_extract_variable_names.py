import ast
import pandas
import sys
from collections import Counter

from SourceCodeTools.code.data.sourcetrail.common import custom_tqdm
from SourceCodeTools.code.data.sourcetrail.file_utils import *

pandas.options.mode.chained_assignment = None

def extract_var_names(nodes, bodies, lang):

    if lang == "python":
        pass
    elif lang == "java":
        pass
        import javac_parser
        java = javac_parser.Java()
    else:
        raise ValueError("Valid languages: python, java")

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

    for body_ind, (ind, row) in custom_tqdm(
            enumerate(bodies.iterrows()), message="Extracting variable names", total=len(bodies)
    ):
        variables = []
        try:
            if lang == "python":
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

        # print(f"\r{body_ind}/{len(bodies)}", end="")
    # print(" " * 30, end ="\r")

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
        return pairs
    else:
        return None


if __name__ == "__main__":
    lang = sys.argv[1]
    working_directory = sys.argv[2]

    nodes = read_nodes(working_directory)
    bodies = read_processed_bodies(working_directory)

    variables = extract_var_names(nodes, bodies, lang)

    if variables is not None:
        persist(variables, os.path.join(working_directory, filenames["function_variable_pairs"]))