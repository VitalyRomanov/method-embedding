import ast

from SourceCodeTools.proc.entity.annotator.annotator_utils import to_offsets, resolve_self_collision

def get_mentions(function, root, mention):
    mentions = []

    for node in ast.walk(root):
        if isinstance(node, ast.Name):
            if node.id == mention:
                offset = to_offsets(function,
                                    [(node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "mention")])

                mentions.extend(offset)

    # hack for deduplication
    # the origin of duplicates is still unknown
    # it apears that mention contain false alarms....
    mentions = resolve_self_collision(mentions)

    return mentions


def get_descendants(function, children):

    descendants = []

    for chld in children:
        for node in ast.walk(chld):
            if isinstance(node, ast.Name):
                offset = to_offsets(function,
                                    [(node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "new_var")])
                descendants.append((node.id, offset[-1]))

    return descendants



def get_declarations(function):
    root = ast.parse(function)

    declarations = {}
    added = set()

    for node in ast.walk(root):
        if isinstance(node, ast.arg):
            if node.annotation is None:
                offset = to_offsets(function,
                                    [(node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "arg")])

                assert function[offset[-1][0]:offset[-1][1]] == node.arg, f"{function[offset[-1][0]:offset[-1][1]]} != {node.arg}"

                declarations[offset[-1]] = get_mentions(function, root, node.arg)
                added.add(node.arg)
        elif isinstance(node, ast.Assign):
            desc = get_descendants(function, node.targets)

            for d in desc:
                if d[0] not in added:
                    mentions = get_mentions(function, root, d[0])
                    valid_mentions = list(filter(lambda mention: mention[0] >= d[1][0], mentions))
                    declarations[d[1]] = valid_mentions
                    added.add(d[0])

    return declarations




if __name__ == "__main__":
    f = """def get_signature_declarations(function):
    root = ast.parse(function)

    declarations = {}
    
    function = 4

    for node in ast.walk(root):
        if isinstance(node, ast.arg):
            if node.annotation is None:
                offset = to_offsets(function, [(node.lineno, node.end_lineno, node.col_offset, node.end_col_offset, "arg")])
                # print(ast.dump(node), node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
                declarations[offset[-1]] = get_mentions(function, root, node.arg)
                print(declarations)
    """
    declarations = get_declarations(f)

    for dec, mentions in declarations.items():
        print(f"{f[dec[0]: dec[1]]} {dec}", end=": ")
        for m in mentions:
            print(f"{f[m[0]: m[1]]} {m}", end="\t ")
        print()

