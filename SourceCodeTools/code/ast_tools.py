import ast

from SourceCodeTools.nlp.entity.annotator.annotator_utils import to_offsets, resolve_self_collision, adjust_offsets2


def get_mentions(function, root, mention):
    """
    Find all mentions of a variable in the function's body
    :param function: string that contains function's body
    :param root: body parsed with ast package
    :param mention: the name of a variable to look for
    :return: list of offsets where the variable is mentioned
    """
    mentions = []

    for node in ast.walk(root):
        if isinstance(node, ast.Name): # a variable or a ...
            if node.id == mention:
                offset = to_offsets(function,
                                    [(node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "mention")], as_bytes=True)

                mentions.extend(offset)

    # hack for deduplication
    # the origin of duplicates is still unknown
    # it apears that mention contain false alarms....
    mentions = resolve_self_collision(mentions)

    return mentions


def get_descendants(function, children):
    """

    :param function: function string
    :param children: List of targets.
    :return: Offsets for attributes or names that are used as target for assignment operation. Subscript, Tuple and List
    targets are skipped.
    """
    descendants = []

    # if isinstance(children, ast.Tuple):
    #     descendants.extend(get_descendants(function, children.elts))
    # else:
    for chld in children:
        # for node in ast.walk(chld):
        node = chld
        if isinstance(node, ast.Attribute) or isinstance(node, ast.Name):
        # if isinstance(node, ast.Name):
            offset = to_offsets(function,
                                [(node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "new_var")], as_bytes=True)
            # descendants.append((node.id, offset[-1]))
            descendants.append((function[offset[-1][0]:offset[-1][1]], offset[-1]))
        # elif isinstance(node, ast.Tuple):
        #     descendants.extend(get_descendants(function, node.elts))
        elif isinstance(node, ast.Subscript) or isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            pass # skip for now
        else:
            raise Exception("")

    return descendants



def get_declarations(function_):
    """

    :param function:
    :return:
    """
    function = function_.lstrip()
    initial_strip = function_[:len(function_) - len(function)]

    root = ast.parse(function)

    declarations = {}
    added = set()

    for node in ast.walk(root):
        if isinstance(node, ast.arg): # function argument
            # TODO
            # not quite sure why this if statement was needed, but there should be no annotations in the code
            if node.annotation is None:
                offset = to_offsets(function,
                                    [(node.lineno-1, node.end_lineno-1, node.col_offset, node.end_col_offset, "arg")], as_bytes=True)

                assert function[offset[-1][0]:offset[-1][1]] == node.arg, f"{function[offset[-1][0]:offset[-1][1]]} != {node.arg}"

                declarations[offset[-1]] = get_mentions(function, root, node.arg)
                added.add(node.arg) # mark variable name as seen
        elif isinstance(node, ast.Assign):
            desc = get_descendants(function, node.targets)

            for d in desc:
                if d[0] not in added:
                    mentions = get_mentions(function, root, d[0])
                    valid_mentions = list(filter(lambda mention: mention[0] >= d[1][0], mentions))
                    declarations[d[1]] = valid_mentions
                    added.add(d[0])

    initial_strip_len = len(initial_strip)
    declarations = {
        adjust_offsets2([key], initial_strip_len)[0]: adjust_offsets2(val, initial_strip_len) for key, val in declarations.items()
    }

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

