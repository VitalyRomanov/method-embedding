import ast
from collections import defaultdict
from functools import partial

from SourceCodeTools.code.annotator_utils import to_offsets, resolve_self_collision, adjust_offsets2

# def get_mentions(function, root, mention):
#     """
#     Find all mentions of a variable in the function's body
#     :param function: string that contains function's body
#     :param root: body parsed with ast package
#     :param mention: the name of a variable to look for
#     :return: list of offsets where the variable is mentioned
#     """
#     mentions = []
#
#     for node in ast.walk(root):
#         if isinstance(node, ast.Name) or isinstance(node, ast.Attribute): # a variable or a ...
#             offset = get_node_offset(node, function, "mention")
#             if function[offset[0]:offset[1]] == mention:
#                 mentions.append(offset)
#     # hack for deduplication
#     # the origin of duplicates is still unknown
#     # it apears that mention contain false alarms....
#     mentions = resolve_self_collision(mentions)
#
#     return mentions

def get_node_offset(node, function, label):
    return to_offsets(
        function,
        [(node.lineno - 1, node.end_lineno - 1, node.col_offset, node.end_col_offset, label)]
        , as_bytes=True
    )[0]


# def get_descendants(function, children):
#     """
#
#     :param function: function string
#     :param children: List of targets.
#     :return: Offsets for attributes or names that are used as target for assignment operation. Subscript, Tuple and List
#     targets are skipped.
#     """
#     descendants = []
#
#     # if isinstance(children, ast.Tuple):
#     #     descendants.extend(get_descendants(function, children.elts))
#     # else:
#     for chld in children:
#         # for node in ast.walk(chld):
#         node = chld
#         if isinstance(node, ast.Attribute) or isinstance(node, ast.Name):
#         # if isinstance(node, ast.Name):
#             offset = get_node_offset(node, function, "new_var")
#             # descendants.append((node.id, offset[-1]))
#             descendants.append((function[offset[0]:offset[1]], offset))
#         # elif isinstance(node, ast.Tuple):
#         #     descendants.extend(get_descendants(function, node.elts))
#         elif isinstance(node, ast.Subscript) or isinstance(node, ast.Tuple) or isinstance(node, ast.List):
#             pass # skip for now
#         else:
#             raise Exception("")
#
#     return descendants


def strip_annotation_and_default_value_from_offset(offset, function):
    arg_str = function[offset[0]:offset[1]]
    for arg_smb in [":", "="]:
        if arg_smb in arg_str:
            arg_str_clr = arg_str.split(arg_smb)[0].strip()
            offset = (offset[0], offset[0] + len(arg_str_clr), offset[2])
            break
    return offset

class SpanBank:
    def __init__(self):
        self.spans = set()

    def __contains__(self, item):
        return item[:2] in self.spans

    def add(self, item):
        self.spans.add(item[:2])

    def add_bunch(self, items):
        for item in items:
            self.add(item)


class VariableTracker:
    def __init__(self, function):
        self.declarations = defaultdict(list)
        self.added_variable_names = set()
        self.added_offsets = SpanBank()

        self.function = function.lstrip()
        self.initial_strip = function[:len(function) - len(self.function)]

    def get_arg_declaration(self, node):
        function = self.function
        offset = strip_annotation_and_default_value_from_offset(
            get_node_offset(node, function, "arg"), function
        )
        assert function[offset[0]:offset[1]] == node.arg, f"{function[offset[0]:offset[1]]} != {node.arg}"

        variable_name = node.arg
        return variable_name, offset

    def get_var_declaration(self, node):
        function = self.function
        offset = strip_annotation_and_default_value_from_offset(
            get_node_offset(node, function, "new_var"), function
        )

        variable_name = function[offset[0]:offset[1]]
        return variable_name, offset

    def get_var_from_node(self, node, label="mention"):
        if isinstance(node, ast.Attribute) or isinstance(node, ast.Name):
            offset = get_node_offset(node, self.function, label)
            return self.function[offset[0]:offset[1]], offset
        elif isinstance(node, ast.Subscript) or isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            return None, None
        else:
            raise Exception("")

    def track_offsets_from_definition(self, offset):
        self.added_offsets.add(offset)
        self.added_offsets.add_bunch(self.declarations[offset])

    def add_declarations_from_definition(self, node, decl_fn):
        variable_name, offset = decl_fn(node)
        if offset is None:
            return

        if offset in self.added_offsets:
            return

        mentions = self.get_mentions(variable_name)
        mentions = list(filter(lambda mention: mention[0] > offset[0] and mention not in self.added_offsets, mentions))

        self.declarations[offset] = mentions

        self.added_variable_names.add(variable_name)  # mark variable name as seen
        self.track_offsets_from_definition(offset)

    def get_mentions(self, mention):
        mentions = []

        for mention_name, mention_span in self.categories["mentions"]:
            if mention_name == mention:
                mentions.append(mention_span)
        # function = self.function
        #
        # for node in ast.walk(root):
        #     if isinstance(node, ast.Name) or isinstance(node, ast.Attribute):  # a variable or a ...
        #         variable_name, offset = self.get_var_from_node(node, label="mention")
        #         offset = get_node_offset(node, function, "mention")
        #         if variable_name == mention:
        #             mentions.append(offset)
        # hack for deduplication
        # the origin of duplicates is still unknown
        # it apears that mention contain false alarms....
        mentions = resolve_self_collision(mentions)

        return mentions

    def get_attribute_root(self, node):
        if isinstance(node, ast.Attribute):
            return self.get_attribute_root(node.value)
        elif isinstance(node, ast.Name):
            return self.get_var_from_node(node, label="mention_helper")
        else:
            return None

    def inspect_tree(self, root):
        if isinstance(root, ast.Name) or isinstance(root, ast.Attribute):
            self.categories["mentions"].append(self.get_var_from_node(root, label="mention"))
            if isinstance(root, ast.Attribute):
                attr_root = self.get_attribute_root(root)
                if attr_root is not None:
                    self.categories["mentions"].append(attr_root)
        else:
            for node in ast.iter_child_nodes(root):
                if isinstance(node, ast.FunctionDef):
                    if self.function_def_met is False:
                        self.function_def_met = True
                        if hasattr(node, "args"):
                            for arg in node.args.args:
                                self.categories["arguments"].append(arg)
                            if node.args.kwarg is not None:
                                self.categories["arguments"].append(node.args.kwarg)
                            if node.args.vararg is not None:
                                self.categories["arguments"].append(node.args.vararg)
                    self.inspect_tree(node)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Subscript):
                            self.inspect_tree(t)
                        elif isinstance(t, ast.Tuple):
                            for elt in t.elts:
                                self.categories["declaration"].append(elt)
                                self.inspect_tree(elt) # add as mention
                        else:
                            self.categories["declaration"].append(t)
                            self.inspect_tree(t)  # add as mention
                        self.inspect_tree(node.value)
                elif isinstance(node, ast.AnnAssign):
                    t = node.target
                    if isinstance(t, ast.Subscript):
                        self.inspect_tree(t)
                    else:
                        self.categories["declaration"].append(t)
                        self.inspect_tree(t)  # add as mention
                    self.inspect_tree(node.value)
                # elif isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
                #     self.categories["mentions"].append(self.get_var_from_node(node, label="mention"))
                #     if isinstance(node, ast.Attribute):
                #         self.categories["mentions"].append(self.get_attribute_root(node))
                else:
                    self.inspect_tree(node)

    def get_all_mentions(self):
        root = ast.parse(self.function)

        self.categories = defaultdict(list)
        self.function_def_met = False
        self.inspect_tree(root)

        self.categories["mentions"] = sorted(self.categories["mentions"], key=lambda x: x[1][0])

        return self.categories

    def get_declarations(self):
        mention_categories = self.get_all_mentions()

        for node in mention_categories["arguments"]:
            self.add_declarations_from_definition(node, self.get_arg_declaration)

        for node in mention_categories["declaration"]:
            self.add_declarations_from_definition(node, self.get_var_declaration)

        rest_heads = {}
        for mention_name, mention_span in self.categories["mentions"]:
            if mention_span not in self.added_offsets:
                if mention_name not in rest_heads:
                    rest_heads[mention_name] = mention_span
                else:
                    self.declarations[rest_heads[mention_name]].append(mention_span)

        initial_strip_len = len(self.initial_strip)
        declarations = {
            adjust_offsets2([key], initial_strip_len)[0]: adjust_offsets2(val, initial_strip_len) for key, val in
            self.declarations.items()
        }



        # declarations[None] =

        return declarations


def get_declarations(function_):
    """

    :param function:
    :return:
    """

    tracker = VariableTracker(function_)
    declarations = tracker.get_declarations()
    # function = function_.lstrip()
    # initial_strip = function_[:len(function_) - len(function)]
    #
    # root = ast.parse(function)
    #
    # declarations = {}
    # added = set()
    # added_offsets = SpanBank()
    #
    # for node in ast.walk(root):
    #     if isinstance(node, ast.arg): # function argument
    #         offset = strip_annotation_and_default_value_from_offset(
    #             get_node_offset(node, function, "arg"), function
    #         )
    #         assert function[offset[0]:offset[1]] == node.arg, f"{function[offset[0]:offset[1]]} != {node.arg}"
    #
    #         declarations[offset] = get_mentions(function, root, node.arg)
    #
    #         added.add(node.arg) # mark variable name as seen
    #         added_offsets.add(offset)
    #         added_offsets.add_bunch(declarations[offset])
    #
    # for node in ast.walk(root):
    #     if isinstance(node, ast.Assign):
    #         desc = get_descendants(function, node.targets)
    #
    #         for desc_name, desc_offset in desc:
    #             if desc_name not in added:
    #                 mentions = get_mentions(function, root, desc_name)
    #                 valid_mentions = list(filter(lambda mention: mention[0] > desc_offset[0], mentions))
    #                 declarations[desc_offset] = valid_mentions
    #                 added.add(desc_name)
    #
    # initial_strip_len = len(initial_strip)
    # declarations = {
    #     adjust_offsets2([key], initial_strip_len)[0]: adjust_offsets2(val, initial_strip_len) for key, val in declarations.items()
    # }

    return declarations


def find_replacement():
    from itertools import chain
    import json
    from tqdm import tqdm

    def read_pairs(path):
        with open(path, "r") as source:
            while True:
                original_line = source.readline()
                modified_line = source.readline()
                if original_line == "" or modified_line == "":
                    break
                original = json.loads(original_line)
                assert original["label"] == "Correct"
                modified = json.loads(modified_line)
                assert original["function"][:10] == modified["function"][:10]

                yield original, modified

    for ind, (original, modified) in enumerate(tqdm(read_pairs("/home/ltv/Shared/var_misuse.jsonl"))):

        info_parts = modified["info"].split("`")
        var_o, var_m = info_parts[-4], info_parts[-2]

        decl_o = get_declarations(original["function"])
        decl_m = get_declarations(modified["function"])

        def get_key(function, declarations, var_name):
            for decl in declarations:
                if function[decl[0]: decl[1]] == var_name:
                    return decl

        def get_mentions(function, declarations, var_name):
            return declarations[get_key(function, declarations, var_name)]

        assert len(get_mentions(original["function"], decl_o, var_o)) - len(get_mentions(modified["function"], decl_m, var_o)) == 1
        assert len(get_mentions(modified["function"], decl_m, var_m)) - len(get_mentions(original["function"], decl_o, var_m)) == 1

        def get_incorrect():
            mentions_m = get_mentions(modified["function"], decl_m, var_m)
            mentions_o = get_mentions(original["function"], decl_o, var_m)
            for ind, m in enumerate(mentions_m):
                if ind >= len(mentions_o) or mentions_m != mentions_o[ind]:
                    return m
            return mentions_m[-1]

        incorrect = get_incorrect()
        correct = get_mentions(modified["function"], decl_m, var_o) + [get_key(modified["function"], decl_m, var_o)]
        candidates = list(decl_m.keys()) + list(chain(*decl_m.values()))
        print()



if __name__ == "__main__":
    find_replacement()
    f = """\n\ndef __init__(self, connection):\n    self.connection = self\n"""
    # f = """\n\ndef test_multipath_joins():\n    (app, db, admin) = setup()\n\n    class Model1(db.Model):\n        id = db.Column(db.Integer, primary_key=True)\n        val1 = db.Column(db.String(20))\n        test = db.Column(db.String(20))\n\n    class Model2(db.Model):\n        id = db.Column(db.Integer, primary_key=True)\n        val2 = db.Column(db.String(20))\n        first_id = db.Column(db.Integer, db.ForeignKey(Model1.id))\n        first = db.relationship(Model1, backref='first', foreign_keys=[first_id])\n        second_id = db.Column(db.Integer, db.ForeignKey(Model1.id))\n        second = db.relationship(Model1, backref='second', foreign_keys=[second_id])\n    db.create_all()\n    view = CustomModelView(Model2, db.session, filters=['first.test'])\n    admin.add_view(view)\n    client = app.test_client()\n    rv = client.get('/admin/model2/')\n    eq_(rv.status_code, 200)\n"""
    # f = """def get_signature_declarations(function:A, b=c):
    # root = ast.parse(function)
    #
    # declarations = {}
    #
    # function = 4
    #
    # for node in ast.walk(root):
    #     if isinstance(node, ast.arg):
    #         if node.annotation is None:
    #             offset = to_offsets(function, [(node.lineno, node.end_lineno, node.col_offset, node.end_col_offset, "arg")])
    #             # print(ast.dump(node), node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
    #             declarations[offset[-1]] = get_mentions(function, root, node.arg)
    #             print(declarations)
    # """
    print(f)
    declarations = get_declarations(f)

    for dec, mentions in declarations.items():
        print(f"{f[dec[0]: dec[1]]} {dec}", end=": ")
        for m in mentions:
            print(f"{f[m[0]: m[1]]} {m}", end="\t ")
        print()

