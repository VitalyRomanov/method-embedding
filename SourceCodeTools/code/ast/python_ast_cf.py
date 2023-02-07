import ast
from copy import copy
from pprint import pprint
from time import time_ns
from collections.abc import Iterable
import pandas as pd

from SourceCodeTools.code.annotator_utils import to_offsets


class GNode:
    name = None
    type = None
    id = None

    def __init__(self, **kwargs):
        self.string = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other):
        if self.name == other.name and self.type == other.type:
            return True
        else:
            return False

    def __repr__(self):
        return self.__dict__.__repr__()

    def __hash__(self):
        return (self.name, self.type).__hash__()

    def setprop(self, key, value):
        setattr(self, key, value)


class AstGraphGenerator(object):

    def __init__(self, source, **kwargs):
        self.source = source.split("\n")  # lines of the source code
        self.full_source = source
        self.root = ast.parse(source)
        self.current_condition = []
        self.condition_status = []
        self.scope = []

    def get_source_from_ast_range(self, start_line, end_line, start_col, end_col):
        source = ""
        num_lines = end_line - start_line + 1
        if start_line == end_line:
            source += self.source[start_line - 1].encode("utf8")[start_col:end_col].decode(
                "utf8").strip()
        else:
            for ind, lineno in enumerate(range(start_line - 1, end_line)):
                if ind == 0:
                    source += self.source[lineno].encode("utf8")[start_col:].decode(
                        "utf8").strip()
                elif ind == num_lines - 1:
                    source += self.source[lineno].encode("utf8")[:end_col].decode(
                        "utf8").strip()
                else:
                    source += self.source[lineno].strip()

        return source

    def get_name(self, *, node=None, name=None, type=None, add_random_identifier=False):

        if node is not None:
            name = node.__class__.__name__ + "_" + str(hex(int(time_ns())))
            type = node.__class__.__name__
        else:
            if add_random_identifier:
                name += f"_{str(hex(int(time_ns())))}"

        if len(self.scope) > 0:
            return GNode(name=name, type=type, scope=copy(self.scope[-1]))
        else:
            return GNode(name=name, type=type)
        # return (node.__class__.__name__ + "_" + str(hex(int(time_ns()))), node.__class__.__name__)

    def get_edges(self, as_dataframe=False):
        edges = []
        edges.extend(self.parse(self.root)[0])
        # for f_def_node in ast.iter_child_nodes(self.root):
        #     if type(f_def_node) == ast.FunctionDef:
        #         edges.extend(self.parse(f_def_node))
        #         break  # to avoid going through nested definitions

        if not as_dataframe:
            return edges
        df = pd.DataFrame(edges)
        return df.astype({col: "Int32" for col in df.columns if col not in {"src", "dst", "type"}})

    def parse(self, node):
        n_type = type(node)
        method_name = "parse_" + n_type.__name__
        if hasattr(self, method_name):
            return self.__getattribute__(method_name)(node)
        else:
            # print(type(node))
            # print(ast.dump(node))
            # print(node._fields)
            # pprint(self.source)
            return self.parse_as_expression(node)
            # return self.generic_parse(node, node._fields)
            # raise Exception()
            # return [type(node)]

    def parse_body(self, nodes):
        edges = []
        last_node = None
        for node in nodes:
            s = self.parse(node)
            if isinstance(s, tuple):
                # some parsers return edges and names?
                edges.extend(s[0])

                if last_node is not None:
                    edges.append({"dst": s[1], "src": last_node, "type": "next"})
                    edges.append({"dst": last_node, "src": s[1], "type": "prev"})

                last_node = s[1]

                for cond_name, cond_stat in zip(self.current_condition[-1:], self.condition_status[-1:]):
                    edges.append({"scope": copy(self.scope[-1]), "src": last_node, "dst": cond_name, "type": "defined_in_" + cond_stat})
                    edges.append({"scope": copy(self.scope[-1]), "src": cond_name, "dst": last_node, "type": "defined_in_" + cond_stat + "_rev"})
                    # edges.append({"scope": copy(self.scope[-1]), "src": cond_name, "dst": last_node, "type": "execute_when_" + cond_stat})
            else:
                edges.extend(s)
        return edges

    def parse_in_context(self, cond_name, cond_stat, edges, body):
        if isinstance(cond_name, str):
            cond_name = [cond_name]
            cond_stat = [cond_stat]
        elif isinstance(cond_name, GNode):
            cond_name = [cond_name]
            cond_stat = [cond_stat]

        for cn, cs in zip(cond_name, cond_stat):
            self.current_condition.append(cn)
            self.condition_status.append(cs)

        edges.extend(self.parse_body(body))

        for i in range(len(cond_name)):
            self.current_condition.pop(-1)
            self.condition_status.pop(-1)

    def parse_as_expression(self, node, *args, **kwargs):
        offset = to_offsets(self.full_source,
                            [(node.lineno - 1, node.end_lineno - 1, node.col_offset, node.end_col_offset, "expression")],
                            as_bytes=True)
        offset, = offset
        line = self.full_source[offset[0]: offset[1]].replace("@","##at##")
        # name = GNode(name=line, type="Name")
        # expr = GNode(name="Expression" + "_" + str(hex(int(time_ns()))), type="mention")

        expr = GNode(name=f"{line}@{self.scope[-1].name}", type="mention")
        edges = []
        from nltk import RegexpTokenizer
        tok = RegexpTokenizer("\w+|\W")
        for t in tok.tokenize(line):
            edges.append(
                {"scope": copy(self.scope[-1]), "src": GNode(name=t, type="Name"), "dst": expr,
                 "type": "local_mention"},
                # {"scope": copy(self.scope[-1]), "src": self.scope[-1], "dst": mention_name, "type": "mention_scope"}
            )
        # edges = [
        #     {"scope": copy(self.scope[-1]), "src": name, "dst": expr, "type": "local_mention", "line": node.lineno - 1, "end_line": node.end_lineno - 1, "col_offset": node.col_offset, "end_col_offset": node.end_col_offset},
        # ]

        return edges, expr

    def parse_as_mention(self, name):
        mention_name = GNode(name=name + "@" + self.scope[-1].name, type="mention", scope=copy(self.scope[-1]))
        name = GNode(name=name, type="Name")
        # mention_name = (name + "@" + self.scope[-1], "mention")

        # edge from name to mention in a function
        edges = [
            {"scope": copy(self.scope[-1]), "src": name, "dst": mention_name, "type": "local_mention"},
            # {"scope": copy(self.scope[-1]), "src": self.scope[-1], "dst": mention_name, "type": "mention_scope"}
        ]
        return edges, mention_name

    def parse_operand(self, node):
        # need to make sure upper level name is correct; handle @keyword; type placeholder for sourcetrail nodes?
        edges = []
        if isinstance(node, str):
            # fall here when parsing attributes, they are given as strings; should attributes be parsed into subwords?
            if "@" in node:
                parts = node.split("@")
                node = GNode(name=parts[0], type=parts[1])
            else:
                node = GNode(name=node, type="Name")
            iter_ = node
        elif isinstance(node, int) or node is None:
            iter_ = GNode(name=str(node), type="ast_Literal")
            # iter_ = str(node)
        elif isinstance(node, GNode):
            iter_ = node
        else:
            iter_e = self.parse(node)
            if type(iter_e) == str:
                iter_ = iter_e
            elif isinstance(iter_e, GNode):
                iter_ = iter_e
            elif type(iter_e) == tuple:
                ext_edges, name = iter_e
                assert isinstance(name, GNode)
                edges.extend(ext_edges)
                iter_ = name
            else:
                # unexpected scenario
                print(node)
                print(ast.dump(node))
                print(iter_e)
                print(type(iter_e))
                pprint(self.source)
                print(self.source[node.lineno - 1].strip())
                raise Exception()

        return iter_, edges

    def parse_and_add_operand(self, node_name, operand, type, edges):

        operand_name, ext_edges = self.parse_operand(operand)
        edges.extend(ext_edges)

        if hasattr(operand, "lineno"):
            edges.append({"scope": copy(self.scope[-1]), "src": operand_name, "dst": node_name, "type": type, "line": operand.lineno-1, "end_line": operand.end_lineno-1, "col_offset": operand.col_offset, "end_col_offset": operand.end_col_offset})
        else:
            edges.append({"scope": copy(self.scope[-1]), "src": operand_name, "dst": node_name, "type": type})

        # if len(ext_edges) > 0:  # need this to avoid adding reverse edges to operation names and other highly connected nodes
        edges.append({"scope": copy(self.scope[-1]), "src": node_name, "dst": operand_name, "type": type + "_rev"})

    def generic_parse(self, node, operands, with_name=None, ensure_iterables=False):

        edges = []

        if with_name is None:
            node_name = self.get_name(node=node)
        else:
            node_name = with_name

        # if len(self.scope) > 0:
        #     edges.append({"scope": copy(self.scope[-1]), "src": node_name, "dst": self.scope[-1], "type": "mention_scope"})
        #     edges.append({"scope": copy(self.scope[-1]), "src": self.scope[-1], "dst": node_name, "type": "mention_scope_rev"})

        for operand in operands:
            if operand in ["body", "orelse", "finalbody"]:
                self.parse_in_context(node_name, "operand", edges, node.__getattribute__(operand))
            else:
                operand_ = node.__getattribute__(operand)
                if operand_ is not None:
                    if isinstance(operand_, Iterable) and not isinstance(operand_, str):
                        # TODO:
                        #  appears as leaf node if the iterable is empty. suggest adding an "EMPTY" element
                        for oper_ in operand_:
                            self.parse_and_add_operand(node_name, oper_, operand, edges)
                    else:
                        self.parse_and_add_operand(node_name, operand_, operand, edges)

        # TODO
        # need to identify the benefit of this node
        # maybe it is better to use node types in the graph
        # edges.append({"scope": copy(self.scope[-1]), "src": node.__class__.__name__, "dst": node_name, "type": "node_type"})

        return edges, node_name

    def parse_type_node(self, node):
        # node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
        if node.lineno == node.end_lineno:
            type_str = self.source[node.lineno][node.col_offset - 1: node.end_col_offset]
            # print(type_str)
        else:
            type_str = ""
            for ln in range(node.lineno - 1, node.end_lineno):
                if ln == node.lineno - 1:
                    type_str += self.source[ln][node.col_offset - 1:].strip()
                elif ln == node.end_lineno - 1:
                    type_str += self.source[ln][:node.end_col_offset].strip()
                else:
                    type_str += self.source[ln].strip()
        return type_str

    def parse_Module(self, node):
        edges, module_name = self.generic_parse(node, [])
        self.scope.append(module_name)
        self.parse_in_context(module_name, "module", edges, node.body)
        self.scope.pop(-1)
        return edges, module_name

    def parse_FunctionDef(self, node):
        # edges, f_name = self.generic_parse(node, ["name", "args", "returns", "decorator_list"])
        # edges, f_name = self.generic_parse(node, ["args", "returns", "decorator_list"])

        # need to creare function name before generic_parse so that the scope is set up correctly
        # scope is used to create local mentions of variable and function names
        fdef_node_name = self.get_name(node=node)
        self.scope.append(fdef_node_name)

        to_parse = []
        if len(node.args.args) > 0 or node.args.vararg is not None:
            to_parse.append("args")
        if len("decorator_list") > 0:
            to_parse.append("decorator_list")

        edges, f_name = self.generic_parse(node, to_parse, with_name=fdef_node_name)

        if node.returns is not None:
            # returns stores return type annotation
            # can contain quotes
            # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
            # https://www.python.org/dev/peps/pep-0484/#forward-references
            annotation_string = self.get_source_from_ast_range(
                node.returns.lineno, node.returns.end_lineno,
                node.returns.col_offset, node.returns.end_col_offset
            )
            annotation = GNode(name=annotation_string,
                               type="type_annotation")
            edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": f_name, "type": "returned_by", "line": node.returns.lineno - 1, "end_line": node.returns.end_lineno - 1, "col_offset": node.returns.col_offset, "end_col_offset": node.returns.end_col_offset})
            # do not use reverse edges for types, will result in leak from function to function
            # edges.append({"scope": copy(self.scope[-1]), "src": f_name, "dst": annotation, "type": 'returns'})

        # if node.returns:
        #     print(self.source[node.lineno -1]) # can get definition string here

        self.parse_in_context(f_name, "function", edges, node.body)

        self.scope.pop(-1)

        # func_name = GNode(name=node.name, type="Name")
        ext_edges, func_name = self.parse_as_mention(name=node.name)
        edges.extend(ext_edges)

        assert isinstance(node.name, str)
        edges.append({"scope": copy(self.scope[-1]), "src": f_name, "dst": func_name, "type": "function_name"})
        edges.append({"scope": copy(self.scope[-1]), "src": func_name, "dst": f_name, "type": "function_name_rev"})

        return edges, f_name

    def parse_AsyncFunctionDef(self, node):
        return self.parse_FunctionDef(node)

    def parse_ClassDef(self, node):

        edges, class_node_name = self.generic_parse(node, [])

        self.scope.append(class_node_name)
        self.parse_in_context(class_node_name, "class", edges, node.body)
        self.scope.pop(-1)

        ext_edges, cls_name = self.parse_as_mention(name=node.name)
        edges.extend(ext_edges)
        edges.append({"scope": copy(self.scope[-1]), "src": class_node_name, "dst": cls_name, "type": "class_name"})
        edges.append({"scope": copy(self.scope[-1]), "src": cls_name, "dst": class_node_name, "type": "class_name_rev"})

        return edges, class_node_name

    def parse_arg(self, node):
        # node.annotation stores type annotation
        # if node.annotation:
        #     print(self.source[node.lineno-1]) # can get definition string here
        #     print(node.arg)

        # # included mention
        name = self.get_name(node=node)
        edges, mention_name = self.parse_as_mention(node.arg)
        edges.append({"scope": copy(self.scope[-1]), "src": mention_name, "dst": name, "type": 'arg'})
        edges.append({"scope": copy(self.scope[-1]), "src": name, "dst": mention_name, "type": 'arg_rev'})
        # edges, name = self.generic_parse(node, ["arg"])
        if node.annotation is not None:
            # can contain quotes
            # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
            # https://www.python.org/dev/peps/pep-0484/#forward-references
            annotation_string = self.get_source_from_ast_range(
                node.annotation.lineno, node.annotation.end_lineno,
                node.annotation.col_offset, node.annotation.end_col_offset
            )
            annotation = GNode(name=annotation_string,
                               type="type_annotation")
            edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": name, "type": 'annotation_for', "line": node.annotation.lineno-1, "end_line": node.annotation.end_lineno-1, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_line": node.lineno-1, "var_end_line": node.end_lineno-1, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset})
            # do not use reverse edges for types, will result in leak from function to function
            # edges.append({"scope": copy(self.scope[-1]), "src": name, "dst": annotation, "type": 'annotation'})
        return edges, name
        # return self.generic_parse(node, ["arg", "annotation"])

    def parse_arguments(self, node):
        # have missing fields. example:
        #    arguments(args=[arg(arg='self', annotation=None), arg(arg='tqdm_cls', annotation=None), arg(arg='sleep_interval', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])

        # vararg constains type annotations
        return self.generic_parse(node, ["args", "vararg"])

    def parse_With(self, node):
        edges, with_name = self.generic_parse(node, ["items"])

        self.parse_in_context(with_name, "with", edges, node.body)

        return edges, with_name

    def parse_AsyncWith(self, node):
        return self.parse_With(node)

    def parse_withitem(self, node):
        return self.generic_parse(node, ['context_expr', 'optional_vars'])

    def parse_ExceptHandler(self, node):
        # have missing fields. example:
        # not parsing "name" field
        # except handler is unique for every function
        return self.generic_parse(node, ["type"])

    def parse_name(self, node):
        edges = []
        # if type(node) == ast.Attribute:
        #     left, ext_edges = self.parse(node.value)
        #     right = node.attr
        #     return self.parse(node.value) + "___" + node.attr
        if type(node) == ast.Name:
            return self.parse_as_mention(str(node.id))
        elif type(node) == ast.NameConstant:
            return GNode(name=str(node.value), type="NameConstant")

    def parse_Name(self, node):
        return self.parse_name(node)

    def parse_op_name(self, node):
        return GNode(name=node.__class__.__name__, type="Op")
        # return node.__class__.__name__

    def parse_Str(self, node):
        return self.generic_parse(node, [])
        # return node.s

    def parse_If(self, node):

        edges, if_name = self.generic_parse(node, ["test"])

        self.parse_in_context(if_name, "if_true", edges, node.body)
        self.parse_in_context(if_name, "if_false", edges, node.orelse)

        return edges, if_name

    def parse_For(self, node):

        edges, for_name = self.generic_parse(node, ["target", "iter"])
        
        self.parse_in_context(for_name, "for", edges, node.body)
        self.parse_in_context(for_name, "for_orelse", edges, node.orelse)
        
        return edges, for_name

    def parse_AsyncFor(self, node):
        return self.parse_For(node)
        
    def parse_Try(self, node):

        edges, try_name = self.generic_parse(node, [])

        self.parse_in_context(try_name, "try", edges, node.body)
        
        for h in node.handlers:
            
            handler_name, ext_edges = self.parse_operand(h)
            edges.extend(ext_edges)
            # self.parse_in_context([try_name, handler_name], ["try_except", "try_handler"], edges, h.body)
            self.parse_in_context([handler_name], ["try_handler"], edges, h.body)
            edges.append({
                "scope": copy(self.scope[-1]), "src": handler_name, "dst": try_name, "type": 'try_except'
            })
        
        self.parse_in_context(try_name, "try_final", edges, node.finalbody)
        self.parse_in_context(try_name, "try_else", edges, node.orelse)
        
        return edges, try_name
        
    def parse_While(self, node):

        edges, while_name = self.generic_parse(node, ["test"])
        
        # cond_name, ext_edges = self.parse_operand(node.test)
        # edges.extend(ext_edges)

        # self.parse_in_context([while_name, cond_name], ["while", "if_true"], edges, node.body)
        self.parse_in_context([while_name], ["while"], edges, node.body)
        
        return edges, while_name

    def parse_Expr(self, node):
        edges = []
        expr_name, ext_edges = self.parse_operand(node.value)
        edges.extend(ext_edges)
        
        # for cond_name, cons_stat in zip(self.current_condition, self.condition_status):
        #     edges.append({"scope": copy(self.scope[-1]), "src": expr_name, "dst": cond_name, "type": "depends_on_" + cons_stat})
        return edges, expr_name


if __name__ == "__main__":
    import sys
    f_bodies = pd.read_pickle(sys.argv[1])
    failed = 0
    for ind, c in enumerate(f_bodies['body_normalized']):
        if isinstance(c, str):
            # c = """def g():
            # yield 1"""
            try:
                g = AstGraphGenerator(c.lstrip())
            except SyntaxError as e:
                print(e)
                continue
            failed += 1
            edges = g.get_edges()
            # edges.to_csv(os.path.join(os.path.dirname(sys.argv[1]), "body_edges.csv"), mode="a", index=False, header=(ind==0))
            print("\r%d/%d" % (ind, len(f_bodies['body_normalized'])), end="")
        else:
            print("Skipped not a string")

    print(" " * 30, end="\r")
    print(failed, len(f_bodies['body_normalized']))
