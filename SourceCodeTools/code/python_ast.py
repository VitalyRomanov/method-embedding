import ast
from copy import copy
from enum import Enum
from pprint import pprint
from time import time_ns
from collections.abc import Iterable
import pandas as pd
# import os
from SourceCodeTools.code.data.identfier import IdentifierPool


class PythonSyntheticNodeTypes(Enum):  # TODO NOT USED
    NAME = 1  # "Name"
    MENTION = 2  # "mention"
    AST_LITERAL = 3  # "ast_Literal"
    TYPE_ANNOTATION = 4  # "type_annotation"
    NAME_CONSTANT = 5  # "NameConstant"
    CONSTANT = 6  # "Constant"
    OP = 7  # "Op"
    CTL_FLOW_INSTANCE = 8  # "CtlFlowInstance"
    CTL_FLOW = 9  # "CtlFlow"
    JOINED_STR = 10  # "JoinedStr"
    COMPREHENSION = 11  # "comprehension"
    KEYWORD_PROP = 12  # "#keyword#"
    ATTR_PROP = 13  # "#attr#"


class PythonSyntheticEdgeTypes:
    subword_instance = "subword_instance"
    next = "next"
    prev = "prev"
    # depends_on_ = "depends_on_"
    execute_when_ = "execute_when_"
    local_mention = "local_mention"
    mention_scope = "mention_scope"
    returned_by = "returned_by"
    # TODO
    #  make sure every place uses function_name and not fname
    fname = "function_name"
    # fname_for = "fname_for"
    annotation_for = "annotation_for"
    control_flow = "control_flow"


class PythonSharedNodes:
    annotation_types = {"type_annotation", "returned_by"}
    tokenizable_types = {"Name", "#attr#", "#keyword#"}
    python_token_types = {"Op", "Constant", "JoinedStr", "CtlFlow", "ast_Literal"}
    subword_types = {'subword'}

    subword_leaf_types = annotation_types | subword_types | python_token_types
    named_leaf_types = annotation_types | tokenizable_types | python_token_types
    tokenizable_types_and_annotations = annotation_types | tokenizable_types

    shared_node_types = annotation_types | subword_types | tokenizable_types | python_token_types

    # leaf_types = {'subword', "Op", "Constant", "JoinedStr", "CtlFlow", "ast_Literal", "Name", "type_annotation", "returned_by"}
    # shared_node_types = {'subword', "Op", "Constant", "JoinedStr", "CtlFlow", "ast_Literal", "Name", "type_annotation", "returned_by", "#attr#", "#keyword#"}

    @classmethod
    def is_shared(cls, node):
        # nodes that are of stared type
        # nodes that are subwords of keyword arguments
        return cls.is_shared_name_type(node.name, node.type)

    @classmethod
    def is_shared_name_type(cls, name, type):
        if type in cls.shared_node_types or \
                (type == "subword_instance" and "0x" not in name):
            return True
        return False


class GNode:
    # name = None
    # type = None
    # id = None

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

    def __init__(self, source):
        self.source = source.split("\n")  # lines of the source code
        self.root = ast.parse(source)
        self.current_condition = []
        self.condition_status = []
        self.scope = []

        self._identifier_pool = IdentifierPool()

    def get_source_from_ast_range(self, start_line, end_line, start_col, end_col, strip=True):
        source = ""
        num_lines = end_line - start_line + 1
        if start_line == end_line:
            section = self.source[start_line - 1].encode("utf8")[start_col:end_col].decode(
                "utf8")
            source += section.strip() if strip else section + "\n"
        else:
            for ind, lineno in enumerate(range(start_line - 1, end_line)):
                if ind == 0:
                    section = self.source[lineno].encode("utf8")[start_col:].decode(
                        "utf8")
                    source += section.strip() if strip else section + "\n"
                elif ind == num_lines - 1:
                    section = self.source[lineno].encode("utf8")[:end_col].decode(
                        "utf8")
                    source += section.strip() if strip else section + "\n"
                else:
                    section = self.source[lineno]
                    source += section.strip() if strip else section + "\n"

        return source.rstrip()

    def get_name(self, *, node=None, name=None, type=None, add_random_identifier=False):

        random_identifier = self._identifier_pool.get_new_identifier()

        if node is not None:
            name = f"{node.__class__.__name__}_{random_identifier}"
            type = node.__class__.__name__
        else:
            if add_random_identifier:
                name = f"{name}_{random_identifier}"

        if hasattr(node, "lineno"):
            node_string = self.get_source_from_ast_range(node.lineno, node.end_lineno, node.col_offset, node.end_col_offset, strip=False)
            # if "\n" in node_string:
            #     node_string = None
        else:
            node_string = None

        if len(self.scope) > 0:
            return GNode(name=name, type=type, scope=copy(self.scope[-1]), string=node_string)
        else:
            return GNode(name=name, type=type, string=node_string)
        # return (node.__class__.__name__ + "_" + str(hex(int(time_ns()))), node.__class__.__name__)

    def get_edges(self, as_dataframe=True):
        edges = []
        for f_def_node in ast.iter_child_nodes(self.root):
            if type(f_def_node) == ast.FunctionDef:
                edges.extend(self.parse(f_def_node))
                break  # to avoid going through nested definitions

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
            print(type(node))
            print(ast.dump(node))
            print(node._fields)
            pprint(self.source)
            return self.generic_parse(node, node._fields)
            # raise Exception()
            # return [type(node)]

    def parse_body(self, nodes):
        edges = []
        last_node = None
        for node in nodes:
            s = self.parse(node)
            if isinstance(s, tuple):
                if s[1].type == "Constant":  # this happens when processing docstring, as a result a lot of nodes are connected to the node Constant_
                    continue                    # in general, constant node has no affect as a body expression, can skip
                # some parsers return edges and names?
                edges.extend(s[0])

                if last_node is not None:
                    edges.append({"dst": s[1], "src": last_node, "type": "next", "scope": copy(self.scope[-1])})
                    edges.append({"dst": last_node, "src": s[1], "type": "prev", "scope": copy(self.scope[-1])})

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

    def parse_Assign(self, node):

        edges, assign_name = self.generic_parse(node, ["value", "targets"])

        # for cond_name, cons_stat in zip(self.current_condition, self.condition_status):
        #     edges.append({"scope": copy(self.scope[-1]), "src": assign_name, "dst": cond_name, "type": "depends_on_" + cons_stat})

        return edges, assign_name

    def parse_AugAssign(self, node):
        return self.generic_parse(node, ["target", "op", "value"])

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

    def parse_ImportFrom(self, node):
        # # similar issues as with parsing alias, module name is parsed as a long chunk
        # # print(ast.dump(node))
        # edges, name = self.generic_parse(node, ["names"])
        # # if node.module:
        # #     name_from, edges_from = self.parse_operand(ast.parse(node.module).body[0].value)
        # #     edges.extend(edges_from)
        # #     edges.append({"scope": copy(self.scope[-1]), "src": name_from, "dst": name, "type": "module"})
        # return edges, name
        return self.generic_parse(node, ["module", "names"])

    def parse_Import(self, node):
        return self.generic_parse(node, ["names"])

    def parse_Delete(self, node):
        return self.generic_parse(node, ["targets"])

    def parse_Global(self, node):
        return self.generic_parse(node, ["names"])

    def parse_Nonlocal(self, node):
        return self.generic_parse(node, ["names"])

    def parse_With(self, node):
        edges, with_name = self.generic_parse(node, ["items"])

        self.parse_in_context(with_name, "with", edges, node.body)

        return edges, with_name

    def parse_AsyncWith(self, node):
        return self.parse_With(node)

    def parse_withitem(self, node):
        return self.generic_parse(node, ['context_expr', 'optional_vars'])

    def parse_alias(self, node):
        # # TODO
        # # aliases should be handled by sourcetrail. here i am trying to assign alias to a
        # # local mention of the module. maybe i should simply ignore aliases altogether
        #
        # name = self.get_name(node)
        # edges = []
        # # name, edges = self.parse_operand(ast.parse(node.name).body[0].value) # <- this was the functional line
        # # # if node.asname:
        # # #     edges.append({"scope": copy(self.scope[-1]), "src": name, "dst": node.asname, "type": "alias"})
        # return edges, name
        return self.generic_parse(node, ["name", "asname"])

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
            mention_name = GNode(name=node.arg + "@" + self.scope[-1].name, type="mention", scope=copy(self.scope[-1]))
            edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": mention_name, "type": 'annotation_for', "line": node.annotation.lineno-1, "end_line": node.annotation.end_lineno-1, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_line": node.lineno-1, "var_end_line": node.end_lineno-1, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset})
            # edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": name, "type": 'annotation_for', "line": node.annotation.lineno-1, "end_line": node.annotation.end_lineno-1, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_line": node.lineno-1, "var_end_line": node.end_lineno-1, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset})
            # # do not use reverse edges for types, will result in leak from function to function
            # # edges.append({"scope": copy(self.scope[-1]), "src": name, "dst": annotation, "type": 'annotation'})
        return edges, name
        # return self.generic_parse(node, ["arg", "annotation"])

    def parse_AnnAssign(self, node):
        # stores annotation information for variables
        #
        # paths: List[Path] = []
        # AnnAssign(target=Name(id='paths', ctx=Store()), annotation=Subscript(value=Name(id='List', ctx=Load()),
        #           slice=Index(value=Name(id='Path', ctx=Load())),
        #           ctx=Load()), value=List(elts=[], ctx=Load()), simple=1)

        # if node.annotation:
        #     print(self.source[node.lineno-1]) # can get definition string here

        # can contain quotes
        # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
        # https://www.python.org/dev/peps/pep-0484/#forward-references
        annotation_string = self.get_source_from_ast_range(
            node.annotation.lineno, node.annotation.end_lineno,
            node.annotation.col_offset, node.annotation.end_col_offset
        )
        annotation = GNode(name=annotation_string,
                           type="type_annotation")
        edges, name = self.generic_parse(node, ["target"])
        try:
            mention_name = GNode(name=node.target.id + "@" + self.scope[-1].name, type="mention", scope=copy(self.scope[-1]))
            edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": mention_name, "type": 'annotation_for', "line": node.annotation.lineno-1, "end_line": node.annotation.end_lineno-1, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_line": node.lineno-1, "var_end_line": node.end_lineno-1, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset})
        except Exception as e:
            edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": name, "type": 'annotation_for', "line": node.annotation.lineno-1, "end_line": node.annotation.end_lineno-1, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_line": node.lineno-1, "var_end_line": node.end_lineno-1, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset})
            # print(e)  # don't know how I should parse this "Attribute(value=Name(id='self', ctx=Load()), attr='srctrlrpl_1631733463030025000@#attr#', ctx=Store())"
        # edges.append({"scope": copy(self.scope[-1]), "src": annotation, "dst": name, "type": 'annotation_for', "line": node.annotation.lineno-1, "end_line": node.annotation.end_lineno-1, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_line": node.lineno-1, "var_end_line": node.end_lineno-1, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset})
        # # do not use reverse edges for types, will result in leak from function to function
        # # edges.append({"scope": copy(self.scope[-1]), "src": name, "dst": annotation, "type": 'annotation'})
        return edges, name
        # return self.generic_parse(node, ["target", "annotation"])

    def parse_Subscript(self, node):
        return self.generic_parse(node, ["value", "slice"])

    def parse_Slice(self, node):
        return self.generic_parse(node, ["lower", "upper", "step"])
    
    def parse_ExtSlice(self, node):
        return self.generic_parse(node, ["dims"])

    def parse_Index(self, node):
        return self.generic_parse(node, ["value"])

    def parse_Lambda(self, node):
        # this is too ambiguous
        edges, lmb_name = self.generic_parse(node, [])
        self.parse_and_add_operand(lmb_name, node.body, "lambda", edges)

        return edges, lmb_name

    def parse_Starred(self, node):
        return self.generic_parse(node, ["value"])

    def parse_Yield(self, node):
        return self.generic_parse(node, ["value"])

    def parse_IfExp(self, node):
        edges, ifexp_name = self.generic_parse(node, ["test"])
        self.parse_and_add_operand(ifexp_name, node.body, "body", edges)
        self.parse_and_add_operand(ifexp_name, node.orelse, "orelse", edges)
        return edges, ifexp_name

    def parse_ExceptHandler(self, node):
        # have missing fields. example:
        # not parsing "name" field
        # except handler is unique for every function
        return self.generic_parse(node, ["type"])

    def parse_Call(self, node):
        return self.generic_parse(node, ["func", "args", "keywords"])
        # edges = []
        # # print("\t\t", ast.dump(node))
        # call_name = self.get_name(node)
        # # print(ast.dump(node.func))
        # self.parse_and_add_operand(call_name, node.func, "call_func", edges)
        # # f_name = self.parse(node.func)
        # # call_args = (self.parse(a) for a in node.args)
        # # edges.append({"scope": copy(self.scope[-1]), "src": f_name, "dst": call_name, "type": "call_func"})
        # for a in node.args:
        #     self.parse_and_add_operand(call_name, a, "call_arg", edges)
        #     # arg_name, ext_edges = self.parse_operand(a)
        #     # edges.extend(ext_edges)
        #     # edges.append({"scope": copy(self.scope[-1]), "src": arg_name, "dst": call_name, "type": "call_arg"})
        # return edges, call_name
        # # return get_call(node.func), tuple(parse(a) for a in node.args)

    def parse_keyword(self, node):
        # change arg name so that it does not mix with variable names
        if isinstance(node.arg, str):
            node.arg += "@#keyword#"
            return self.generic_parse(node, ["arg", "value"])
        else:
            return self.generic_parse(node, ["value"])

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

    def parse_Attribute(self, node):
        if node.attr is not None:
            node.attr += "@#attr#"
        return self.generic_parse(node, ["value", "attr"])

    def parse_Name(self, node):
        return self.parse_name(node)

    def parse_NameConstant(self, node):
        return self.parse_name(node)

    def parse_Constant(self, node):
        # TODO
        # decide whether this name should be unique or not
        name = GNode(name="Constant_", type="Constant")
        # name = "Constant_"
        # if node.kind is not None:
        #     name += ""
        return name

    def parse_op_name(self, node):
        return GNode(name=node.__class__.__name__, type="Op")
        # return node.__class__.__name__

    def parse_And(self, node):
        return self.parse_op_name(node)
    
    def parse_Or(self, node):
        return self.parse_op_name(node)

    def parse_Not(self, node):
        return self.parse_op_name(node)

    def parse_Is(self, node):
        return self.parse_op_name(node)

    def parse_Gt(self, node):
        return self.parse_op_name(node)

    def parse_Lt(self, node):
        return self.parse_op_name(node)

    def parse_GtE(self, node):
        return self.parse_op_name(node)

    def parse_LtE(self, node):
        return self.parse_op_name(node)

    def parse_Add(self, node):
        return self.parse_op_name(node)

    def parse_Mod(self, node):
        return self.parse_op_name(node)

    def parse_Sub(self, node):
        return self.parse_op_name(node)

    def parse_UAdd(self, node):
        return self.parse_op_name(node)

    def parse_USub(self, node):
        return self.parse_op_name(node)

    def parse_Div(self, node):
        return self.parse_op_name(node)

    def parse_Mult(self, node):
        return self.parse_op_name(node)

    def parse_MatMult(self, node):
        return self.parse_op_name(node)

    def parse_Pow(self, node):
        return self.parse_op_name(node)

    def parse_FloorDiv(self, node):
        return self.parse_op_name(node)
    
    def parse_RShift(self, node):
        return self.parse_op_name(node)

    def parse_LShift(self, node):
        return self.parse_op_name(node)

    def parse_BitXor(self, node):
        return self.parse_op_name(node)

    def parse_BitAnd(self, node):
        return self.parse_op_name(node)

    def parse_BitOr(self, node):
        return self.parse_op_name(node)

    def parse_IsNot(self, node):
        return self.parse_op_name(node)

    def parse_NotIn(self, node):
        return self.parse_op_name(node)

    def parse_In(self, node):
        return self.parse_op_name(node)

    def parse_Invert(self, node):
        return self.parse_op_name(node)

    def parse_Eq(self, node):
        return self.parse_op_name(node)

    def parse_NotEq(self, node):
        return self.parse_op_name(node)

    def parse_Ellipsis(self, node):
        return self.parse_op_name(node)

    def parse_Num(self, node):
        return str(node.n)

    def parse_Str(self, node):
        return self.generic_parse(node, [])
        # return node.s

    def parse_Bytes(self, node):
        return repr(node.s)

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
            self.parse_in_context([try_name, handler_name], ["try_except", "try_handler"], edges, h.body)
        
        self.parse_in_context(try_name, "try_final", edges, node.finalbody)
        self.parse_in_context(try_name, "try_else", edges, node.orelse)
        
        return edges, try_name
        
    def parse_While(self, node):

        edges, while_name = self.generic_parse(node, [])
        
        cond_name, ext_edges = self.parse_operand(node.test)
        edges.extend(ext_edges)

        self.parse_in_context([while_name, cond_name], ["while", "if_true"], edges, node.body)
        
        return edges, while_name

    def parse_Compare(self, node):
        return self.generic_parse(node, ["left", "ops", "comparators"])

    def parse_BoolOp(self, node):
        return self.generic_parse(node, ["values", "op"])

    def parse_Expr(self, node):
        edges = []
        expr_name, ext_edges = self.parse_operand(node.value)
        edges.extend(ext_edges)
        
        # for cond_name, cons_stat in zip(self.current_condition, self.condition_status):
        #     edges.append({"scope": copy(self.scope[-1]), "src": expr_name, "dst": cond_name, "type": "depends_on_" + cons_stat})
        return edges, expr_name

    def parse_control_flow(self, node):
        edges = []
        ctrlflow_name = self.get_name(name="ctrl_flow", type="CtlFlowInstance", add_random_identifier=True)
        # ctrlflow_name = GNode(name="ctrl_flow_" + str(hex(int(time_ns()))), type="CtlFlowInstance")
        # ctrlflow_name = "ctrl_flow_" + str(int(time_ns()))
        edges.append({"scope": copy(self.scope[-1]), "src": GNode(name=node.__class__.__name__, type="CtlFlow"), "dst": ctrlflow_name, "type": "control_flow"})

        # for cond_name, cons_stat in zip(self.current_condition, self.condition_status):
        #     edges.append({"scope": copy(self.scope[-1]), "src": call_name, "dst": cond_name, "type": "depends_on_" + cons_stat})
        return edges, ctrlflow_name

    def parse_Continue(self, node):
        return self.parse_control_flow(node)

    def parse_Break(self, node):
        return self.parse_control_flow(node)
    
    def parse_Pass(self, node):
        return self.parse_control_flow(node)

    def parse_Assert(self, node):
        return self.generic_parse(node, ["test", "msg"])

    def parse_List(self, node):
        return self.generic_parse(node, ["elts"], ensure_iterables=True)

    def parse_Tuple(self, node):
        return self.generic_parse(node, ["elts"], ensure_iterables=True)

    def parse_Set(self, node):
        return self.generic_parse(node, ["elts"], ensure_iterables=True)

    def parse_Dict(self, node):
        return self.generic_parse(node, ["keys", "values"], ensure_iterables=True)

    def parse_UnaryOp(self, node):
        return self.generic_parse(node, ["operand", "op"])

    def parse_BinOp(self, node):
        return self.generic_parse(node, ["left", "right", "op"])

    def parse_Await(self, node):
        return self.generic_parse(node, ["value"])

    def parse_JoinedStr(self, node):
        joinedstr_name = GNode(name="JoinedStr_", type="JoinedStr")
        return [], joinedstr_name
        # return self.generic_parse(node, [])
        # return self.generic_parse(node, ["values"])

    def parse_FormattedValue(self, node):
        # have missing fields. example:
        # FormattedValue(value=Subscript(value=Name(id='args', ctx=Load()), slice=Index(value=Num(n=0)), ctx=Load()),conversion=-1, format_spec=None)
        # 'conversion', 'format_spec' not parsed
        return self.generic_parse(node, ["value"])

    def parse_GeneratorExp(self, node):
        return self.generic_parse(node, ["elt", "generators"])

    def parse_ListComp(self, node):
        return self.generic_parse(node, ["elt", "generators"])

    def parse_DictComp(self, node):
        return self.generic_parse(node, ["key", "value", "generators"])

    def parse_SetComp(self, node):
        return self.generic_parse(node, ["elt", "generators"])

    def parse_Return(self, node):
        return self.generic_parse(node, ["value"])

    def parse_Raise(self, node):
        return self.generic_parse(node, ["exc", "cause"])

    def parse_YieldFrom(self, node):
        return self.generic_parse(node, ["value"])

    def parse_arguments(self, node):
        # have missing fields. example:
        #    arguments(args=[arg(arg='self', annotation=None), arg(arg='tqdm_cls', annotation=None), arg(arg='sleep_interval', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])

        # vararg constains type annotations
        return self.generic_parse(node, ["args", "vararg"]) # kwarg, kwonlyargs, posonlyargs???

    def parse_comprehension(self, node):
        edges = []

        cph_name = self.get_name(name="comprehension", type="comprehension", add_random_identifier=True)
        # cph_name = GNode(name="comprehension_" + str(hex(int(time_ns()))), type="comprehension")

        # if len(self.scope) > 0:
        #     edges.append({"scope": copy(self.scope[-1]), "src": cph_name, "dst": self.scope[-1], "type": "mention_scope"})
        #     edges.append({"scope": copy(self.scope[-1]), "src": self.scope[-1], "dst": cph_name, "type": "mention_scope_rev"})

        target, ext_edges = self.parse_operand(node.target)
        edges.extend(ext_edges)
        if hasattr(node.target, "lineno"):
            edges.append({"scope": copy(self.scope[-1]), "src": target, "dst": cph_name, "type": "target", "line": node.target.lineno-1, "end_line": node.target.end_lineno-1, "col_offset": node.target.col_offset, "end_col_offset": node.target.end_col_offset})
        else:
            edges.append({"scope": copy(self.scope[-1]), "src": target, "dst": cph_name, "type": "target"})
        # if len(ext_edges) > 0:
        edges.append({"scope": copy(self.scope[-1]), "src": cph_name, "dst": target, "type": "target_for"})

        iter_, ext_edges = self.parse_operand(node.iter)
        edges.extend(ext_edges)
        if hasattr(node.iter, "lineno"):
            edges.append({"scope": copy(self.scope[-1]), "src": iter_, "dst": cph_name, "type": "iter", "line": node.iter.lineno-1, "end_line": node.iter.end_lineno-1, "col_offset": node.iter.col_offset, "end_col_offset": node.iter.end_col_offset})
        else:
            edges.append({"scope": copy(self.scope[-1]), "src": iter_, "dst": cph_name, "type": "iter"})
        # if len(ext_edges) > 0:
        edges.append({"scope": copy(self.scope[-1]), "src": cph_name, "dst": iter_, "type": "iter_for"})

        for if_ in node.ifs:
            if_n, ext_edges = self.parse_operand(if_)
            edges.extend(ext_edges)
            edges.append({"scope": copy(self.scope[-1]), "src": if_n, "dst": cph_name, "type": "ifs"})
            # if len(ext_edges) > 0:
            edges.append({"scope": copy(self.scope[-1]), "src": cph_name, "dst": if_n, "type": "ifs_rev"})

        return edges, cph_name

if __name__ == "__main__":
    import sys
    f_bodies = pd.read_csv(sys.argv[1])
    failed = 0
    for ind, c in enumerate(f_bodies['body_normalized']):
        if isinstance(c, str):
            c = """def g():
            yield 1"""
            try:
                g = AstGraphGenerator(c.lstrip())
            except SyntaxError as e:
                print(e)
                continue
            failed += 1
            edges = g.get_edges()
            # edges.to_csv(os.path.join(os.path.dirname(sys.argv[1]), "body_edges.csv"), mode="a", index=False, header=(ind==0))
            print("\r%d/%d" % (ind, len(f_bodies['normalized_body'])), end="")
        else:
            print("Skipped not a string")

    print(" " * 30, end="\r")
    print(failed, len(f_bodies['normalized_body']))
