import ast
import sys
from collections import defaultdict
from pprint import pprint
import pygraphviz
from time import time_ns
from collections.abc import Iterable
import pandas as pd
from traceback import print_stack
"""
Possible strategies
1. Take Sourcetrail index and replace all occurrences in the source code with sourcetrail objects. This way all object will be disambiguated. Then, pass everything to ast parser. The problem is that I need to find field usages. This is probably not that hard, just start from the end of the line, and replace everything that is connected with a dot.
2. If a statement confitions on smth, add "conditions at" edge to every node
"""

# func = open(sys.argv[1]).read()

class AstGraphGenerator(object):

    def __init__(self, source):
        self.graph = defaultdict(lambda: [])
        self.source = source.split("\n")  # lines of the source code
        self.root = ast.parse(source)
        self.current_contition = []
        self.contition_status = []

    def get_name(self, node):
        return node.__class__.__name__ + str(hex(int(time_ns())))

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self):
        """Called if no explicit visitor function exists for a node."""
        edges = []
        for f_def_node in ast.iter_child_nodes(self.root):
            # print(ast.dump(f_def_node))
            # enter module
            if type(f_def_node) == ast.FunctionDef:
                edges.extend(self.parse(f_def_node))
                # edges.extend(self.parse_body(ast.iter_child_nodes(f_def_node)))
        # pprint(edges)

    # def get_call(self, node):
    #     if type(node) == ast.Attribute:
    #         # print(ast.dump(node))
    #         return self.get_call(node.value) + "_" + node.attr
    #     elif type(node) == ast.Name:
    #         return node.id

    def parse_in_context(self, cond_name, cond_stat, edges, body):
        # print(self.current_contition, self.contition_status)
        if isinstance(cond_name, str):
            cond_name = [cond_name]
            cond_stat = [cond_stat]
        for cn, cs in zip(cond_name, cond_stat):
            self.current_contition.append(cn)
            self.contition_status.append(cs)

        # print(body)
        edges.extend(self.parse_body(body))

        # print(self.current_contition, self.contition_status)
        for i in range(len(cond_name)):
            self.current_contition.pop(-1)
            self.contition_status.pop(-1)

    def parse_FunctionDef(self, node):
        edges, f_name = self.generic_parse(node, ["name", "args", "returns"])

        self.parse_in_context(f_name, "defined_in", edges, node.body)

        # name  = node.name
        # args =  (self.parse(n) for n in node.args.args)

        # edges = []
        # for a in args:
        #     edges.append({"src": a, "dst": name, "type":"def_arg"})
        return edges
        # return node.name, tuple(parse(n) for n in node.args.args)

    def parse_Assign(self, node):
        # edges = []

        # value, ext_edges = self.parse_operand(node.value)
        # edges.extend(ext_edges)
        
        # assign_name = "assign" + str(int(time_ns()))

        # dsts = (self.parse_name(t) for t in node.targets)
        # edges.append({"src": value, "dst": assign_name, "type": "assign_from"})
        # for dst in dsts:
        #     edges.append({"src": dst, "dst": assign_name, "type": "assign_to"})

        edges, assign_name = self.generic_parse(node, ["value", "targets"])

        for cond_name, cons_stat in zip(self.current_contition, self.contition_status):
            edges.append({"src": assign_name, "dst": cond_name, "type": "depends_on_" + cons_stat})

        return edges, assign_name
        # return tuple(parse_name(t) for t in node.targets), parse(node.value)

    def parse_AugAssign(self, node):
        return self.generic_parse(node, ["target", "op", "value"])

    def parse_ClassDef(self, node):
        edges, class_name = self.generic_parse(node, ["name"])

        self.parse_in_context(class_name, "True", edges, node.body)
        return edges, class_name

    def parse_ImportFrom(self, node):
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

    def parse_withitem(self, node):
        # print(node._fields)
        return self.generic_parse(node, ['context_expr', 'optional_vars'])

    def parse_alias(self, node):
        return self.generic_parse(node, ["name", "asname"])

    def parse_arg(self, node):
        return node.arg

    def parse_operand(self, node):
        # need to make sure upper level name is correct
        edges = []
        if isinstance(node, str):
            # fall here when parsing attributes, they are given as strings
            iter_ = node
        elif isinstance(node, int) or node is None:
            iter_ = str(node)
        else:
            iter_e = self.parse(node)
            if type(iter_e) == str:
                iter_ = iter_e
            elif type(iter_e) == tuple:
                ext_edges, name = iter_e
                edges.extend(ext_edges)
                iter_ = name
                # print(node, type(node), type(iter_e), iter_e)
                # print(ast.dump(node))
                # print(iter_e)
                # iter_ = iter_e[0]['dst']
            else:
                print(node)
                print(ast.dump(node))
                print(iter_e)
                print(type(iter_e))
                pprint(self.source)
                print(self.source[node.lineno-1].strip())
                raise Exception()

        return iter_, edges

    def parse_and_add_operand(self, node_name, operand, type, edges):

        operand_name, ext_edges = self.parse_operand(operand)
        edges.extend(ext_edges)

        edges.append({"src": operand_name, "dst": node_name, "type": type})

    def generic_parse(self, node, operands):
        # print("\n",ast.dump(node),type(node) )
        edges = []

        node_name = self.get_name(node)

        for operand in operands:
            if operand in ["body", "orelse", "finalbody"]:
                print(node)
                print(operand)
                print(node.__getattribute__(operand))
                self.parse_in_context(node_name, "operand", edges, node.__getattribute__(operand))
            else:
                operand_ = node.__getattribute__(operand)
                if operand_ is not None:
                    if isinstance(operand_, Iterable) and not isinstance(operand_, str):
                        for oper_ in operand_:
                            self.parse_and_add_operand(node_name, oper_, operand, edges)
                    else:
                        self.parse_and_add_operand(node_name, operand_, operand, edges)
        
        return edges, node_name

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
        # return self.generic_parse(node, ["test", "body", "orelse"])

    def parse_ExceptHandler(self, node):
        # not parsing "name" field
        # except handler is unique for every function
        return self.generic_parse(node, ["type"])

    def parse_Call(self, node):
        return self.generic_parse(node, ["func", "args"])
        # edges = []
        # # print("\t\t", ast.dump(node))
        # call_name = self.get_name(node)
        # # print(ast.dump(node.func))
        # self.parse_and_add_operand(call_name, node.func, "call_func", edges)
        # # f_name = self.parse(node.func)
        # # call_args = (self.parse(a) for a in node.args)
        # # edges.append({"src": f_name, "dst": call_name, "type": "call_func"})
        # for a in node.args:
        #     self.parse_and_add_operand(call_name, a, "call_arg", edges)
        #     # arg_name, ext_edges = self.parse_operand(a)
        #     # edges.extend(ext_edges)
        #     # edges.append({"src": arg_name, "dst": call_name, "type": "call_arg"})
        # return edges, call_name
        # # return get_call(node.func), tuple(parse(a) for a in node.args)

    def parse_name(self, node):
        edges = []
        # if type(node) == ast.Attribute:
        #     left, ext_edges = self.parse(node.value)
        #     right = node.attr
        #     return self.parse(node.value) + "___" + node.attr
        if type(node) == ast.Name:
            return str(node.id)
        elif type(node) == ast.NameConstant:
            return str(node.value)

    def parse_Attribute(self, node):
        return self.generic_parse(node, ["value", "attr"])

    def parse_Name(self, node):
        return self.parse_name(node)

    def parse_NameConstant(self, node):
        return self.parse_name(node)

    def parse_op_name(self, node):
        return node.__class__.__name__

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
    # def parse_name_const(self, node):
    #     return node.value

    def parse_Num(self, node):
        return str(node.n)

    def parse_Str(self, node):
        return node.s

    def parse_Bytes(self, node):
        return repr(node.s)

    def parse_If(self, node):
        # edges = []
        # cond_name, ext_edges = self.parse_operand(node.test)
        # edges.extend(ext_edges)
        edges, if_name = self.generic_parse(node, ["test"])

        self.parse_in_context(if_name, "True", edges, node.body)
        self.parse_in_context(if_name, "False", edges, node.orelse)

        return edges


    def parse_For(self, node):
        # edges = []

        # for_name = "for" + str(int(time_ns()))

        # iter_, iter_e = self.parse_operand(node.iter)
        # edges.extend(iter_e)
        # edges.append({"src": iter_, "dst": self.parse(node.target), "type": "iter"})

        edges, for_name = self.generic_parse(node, ["target", "iter"])
        
        self.parse_in_context(for_name, "for", edges, node.body)
        self.parse_in_context(for_name, "orelse", edges, node.orelse)
        
        return edges #, for_name
        
    def parse_Try(self, node):
        # edges = []
        # try_name = "try" + str(int(time_ns()))

        edges, try_name = self.generic_parse(node, [])

        self.parse_in_context(try_name, "try", edges, node.body)
        
        for h in node.handlers:
            
            handler_name, ext_edges = self.parse_operand(h)
            edges.extend(ext_edges)
            self.parse_in_context([try_name, handler_name], ["except", "handler"], edges, h.body)
        
        self.parse_in_context(try_name, "final", edges, node.finalbody)
        self.parse_in_context(try_name, "else", edges, node.orelse)
        
        return edges #, try_name   
        
    def parse_While(self, node):
        # edges = []
        # while_name = "while" + str(int(time_ns()))

        edges, while_name = self.generic_parse(node, [])
        
        cond_name, ext_edges = self.parse_operand(node.test)
        edges.extend(ext_edges)

        self.parse_in_context([while_name, cond_name], ["while", "True"], edges, node.body)
        
        return edges #, while_name
        
        
    def parse_Compare(self, node):
        return self.generic_parse(node, ["left", "ops", "comparators"])
        # edges = []
        # left = self.parse(node.left)
        # ops = (o.__class__.__name__ for o in node.ops)
        # comps = (self.parse(c) for c in node.comparators)
        # comp_name = "compare" + str(int(time_ns()))
        # edges.append({"src": left, "dst": comp_name, "type": "comp_left"})

        # for o in ops:
        #     edges.append({"src": o, "dst": comp_name, "type": "comp_op"})
        # for c in comps:
        #     edges.append({"src": c, "dst": comp_name, "type": "comp_right"})
        
        # return edges, comp_name
        
    def parse_BoolOp(self, node):
        return self.generic_parse(node, ["values", "op"])
        # edges = []
        # bool_op_name = node.op.__class__.__name__ + str(int(time_ns()))
        # for c in node.values:
        #     op_name, ext_edges = self.parse_operand(c)
        #     edges.extend(ext_edges)
        #     edges.append({"src": op_name, "dst": bool_op_name, "type": "bool_op"})
        # return edges, bool_op_name

    def parse_Expr(self, node):
        edges = []
        expr_name, ext_edges = self.parse_operand(node.value)
        edges.extend(ext_edges)
        
        for cond_name, cons_stat in zip(self.current_contition, self.contition_status):
            edges.append({"src": expr_name, "dst": cond_name, "type": "depends_on_" + cons_stat})
        return edges

    def parse_control_flow(self, node):
        edges = []
        call_name = "call" + str(int(time_ns()))
        edges.append({"src": node.__class__.__name__, "dst": call_name, "type": "control_flow"})

        for cond_name, cons_stat in zip(self.current_contition, self.contition_status):
            edges.append({"src": call_name, "dst": cond_name, "type": "depends_on_" + cons_stat})
        return edges, call_name

    def parse_Continue(self, node):
        return self.parse_control_flow(node)

    def parse_Break(self, node):
        return self.parse_control_flow(node)
    
    def parse_Pass(self, node):
        return self.parse_control_flow(node)

    def parse_Assert(self, node):
        return self.generic_parse(node, ["test", "msg"])

    # def parse_iterable(self, node, name):
    #     edges = []

    #     for e in node.elts:
    #         val, ext_edges = self.parse_operand(e)
    #         edges.extend(ext_edges)
    #         edges.append({"src": val, "dst": name, "type": "element_of"})
        
    #     return edges, name

    def parse_List(self, node):
        return self.generic_parse(node, ["elts"])
        # list_name = "list" + str(int(time_ns()))
        # return self.parse_iterable(node, list_name)

    def parse_Tuple(self, node):
        return self.generic_parse(node, ["elts"])
        # list_name = "tuple" + str(int(time_ns()))
        # return self.parse_iterable(node, list_name)


    def parse_Set(self, node):
        return self.generic_parse(node, ["elts"])
        # list_name = "set" + str(int(time_ns()))
        # return self.parse_iterable(node, list_name)

    def parse_Dict(self, node):
        return self.generic_parse(node, ["keys", "values"])
        # edges = []
        # code = str(int(time_ns()))
        # dict_name = "dict" + code
        # keys_name = "keys" + code
        # vals_name = "vals" + code

        # for k in node.keys:
        #     item_, ext_edges = self.parse_operand(k)
        #     edges.extend(ext_edges)
        #     edges.append({"src": item_, "dst": keys_name, "type": "element_of"})

        # for v in node.values:
        #     item_, ext_edges = self.parse_operand(v)
        #     edges.extend(ext_edges)
        #     edges.append({"src": item_, "dst": vals_name, "type": "element_of"})

        # edges.append({"src": keys_name, "dst": dict_name, "type": "keys"})
        # edges.append({"src": vals_name, "dst": dict_name, "type": "vals"})

        # return edges, dict_name

    def parse_UnaryOp(self, node):
        return self.generic_parse(node, ["operand", "op"])
        # edges = []

        # un_name = "unop" + str(int(time_ns()))

        # operand_, ext_edges = self.parse_operand(node.operand)
        # edges.extend(ext_edges)
        # edges.append({"src": operand_, "dst": un_name, "type": "operand"})
        # edges.append({"src": node.op.__class__.__name__, "dst": un_name, "type": "op"})

        # return edges, un_name


    def parse_BinOp(self, node):
        return self.generic_parse(node, ["left", "right", "op"])
        # edges = []

        # biop_name = "binop" + str(int(time_ns()))

        # left, ext_edges = self.parse_operand(node.left)
        # edges.extend(ext_edges)

        # right, ext_edges = self.parse_operand(node.left)
        # edges.extend(ext_edges)

        # edges.append({"src": left, "dst": biop_name, "type": "leftop"})
        # edges.append({"src": right, "dst": biop_name, "type": "rightop"})
        # edges.append({"src": node.op.__class__.__name__, "dst": biop_name, "type": "op"})
        # return edges, biop_name

    # def parse_comprehension_like(self, node):
    #     edges = []

    #     gen_name = type(node).__name__ + str(int(time_ns()))

    #     name = self.parse(node.elt)
    #     edges.append({"src": name, "dst": gen_name, "type": "element"})
    #     for g in node.generators:
    #         gen, ext_edges = self.parse_operand(g)
    #         edges.extend(ext_edges)
    #         edges.append({"src": gen, "dst": gen_name, "type": "generator"})

    #     return edges, gen_name

    def parse_GeneratorExp(self, node):
        return self.generic_parse(node, ["elt", "generators"])
        # return self.parse_comprehension_like(node)

    def parse_ListComp(self, node):
        return self.generic_parse(node, ["elt", "generators"])
        # return self.parse_comprehension_like(node)

    def parse_DictComp(self, node):
        return self.generic_parse(node, ["key", "value", "generators"])
        # edges = []
        # node_name = self.get_name(node)

        # self.parse_and_add_operand(node_name, node.key, "key", edges)
        # self.parse_and_add_operand(node_name, node.value, "value", edges)

        # for generator in node.generators:
        #     self.parse_and_add_operand(node_name, generator, "generator", edges)

        # return edges, node_name

    def parse_SetComp(self, node):
        return self.generic_parse(node, ["elt", "generators"])
        # edges = []
        # node_name = self.get_name(node)

        # self.parse_and_add_operand(node_name, node.elt, "elt", edges)

        # for generator in node.generators:
        #     self.parse_and_add_operand(node_name, generator, "generator", edges)

        # return edges, node_name

    def parse_Return(self, node):
        return self.generic_parse(node, ["value"])

    def parse_Raise(self, node):
        return self.generic_parse(node, ["exc", "cause"])

    def parse_YieldFrom(self, node):
        return self.generic_parse(node, ["value"])

    def parse_arguments(self, node):
        # have missing fields. example:
        #    arguments(args=[arg(arg='self', annotation=None), arg(arg='tqdm_cls', annotation=None), arg(arg='sleep_interval', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        return self.generic_parse(node, ["args"])

    def parse_comprehension(self, node):
        edges = []

        cph_name = "comprehension" + str(int(time_ns()))

        target = self.parse(node.target)
        edges.append({"src": target, "dst": cph_name, "type": "target"})

        iter_, ext_edges = self.parse_operand(node.iter)
        edges.extend(ext_edges)
        edges.append({"src": iter_, "dst": cph_name, "type": "iter"})

        for if_ in node.ifs:
            if_n, ext_edges = self.parse_operand(if_)
            edges.extend(ext_edges)
            edges.append({"src": if_n, "dst": cph_name, "type": "ifs"})

        return edges, cph_name


    def parse(self, node):
        n_type = type(node)
        method_name = "parse_" + n_type.__name__
        if hasattr(self, method_name):
            return self.__getattribute__(method_name)(node)
        else:
            print(type(node))
            print(ast.dump(node))
            return self.generic_parse(node, node._fields)
            raise Exception()
            # return [type(node)]
        
    def parse_body(self, nodes):
        edges = []
        for node in nodes:
            
            n_type = type(node)
            # if n_type in [ast.Expr, ast.FunctionDef, ast.Assign, ast.If, ast.Try, ast.For, ast.While, ast.Continue, ast.Pass, ast.Return]:
            # print_stack()
            # print("\n",ast.dump(node),type(node) )
            # print(ast.dump(node))
            s = self.parse(node)
            if isinstance(s, tuple):
                edges.extend(s[0])
            else:
                edges.extend(s)
            # else:
            #     print("skipped", type(node))
            #     pass
        return edges

f_bodies = pd.read_csv(sys.argv[1])
for ind, c in enumerate(f_bodies['content']):
    try:
        # print(c.strip())
        # if not isinstance(c, str): continue
        try:
            c.strip()
        except:
            print(c)
            continue
        g = AstGraphGenerator(c.strip())
        # print(ast.dump(ast.parse(func)))
        g.generic_visit()
        print("%d/%d" % (ind, len(f_bodies['content'])))
    except SyntaxError:
        print(c.strip())
# print(func)
# g = AstGraphGenerator(func.strip())
# g.generic_visit()
# pygraphviz.AGraph(g)
# for key, val in g.graph.items():
#     print(key, val, sep="\t")