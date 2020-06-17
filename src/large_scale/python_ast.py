import ast
import sys
from collections import defaultdict
from pprint import pprint
import pygraphviz
from time import time_ns
from collections.abc import Iterable
import pandas as pd

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

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self):
        """Called if no explicit visitor function exists for a node."""
        edges = []
        for f_def_node in ast.iter_child_nodes(self.root):
            # enter module
            if type(f_def_node) == ast.FunctionDef:
                edges.extend(self.parse(f_def_node))
                edges.extend(self.parse_body(ast.iter_child_nodes(f_def_node)))
        # pprint(edges)

    def get_call(self, node):
        if type(node) == ast.Attribute:
            return self.get_call(node.value) + "_" + node.attr
        elif type(node) == ast.Name:
            return node.id

    def parse_in_context(self, cond_name, cond_stat, edges, body):
        # print(self.current_contition, self.contition_status)
        if isinstance(cond_name, str):
            cond_name = [cond_name]
            cond_stat = [cond_stat]
        for cn, cs in zip(cond_name, cond_stat):
            self.current_contition.append(cn)
            self.contition_status.append(cs)

        edges.extend(self.parse_body(body))

        # print(self.current_contition, self.contition_status)
        for i in range(len(cond_name)):
            self.current_contition.pop(-1)
            self.contition_status.pop(-1)

    def parse_FunctionDef(self, node):
        name  = node.name
        args =  (self.parse(n) for n in node.args.args)

        edges = []
        for a in args:
            edges.append({"src": a, "dst": name, "type":"def_arg"})
        return edges
        # return node.name, tuple(parse(n) for n in node.args.args)

    def parse_Assign(self, node):
        edges = []

        value, ext_edges = self.parse_operand(node.value)
        edges.extend(ext_edges)
        
        assign_name = "assign" + str(int(time_ns()))

        dsts = (self.parse_name(t) for t in node.targets)
        edges.append({"src": value, "dst": assign_name, "type": "assign_from"})
        for dst in dsts:
            edges.append({"src": dst, "dst": assign_name, "type": "assign_to"})

        for cond_name, cons_stat in zip(self.current_contition, self.contition_status):
            edges.append({"src": assign_name, "dst": cond_name, "type": "depends_on_" + cons_stat})
        return edges, assign_name
        # return tuple(parse_name(t) for t in node.targets), parse(node.value)

    def parse_arg(self, node):
        return node.arg

    def parse_operand(self, node):
        # need to make sure upper level name is correct
        edges = []
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
            print(ast.dump(node))
            print(iter_e)
            print(self.source[node.lineno-1].strip())
            raise Exception()

        return iter_, edges

    def parse_Call(self, node):
        edges = []
        # print("\t\t", ast.dump(node))
        call_name = "call" + str(int(time_ns()))
        f_name = self.get_call(node.func)
        # call_args = (self.parse(a) for a in node.args)
        edges.append({"src": f_name, "dst": call_name, "type": "call_func"})
        for a in node.args:
            arg_name, ext_edges = self.parse_operand(a)
            edges.extend(ext_edges)
            edges.append({"src": arg_name, "dst": call_name, "type": "call_arg"})
        return edges, call_name
        # return get_call(node.func), tuple(parse(a) for a in node.args)

    def parse_name(self, node):
        if type(node) == ast.Attribute:
            return self.parse_name(node.value) + "___" + node.attr
        elif type(node) == ast.Name:
            return str(node.id)
        elif type(node) == ast.NameConstant:
            return str(node.value)

    def parse_Attribute(self, node):
        return self.parse_name(node)

    def parse_Name(self, node):
        return self.parse_name(node)

    def parse_NameConstant(self, node):
        return self.parse_name(node)

    # def parse_name_const(self, node):
    #     return node.value

    def parse_Num(self, node):
        return str(node.n)

    def parse_Str(self, node):
        return node.s

    def parse_If(self, node):
        edges = []
        cond_name, ext_edges = self.parse_operand(node.test)
        edges.extend(ext_edges)

        self.parse_in_context(cond_name, "True", edges, node.body)
        self.parse_in_context(cond_name, "False", edges, node.orelse)

        return edges


    def parse_For(self, node):
        edges = []

        for_name = "for" + str(int(time_ns()))

        iter_, iter_e = self.parse_operand(node.iter)
        edges.extend(iter_e)
        edges.append({"src": iter_, "dst": self.parse(node.target), "type": "iter"})
        
        self.parse_in_context(for_name, "for", edges, node.body)
        self.parse_in_context(for_name, "orelse", edges, node.orelse)
        
        return edges #, for_name
        
    def parse_Try(self, node):
        edges = []
        try_name = "try" + str(int(time_ns()))

        self.parse_in_context(try_name, "try", edges, node.body)
        
        for h in node.handlers:
            
            handler_name = "None" if not h.type else self.parse(h.type)
            self.parse_in_context(try_name, "except_"+handler_name, edges, h.body)
        
        self.parse_in_context(try_name, "final", edges, node.finalbody)
        
        self.parse_in_context(try_name, "else", edges, node.orelse)
        
        return edges #, try_name   
        
    def parse_While(self, node):
        edges = []

        while_name = "while" + str(int(time_ns()))
        
        cond_name, ext_edges = self.parse_operand(node.test)
        edges.extend(ext_edges)

        self.parse_in_context([while_name, cond_name], ["while", "True"], edges, node.body)
        
        return edges #, while_name
        
        
    def parse_Compare(self, node):
        edges = []
        left = self.parse(node.left)
        ops = (o.__class__.__name__ for o in node.ops)
        comps = (self.parse(c) for c in node.comparators)
        comp_name = "compare" + str(int(time_ns()))
        edges.append({"src": left, "dst": comp_name, "type": "comp_left"})

        for o in ops:
            edges.append({"src": o, "dst": comp_name, "type": "comp_op"})
        for c in comps:
            edges.append({"src": c, "dst": comp_name, "type": "comp_right"})
        
        return edges, comp_name
        
    def parse_BoolOp(self, node):
        edges = []
        bool_op_name = node.op.__class__.__name__ + str(int(time_ns()))
        for c in node.values:
            op_name, ext_edges = self.parse_operand(c)
            edges.extend(ext_edges)
            edges.append({"src": op_name, "dst": bool_op_name, "type": "bool_op"})
        return edges, bool_op_name

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
    
    def parse_Pass(self, node):
        return self.parse_control_flow(node)

    def parse_iterable(self, node, name):
        edges = []

        for e in node.elts:
            val, ext_edges = self.parse_operand(e)
            edges.extend(ext_edges)
            edges.append({"src": val, "dst": name, "type": "element_of"})
        
        return edges, name

    def parse_List(self, node):
        list_name = "list" + str(int(time_ns()))
        return self.parse_iterable(node, list_name)

    def parse_Tuple(self, node):
        list_name = "tuple" + str(int(time_ns()))
        return self.parse_iterable(node, list_name)


    def parse_Set(self, node):
        list_name = "set" + str(int(time_ns()))
        return self.parse_iterable(node, list_name)

    def parse_Dict(self, node):
        edges = []
        code = str(int(time_ns()))
        dict_name = "dict" + code
        keys_name = "keys" + code
        vals_name = "vals" + code

        for k in node.keys:
            item_, ext_edges = self.parse_operand(k)
            edges.extend(ext_edges)
            edges.append({"src": item_, "dst": keys_name, "type": "element_of"})

        for v in node.values:
            item_, ext_edges = self.parse_operand(v)
            edges.extend(ext_edges)
            edges.append({"src": item_, "dst": vals_name, "type": "element_of"})

        edges.append({"src": keys_name, "dst": dict_name, "type": "keys"})
        edges.append({"src": vals_name, "dst": dict_name, "type": "vals"})

        return edges, dict_name

    def parse_UnaryOp(self, node):

        edges = []

        un_name = "unop" + str(int(time_ns()))

        operand_, ext_edges = self.parse_operand(node.operand)
        edges.extend(ext_edges)
        edges.append({"src": operand_, "dst": un_name, "type": "operand"})
        edges.append({"src": node.op.__class__.__name__, "dst": un_name, "type": "op"})

        return edges, un_name


    def parse_BinOp(self, node):
        edges = []

        biop_name = "binop" + str(int(time_ns()))

        left, ext_edges = self.parse_operand(node.left)
        edges.extend(ext_edges)

        right, ext_edges = self.parse_operand(node.left)
        edges.extend(ext_edges)

        edges.append({"src": left, "dst": biop_name, "type": "leftop"})
        edges.append({"src": right, "dst": biop_name, "type": "rightop"})
        edges.append({"src": node.op.__class__.__name__, "dst": biop_name, "type": "op"})
        return edges, biop_name

    def parse_comprehension_like(self, node):
        edges = []

        gen_name = type(node).__name__ + str(int(time_ns()))

        name = self.parse(node.elt)
        edges.append({"src": name, "dst": gen_name, "type": "element"})
        for g in node.generators:
            gen, ext_edges = self.parse_operand(g)
            edges.extend(ext_edges)
            edges.append({"src": gen, "dst": gen_name, "type": "generator"})

        return edges, gen_name

    def parse_GeneratorExp(self, node):
        return self.parse_comprehension_like(node)

    def parse_ListComp(self, node):
        return self.parse_comprehension_like(node)

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
            return [type(node)]
        
    def parse_body(self, nodes):
        cond_name = ""
        cond_stat = ""
        edges = []
        for node in nodes:
            # print(ast.dump(node))
            n_type = type(node)
            if n_type in [ast.Expr, ast.FunctionDef, ast.Assign, ast.If, ast.Try, ast.For, ast.While, ast.Continue, ast.Pass]:
                s = self.parse(node)
                if isinstance(s, tuple):
                    edges.extend(self.parse(node)[0])
                else:
                    edges.extend(self.parse(node))
        return edges

f_bodies = pd.read_csv(sys.argv[1])
for c in f_bodies['content']:
    g = AstGraphGenerator(c.strip())
    # print(ast.dump(ast.parse(func)))
    g.generic_visit()
# pygraphviz.AGraph(g)
# for key, val in g.graph.items():
#     print(key, val, sep="\t")