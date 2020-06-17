import ast
import sys
from collections import defaultdict
from pprint import pprint
import pygraphviz
from time import time_ns
from collections.abc import Iterable

"""
Possible strategies
1. Take Sourcetrail index and replace all occurrences in the source code with sourcetrail objects. This way all object will be disambiguated. Then, pass everything to ast parser. The problem is that I need to find field usages. This is probably not that hard, just start from the end of the line, and replace everything that is connected with a dot.
2. If a statement confitions on smth, add "conditions at" edge to every node
"""

func = open(sys.argv[1]).read()

class AstGraphGenerator(object):

    def __init__(self, source):
        self.graph = defaultdict(lambda: [])
        self.source = source  # lines of the source code
        self.current_contition = []
        self.contition_status = []

    def __str__(self):
        return str(self.graph)

    def _getid(self, node):
        
        try:
            n_type = type(node)
            if n_type == ast.FunctionDef:
                return parse(node)
            elif n_type == ast.arguments:
                return parse(node)
            elif n_type == ast.Expr:
                expr_type = type(node.value)
                if expr_type == ast.Call:
                    return parse(node.value.func), 
                return type(node.value) #[a.arg for a in node.args]
            lineno = node.lineno - 1
            # return node._fields
            return "%s: %s" % (type(node), self.source[lineno].strip())
    
        except AttributeError:
            return type(node)

    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        edges = []
        for f_def_node in ast.iter_child_nodes(node):
            # enter module
            if type(f_def_node) == ast.FunctionDef:
                edges.extend(self.parse(f_def_node))
                edges.extend(self.parse_body(ast.iter_child_nodes(f_def_node)))
        pprint(edges)

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
        # if type(node.value) == ast.Call:
        #     call_e = self.parse(node.value)
        #     edges.extend(call_e)
        #     value = call_e[0]['dst']
        # else:
        #     value = self.parse(node.value)
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

    def parse_If(self, node):
        edges = []
        cond_name, ext_edges = self.parse_operand(node.test)
        edges.extend(ext_edges)
        # if type(node.test) == ast.Name:
        #     cond_name = self.parse(node.test)
        # else:
        #     condition = self.parse(node.test)
        #     # print(condition)
        #     cond_name = condition[0]['dst']
        #     edges.extend(condition)

        self.parse_in_context(cond_name, "True", edges, node.body)
        # self.current_contition.append(cond_name)
        # self.contition_status.append("True")
        # edges.extend(self.parse_body(node.body))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)

        self.parse_in_context(cond_name, "False", edges, node.orelse)
        # self.current_contition.append(cond_name)
        # self.contition_status.append("False")
        # edges.extend(self.parse_body(node.orelse))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)

        return edges
        # return parse(node.test), parse_body(node.body), parse_body(node.orelse)

    def parse_For(self, node):
        edges = []
        # print("\t\tfordump", ast.dump(node))
        for_name = "for" + str(int(time_ns()))

        iter_, iter_e = self.parse_operand(node.iter)
        edges.extend(iter_e)
        # if type(node.iter) == ast.Name:
        #     iter_ = self.parse(node.iter)
        # else:
        #     iter_e = self.parse(node.iter)
        #     edges.extend(iter_e)
        #     iter_ = iter_e[0]['dst']
        edges.append({"src": iter_, "dst": self.parse(node.target), "type": "iter"})
        # print("\t\tfordump", edges)
        
        self.parse_in_context(for_name, "for", edges, node.body)
        # self.current_contition.append(for_name)
        # self.contition_status.append("for")
        # edges.extend(self.parse_body(node.body))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)

        self.parse_in_context(for_name, "orelse", edges, node.orelse)
        # self.current_contition.append(for_name)
        # self.contition_status.append("orelse")
        # edges.extend(self.parse_body(node.orelse))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)

        # print("\t\tfordump", edges)

        return edges #, for_name
        # return [type(node)]

    def parse_Try(self, node):
        edges = []
        try_name = "try" + str(int(time_ns()))

        self.parse_in_context(try_name, "try", edges, node.body)
        # self.current_contition.append(try_name)
        # self.contition_status.append("try")
        # edges.extend(self.parse_body(node.body))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)

        for h in node.handlers:
            
            handler_name = "None" if not h.type else self.parse(h.type)
            self.parse_in_context(try_name, "except_"+handler_name, edges, h.body)
            # self.current_contition.append(try_name)
            # self.contition_status.append("except_"+handler_name)
            # edges.extend(self.parse_body(h.body))
            # self.current_contition.pop(-1)
            # self.contition_status.pop(-1)

        self.parse_in_context(try_name, "final", edges, node.finalbody)
        # self.current_contition.append(try_name)
        # self.contition_status.append("final")
        # edges.extend(self.parse_body(node.finalbody))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)

        self.parse_in_context(try_name, "else", edges, node.orelse)
        # self.current_contition.append(try_name)
        # self.contition_status.append("else")
        # edges.extend(self.parse_body(node.orelse))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)

        return edges #, try_name   
        # return [type(node)]

    def parse_While(self, node):
        edges = []

        while_name = "while" + str(int(time_ns()))
        
        cond_name, ext_edges = self.parse_operand(node.test)
        edges.extend(ext_edges)

        # if type(node.test) == ast.Name:
        #     cond_name = self.parse(node.test)
        # else:
        #     condition = self.parse(node.test)
        #     # print(condition)
        #     cond_name = condition[0]['dst']
        #     edges.extend(condition)

        self.parse_in_context([while_name, cond_name], ["while", "True"], edges, node.body)
        # self.current_contition.append(while_name)
        # self.contition_status.append("while")
        # self.current_contition.append(cond_name)
        # self.contition_status.append("True")
        # edges.extend(self.parse_body(node.body))
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)
        # self.current_contition.pop(-1)
        # self.contition_status.pop(-1)
        # print("\t\twhiledump", print(edges))
        return edges #, while_name
        # return [type(node)]

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
        # return parse(node.left), tuple(parse(o) for o in node.ops), tuple(parse(c) for c in node.comparators)

    def parse_BoolOp(self, node):
        edges = []
        bool_op_name = node.op.__class__.__name__ + str(int(time_ns()))
        for c in node.values:
            op_name, ext_edges = self.parse_operand(c)
            # op = self.parse(c)
            # op_name = op[0]['dst']
            edges.extend(ext_edges)
            edges.append({"src": op_name, "dst": bool_op_name, "type": "bool_op"})
        return edges, bool_op_name

    def parse_Expr(self, node):
        edges = []

        # edges.extend(self.parse(node.value))
        expr_name, ext_edges = self.parse_operand(node.value)
        edges.extend(ext_edges)
        # expr_name = edges[0]['dst']

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
        # edges = []

        list_name = "list" + str(int(time_ns()))

        return self.parse_iterable(node, list_name)

        # for e in node.elts:
        #     val, ext_edges = self.parse_operand(e)
        #     edges.extend(ext_edges)
        #     edges.append({"src": val, "dst": list_name, "type": "element_of"})
        
        # return edges

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
        # if n_type == ast.FunctionDef:
        #     return self.parse_func_def(node)
        # elif n_type == ast.arg:
        #     return self.parse_arg(node)
        # elif n_type == ast.arguments:
        #     return "XXX", type(node)#[parse(a.arg) for a in node.args]
        # elif n_type == ast.Call:
        #     return self.parse_call(node)
        # elif n_type == ast.Name:
        #     return self.parse_name(node)
        # elif n_type == ast.NameConstant:
        #     return self.parse_name(node)
        # elif n_type == ast.Attribute:
        #     return self.parse_name(node)
        # elif n_type == ast.Assign:
        #     return self.parse_assign(node)
        # elif n_type == ast.NameConstant:
        #     return self.parse_name_const(node)
        # elif n_type == ast.Num:
        #     return self.parse_num(node)
        # elif n_type == ast.If:
        #     return self.parse_if(node)
        # elif n_type == ast.Try:
        #     return self.parse_try(node)
        # elif n_type == ast.For:
        #     return self.parse_for(node)
        # elif n_type == ast.While:
        #     return self.parse_while(node)
        # elif n_type == ast.Compare:
        #     return self.parse_compare(node)
        # elif n_type == ast.BoolOp:
        #     return self.parse_bool_op(node)
        # elif n_type == ast.Expr:
        #     return self.parse_expr(node)
        # elif n_type == ast.Continue:
        #     return self.parse_continue(node)
        # elif n_type == ast.Pass:
        #     return self.parse_pass(node)
        # elif n_type == ast.List:
        #     return self.parse_list(node)
        # elif n_type == ast.Tuple:
        #     return self.parse_tuple(node)
        # elif n_type == ast.Set:
        #     return self.parse_set(node)
        # elif n_type == ast.Dict:
        #     return self.parse_dict(node)
        # elif n_type == ast.BinOp:
        #     return self.parse_binop(node)
        # elif n_type == ast.GeneratorExp or n_type == ast.ListComp:
        #     return self.parse_generator(node)
        # elif n_type == ast.comprehension:
        #     return self.parse_compreh(node)
        # else:
        #     return [type(node)]#type(node)
            # return node.__class__.__name__

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
        #     if n_type == ast.Expr:
        #         edges.extend(self.parse(node)[0])
        #         # if expr_type == ast.Call:
        #         #     edges.extend(self.parse(node.value))
        #         # else:
        #         #     edges.extend(type(node.value))
        #     elif n_type == ast.FunctionDef:
        #         edges.extend(self.parse(node))
        #     elif n_type == ast.Assign:
        #         edges.extend(self.parse(node)[0])
        #     elif n_type == ast.If:
        #         edges.extend(self.parse(node))
        #     elif n_type == ast.Try:
        #         edges.extend(self.parse(node))
        #     elif n_type == ast.For:
        #         edges.extend(self.parse(node))
        #     elif n_type == ast.While:
        #         edges.extend(self.parse(node))
        #     elif n_type == ast.Continue:
        #         edges.extend(self.parse(node))
        #     elif n_type == ast.Pass:
        #         edges.extend(self.parse(node))
        return edges

g = AstGraphGenerator(func.split("\n"))
print(ast.dump(ast.parse(func)))
g.generic_visit(ast.parse(func))
# pygraphviz.AGraph(g)
# for key, val in g.graph.items():
#     print(key, val, sep="\t")