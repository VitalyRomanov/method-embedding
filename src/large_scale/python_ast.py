import ast
import sys
from collections import defaultdict
from pprint import pprint
import pygraphviz
from time import time

"""
Possible strategies
1. Take Sourcetrail index and replace all occurrences in the source code with sourcetrail objects. This way all object will be disambiguated. Then, pass everything to ast parser. The problem is that I need to find field usages. This is probably not that hard, just start from the end of the line, and replace everything that is connected with a dot.
2. If a statement confitions on smth, add "conditions at" edge to every node
"""

func = open(sys.argv[1]).read()
# tree = ast.parse(func)
# for node in tree.body:
#     print(node.name)
#     print(node.args)
#     print(node.returns)
#     print(len(node.body))
def get_call(node):
    if type(node) == ast.Attribute:
        return get_call(node.value) + "_" + node.attr
    elif type(node) == ast.Name:
        return node.id

def parse_func_def(node):
    name  = node.name
    args =  (parse(n) for n in node.args.args)

    edges = []
    for a in args:
        edges.append({"src": a, "dst": name, "type":"def_arg"})
    return edges
    # return node.name, tuple(parse(n) for n in node.args.args)

def parse_assign(node):
    edges = []
    if type(node.value) == ast.Call:
        value = parse(node.value.func)
        edges.extend(parse(node.value))
        type_ = "assign_to_from_call"
    else:
        value = parse(node.value)
        type_ = "assign_to"
    print("\t\t", ast.dump(node.value), value)
    dsts = (parse_name(t) for t in node.targets)
    
    for dst in dsts:
        edges.append({"src": value, "dst": dst, "type": type_})
    return edges
    # return tuple(parse_name(t) for t in node.targets), parse(node.value)

def parse_arg(node):
    return node.arg

def parse_call(node):
    edges = []
    # print("\t\t", ast.dump(node))
    f_name = get_call(node.func)
    call_args = (parse(a) for a in node.args)
    for a in call_args:
        edges.append({"src": a, "dst": f_name, "type": "call_arg"})
    return edges
    # return get_call(node.func), tuple(parse(a) for a in node.args)

def parse_name(node):
    if type(node) == ast.Attribute:
        return parse_name(node.value) + "_" + node.attr
    elif type(node) == ast.Name:
        return node.id

def parse_name_const(node):
    return node.value

def parse_num(node):
    return node.n

def parse_if(node):
    edges = []
    condition = parse(node.test)
    print(condition)
    cond_name = condition[0]['dst']

    edges.extend(condition)
    edges.extend(parse_body(node.body))
    edges.extend(parse_body(node.orelse))
    return edges
    # return parse(node.test), parse_body(node.body), parse_body(node.orelse)

def parse_for(node):
    return [type(node)]

def parse_try(node):
    return [type(node)]

def parse_while(node):
    return [type(node)]

def parse_compare(node):
    edges = []
    left = parse(node.left)
    ops = (o.__class__.__name__ for o in node.ops)
    comps = (parse(c) for c in node.comparators)
    comp_name = "compare" + str(int(time()))
    edges.append({"src": left, "dst": comp_name, "type": "comp_left"})

    for o in ops:
        edges.append({"src": o, "dst": comp_name, "type": "comp_op"})
    for c in comps:
        edges.append({"src": c, "dst": comp_name, "type": "comp_right"})
    
    return edges
    # return parse(node.left), tuple(parse(o) for o in node.ops), tuple(parse(c) for c in node.comparators)

def parse_bool_op(node):
    edges = []
    bool_op_name = node.op.__class__.__name__ + str(int(time()))
    for c in node.values:
        op = parse(c)
        op_name = op[0]['dst']
        edges.extend(op)
        edges.append({"src": op_name, "dst": bool_op_name, "type": "bool_op"})
    return edges


def parse(node):
    n_type = type(node)
    if n_type == ast.FunctionDef:
        return parse_func_def(node)
    elif n_type == ast.arg:
        return parse_arg(node)
    elif n_type == ast.arguments:
        return "XXX", type(node)#[parse(a.arg) for a in node.args]
    elif n_type == ast.Call:
        return parse_call(node)
    elif n_type == ast.Name:
        return parse_name(node)
    elif n_type == ast.Attribute:
        return parse_name(node)
    elif n_type == ast.Assign:
        return parse_assign(node)
    elif n_type == ast.NameConstant:
        return parse_name_const(node)
    elif n_type == ast.Num:
        return parse_num(node)
    elif n_type == ast.If:
        return parse_if(node)
    elif n_type == ast.Try:
        return parse_try(node)
    elif n_type == ast.For:
        return parse_for(node)
    elif n_type == ast.While:
        return parse_while(node)
    elif n_type == ast.Compare:
        return parse_compare(node)
    elif n_type == ast.BoolOp:
        return parse_bool_op(node)
    else:
        return [type(node)]#type(node)
        # return node.__class__.__name__

def parse_body(nodes):
    edges = []
    for node in nodes:
        # print(ast.dump(node))
        n_type = type(node)
        if n_type == ast.Expr:
            expr_type = type(node.value)
            if expr_type == ast.Call:
                edges.extend(parse(node.value))
            else:
                edges.extend(type(node.value))
        elif n_type == ast.FunctionDef:
            edges.extend(parse(node))
        elif n_type == ast.Assign:
            edges.extend(parse(node))
        elif n_type == ast.If:
            edges.extend(parse(node))
        elif n_type == ast.Try:
            edges.extend(parse(node))
        elif n_type == ast.For:
            edges.extend(parse(node))
        elif n_type == ast.While:
            edges.extend(parse(node))
    return edges

class AstGraphGenerator(object):

    def __init__(self, source):
        self.graph = defaultdict(lambda: [])
        self.source = source  # lines of the source code

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
                edges.extend(parse(f_def_node))
                edges.extend(parse_body(ast.iter_child_nodes(f_def_node)))
        pprint(edges)
                # for node in ast.iter_child_nodes(f_def_node):
                #     n_type = type(node)
                #     if n_type == ast.Expr:
                #         expr_type = type(node.value)
                #         if expr_type == ast.Call:
                #             print(parse(node.value), node.lineno)
                #         else:
                #             print(type(node.value), node.lineno)
                #     elif n_type == ast.FunctionDef:
                #         print(parse(node), node.lineno)
                #     elif n_type == ast.Assign:
                #         print(parse(node), node.lineno)
                #     elif n_type == ast.If:
                #         print(parse(node), node.lineno)
                #     elif n_type == ast.Try:
                #         print(parse(node), node.lineno)
                #     elif n_type == ast.For:
                #         print(parse(node), node.lineno)
                #     elif n_type == ast.While:
                #         print(parse(node), node.lineno)
                    # nn = self._getid(node)
                    # print(nn)
        # for _, value in ast.iter_fields(node):
        #     if isinstance(value, list):
        #         for item in value:
        #             if isinstance(item, ast.AST):
        #                 self.visit(item)

        #     elif isinstance(value, ast.AST):
        #         node_source = self._getid(node)
        #         value_source = self._getid(value)
        #         print
        #         self.graph[node_source].append(value_source)
        #         # self.graph[type(node)].append(type(value))
        #         self.visit(value)

g = AstGraphGenerator(func.split("\n"))
print(ast.dump(ast.parse(func)))
g.generic_visit(ast.parse(func))
# pygraphviz.AGraph(g)
# for key, val in g.graph.items():
#     print(key, val, sep="\t")