import ast
import sys
from collections import defaultdict
from pprint import pprint
import pygraphviz

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
    return node.name, tuple(parse(n) for n in node.args.args)

def parse_arg(node):
    return node.arg

def parse_call(node):
    return get_call(node.func), tuple(parse(a) for a in node.args)

def parse_name(node):
    if type(node) == ast.Attribute:
        return parse_name(node.value) + "_" + node.attr
    elif type(node) == ast.Name:
        return node.id

def parse_name_const(node):
    return node.value

def parse_num(node):
    return node.n

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
        return tuple(parse_name(t) for t in node.targets), parse(node.value)
    elif n_type == ast.NameConstant:
        return parse_name_const(node)
    elif n_type == ast.Num:
        return parse_num(node)
    else:
        return type(node)

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
        for f_def_node in ast.iter_child_nodes(node):
            # enter module
            if type(f_def_node) == ast.FunctionDef:
                for node in ast.iter_child_nodes(f_def_node):
                    n_type = type(node)
                    if n_type == ast.Expr:
                        expr_type = type(node.value)
                        if expr_type == ast.Call:
                            print(parse(node.value), node.lineno)
                        else:
                            print(type(node.value), node.lineno)
                    elif n_type == ast.FunctionDef:
                        print(parse(node), node.lineno)
                    elif n_type == ast.Assign:
                        print(parse(node), node.lineno)
                    elif n_type == ast.If:
                        print(parse(node), node.lineno)
                    elif n_type == ast.Try:
                        print(parse(node), node.lineno)
                    elif n_type == ast.For:
                        print(parse(node), node.lineno)
                    elif n_type == ast.While:
                        print(parse(node), node.lineno)
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