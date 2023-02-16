import ast
from itertools import chain
from pprint import pprint

from SourceCodeTools.code.ast.python_ast3 import PythonNodeEdgeDefinitions, \
    PythonAstGraphBuilder, nodes_edges_to_df, make_python_ast_graph
from SourceCodeTools.code.ast.python_examples import PythonCodeExamplesForNodes


class PythonNodeEdgeCFDefinitions(PythonNodeEdgeDefinitions):
    node_type_enum_initialized = False
    edge_type_enum_initialized = False
    shared_node_types_initialized = False

    node_type_enum = None
    edge_type_enum = None

    leaf_types = None
    named_leaf_types = None
    tokenizable_types_and_annotations = None
    shared_node_types = None

    ast_node_type_edges = {
        "Assign": ["value", "targets"],
        "AugAssign": ["target", "op", "value"],
        # "Import": ["names"],
        "alias": ["name", "asname"],
        # "ImportFrom": ["module", "names"],
        "Delete": ["targets"],
        "Global": ["names"],
        "Nonlocal": ["names"],
        # "withitem": ["context_expr", "optional_vars"],
        # "Subscript": ["value", "slice"],
        # "Slice": ["lower", "upper", "step"],
        # "ExtSlice": ["dims"],
        # "Index": ["value"],
        "Starred": ["value", "ctx"],
        "Yield": ["value"],
        "ExceptHandler": ["type"],
        # "Call": ["func", "args", "keywords"],
        # "Compare": ["left", "ops", "comparators"],
        "BoolOp": ["values", "op"],
        "Assert": ["test", "msg"],
        "List": ["elts", "ctx"],
        "Tuple": ["elts", "ctx"],
        "Set": ["elts"],
        "UnaryOp": ["operand", "op"],
        "BinOp": ["left", "right", "op"],
        "Await": ["value"],
        # "GeneratorExp": ["elt", "generators"],
        # "ListComp": ["elt", "generators"],
        # "SetComp": ["elt", "generators"],
        # "DictComp": ["key", "value", "generators"],
        "Return": ["value"],
        "Raise": ["exc", "cause"],
        "YieldFrom": ["value"],
    }

    overriden_collapsing_inside = {
        "Call": ["func", "args", "keywords"],
        "ImportFrom": ["module", "names"],
        "Import": ["names"],
        "GeneratorExp": ["elt", "generators"],
        "ListComp": ["elt", "generators"],
        "SetComp": ["elt", "generators"],
        "DictComp": ["key", "value", "generators"],
        "Compare": ["left", "ops", "comparators"],
        "Subscript": ["value", "slice"],
        "Slice": ["lower", "upper", "step"],
        "ExtSlice": ["dims"],
        "Index": ["value"],
        "withitem": ["context_expr", "optional_vars"],
    }

    overriden_node_type_edges = {
        "Module": [],  # overridden
        "FunctionDef": ["function_name", "args", "decorator_list", "returned_by"],
        # overridden, `function_name` replaces `name`, `returned_by` replaces `returns`
        "AsyncFunctionDef": ["function_name", "args", "decorator_list", "returned_by"],
        # overridden, `function_name` replaces `name`, `returned_by` replaces `returns`
        "ClassDef": ["class_name"],  # overridden, `class_name` replaces `name`
        "AnnAssign": ["target", "value", "annotation_for"],  # overridden, `annotation_for` replaces `annotation`
        "With": ["items"],  # overridden
        "AsyncWith": ["items"],  # overridden
        "arg": ["arg", "annotation_for", "default"],  # overridden, `annotation_for` is custom
        "Lambda": ["lambda"],  # overridden
        "IfExp": ["test", "if_true", "if_false"],
        # overridden, `if_true` renamed from `body`, `if_false` renamed from `orelse`
        "keyword": ["arg", "value"],  # overridden
        "Attribute": ["value", "attr", "ctx"],  # overridden
        "Num": [],  # overridden
        "Str": [],  # overridden
        "Bytes": [],  # overridden
        "If": ["test"],  # overridden
        "For": ["target", "iter"],  # overridden
        "AsyncFor": ["target", "iter"],  # overridden
        "Try": [],  # overridden
        "While": [],  # overridden
        "Expr": ["value"],  # overridden
        "Dict": ["keys", "values"],  # overridden
        "JoinedStr": [],  # overridden
        "FormattedValue": ["value"],  # overridden
        "arguments": ["vararg", "posonlyarg", "arg", "kwonlyarg", "kwarg"],  # ["args", "vararg", "kwarg", "kwonlyargs", "posonlyargs"],  # overridden
        "comprehension": ["target", "iter", "ifs"],
        # overridden, `target_for` is custom, `iter_for` is customm `ifs_rev` is custom
    }

    extra_node_type_edges = {
        "mention": ["local_mention"]
    }

    context_edge_names = {
        "Module": ["defined_in_module"],
        "FunctionDef": ["defined_in_function"],
        "ClassDef": ["defined_in_class"],
        "With": ["executed_inside_with"],
        "AsyncWith": ["executed_inside_with"],
        "If": ["executed_if_true", "executed_if_false"],
        "For": ["executed_in_for", "executed_in_for_orelse"],
        "AsyncFor": ["executed_in_for", "executed_in_for_orelse"],
        "While": ["executed_in_while", "executed_while_true"],
        "Try": ["executed_in_try", "executed_in_try_final", "executed_in_try_else", "executed_in_try_except",
                "executed_with_try_handler"],
    }

    extra_edge_types = {
        "control_flow", "next", "instance", "inside", "node_type"
    }

    # exceptions needed when we do not want to filter some edge types using a simple rule `_rev`
    reverse_edge_exceptions = {
        "next": "prev",
        "local_mention": None,  # from name to variable mention
        "returned_by": None,  # for type annotations
        "annotation_for": None,  # for type annotations
        "control_flow": None,  # for control flow
        "op": None,  # for operations
        "attr": None,  # for attributes
        "node_type": None,
        "ctx": None,  # for context
        "default": None,  # for default value for arg
    }

    iterable_nodes = {  # parse_iterable
        "List", "Tuple", "Set"
    }

    named_nodes = {
        "Name", "NameConstant"  # parse_name
    }

    constant_nodes = {
        "Constant"  # parse_Constant
    }

    operand_nodes = {  # parse_op_name
        "And", "Or", "Not", "Is", "Gt", "Lt", "GtE", "LtE", "Eq", "NotEq", "Ellipsis", "Add", "Mod",
        "Sub", "UAdd", "USub", "Div", "Mult", "MatMult", "Pow", "FloorDiv", "RShift", "LShift", "BitAnd",
        "BitOr", "BitXor", "IsNot", "NotIn", "In", "Invert"
    }

    control_flow_nodes = {  # parse_control_flow
        "Continue", "Break", "Pass"
    }

    ctx_nodes = {  # parse_ctx
        "Load", "Store", "Del"
    }

    # extra node types exist for keywords and attributes to prevent them from
    # getting mixed with local variable mentions
    extra_node_types = {
        "mention",
        "#keyword#",
        "#attr#",
        "astliteral",
        "type_annotation",
        "Op",
        "CtlFlow", "CtlFlowInstance", "instance", "type_node", "ctx"
        # "subword", "subword_instance"
    }

    @classmethod
    def regular_node_types(cls):
        return set(cls.ast_node_type_edges.keys())

    @classmethod
    def overridden_node_types(cls):
        return set(cls.overriden_node_type_edges.keys()) | set(cls.overriden_collapsing_inside.keys())

    @classmethod
    def node_types(cls):
        return list(
            cls.regular_node_types() |
            cls.overridden_node_types() |
            cls.iterable_nodes | cls.named_nodes | cls.constant_nodes |
            cls.extra_node_types  # | cls.operand_nodes | cls.control_flow_nodes
        )

    @classmethod
    def scope_edges(cls):
        return set(map(lambda x: x, chain(*cls.context_edge_names.values())))  # "defined_in_" +

    @classmethod
    def auxiliary_edges(cls):
        direct_edges = cls.scope_edges() | cls.extra_edge_types
        reverse_edges = cls.compute_reverse_edges(direct_edges)
        return direct_edges | reverse_edges

    @classmethod
    def compute_reverse_edges(cls, direct_edges):
        reverse_edges = set()
        for edge in direct_edges:
            if edge in cls.reverse_edge_exceptions:
                reverse = cls.reverse_edge_exceptions[edge]
                if reverse is not None:
                    reverse_edges.add(reverse)
            else:
                reverse_edges.add(edge + "_rev")
        return reverse_edges

    @classmethod
    def edge_types(cls):
        direct_edges = list(
            set(chain(*cls.ast_node_type_edges.values())) |
            set(chain(*cls.overriden_node_type_edges.values())) |
            set(chain(*cls.overriden_collapsing_inside.values())) |
            set(chain(*cls.extra_node_type_edges.values())) |
            cls.scope_edges() | cls.extra_edge_types
             # | cls.named_nodes | cls.constant_nodes |
            # cls.operand_nodes | cls.control_flow_nodes | cls.extra_node_types
        )

        reverse_edges = list(cls.compute_reverse_edges(direct_edges))
        return direct_edges + reverse_edges

    @classmethod
    def _initialize_shared_nodes(cls):
        node_types_enum = cls.make_node_type_enum()
        ctx = {node_types_enum["ctx"]}
        type_nodes = {node_types_enum["type_node"]}
        annotation_types = {node_types_enum["type_annotation"]}
        tokenizable_types = {node_types_enum["Name"], node_types_enum["#attr#"], node_types_enum["#keyword#"]}
        python_token_types = {
            node_types_enum["Op"], node_types_enum["Constant"], node_types_enum["JoinedStr"],
            node_types_enum["CtlFlow"], node_types_enum["astliteral"]
        }
        subword_types = set()  # {node_types_enum["subword"]}

        # cls.leaf_types = annotation_types | subword_types | python_token_types
        # cls.named_leaf_types = annotation_types | tokenizable_types | python_token_types
        # cls.tokenizable_types_and_annotations = annotation_types | tokenizable_types

        cls.shared_node_types = annotation_types | subword_types | tokenizable_types | python_token_types | type_nodes | ctx

        cls.shared_node_types_initialized = True


class PythonCFGraphBuilder(PythonAstGraphBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse(self, node):
        n_type = type(node).__name__
        if n_type in self._graph_definitions.ast_node_type_edges:
            out = self.generic_parse(
                node,
                self._graph_definitions.ast_node_type_edges[n_type]
            )
        elif n_type in self._graph_definitions.overriden_node_type_edges:
            method_name = "parse_" + n_type
            out = self.__getattribute__(method_name)(node)
        elif n_type in self._graph_definitions.overriden_collapsing_inside:
            out = self.parse_collapsing_inside(node)
        elif n_type in self._graph_definitions.iterable_nodes:
            out = self.parse_iterable(node)
        elif n_type in self._graph_definitions.named_nodes:
            out = self.parse_name(node)
        elif n_type in self._graph_definitions.constant_nodes:
            out = self.parse_Constant(node)
        elif n_type in self._graph_definitions.operand_nodes:
            out = self.parse_op_name(node)
        elif n_type in self._graph_definitions.control_flow_nodes:
            out = self.parse_control_flow(node)
        elif n_type in self._graph_definitions.ctx_nodes:
            out = self.parse_ctx(node)
        else:
            print(type(node))
            print(ast.dump(node))
            print(node._fields)
            pprint(self._source_lines)
            out = self.generic_parse(node, node._fields)
        # if isinstance(out, tuple):
        #     edges, node = out
        #     self._add_edge(
        #         edges, src=self._get_node(name=n_type, type=self._node_types["type_node"]),
        #         dst=node, type=self._edge_types["node_type"]
        #     )
        #     return edges, node
        # else:
        return out

    def _what_is_inside(self, edges, attach_to, position_node=None):
        nodes_inside = set()
        edges_ = []
        positions = {}
        for e in edges:
            if self._node_pool[e.src].type.name in {"type_annotation", "returned_by"} or \
                    e.type.name in {"local_mention", "instance", "ctx"}:
                edges_.append(e)
            else:
                nodes_inside.add(e.src)
                nodes_inside.add(e.dst)
            if e.positions is not None:
                positions[e.src] = e.positions

        for n in nodes_inside:
            if self._node_pool[n].type.name in {"mention", "instance"} or self._node_pool[n].name in {
                "mention", "comprehension", "FormattedValue", "JoinedStr", "Dict", "Expr", "Bytes", "Str",
                "Num", "IfExp", "Lambda", "YieldFrom", "DictComp", "SetComp", "ListComp", "BoolOp", "Starred",
                "Subscript", "Slice", "ExtSlice", "Index"
            } or self._node_pool[n].type.name in {
                "type_annotation", "Name", "#attr#", "#keyword#", "Op", "Constant", "JoinedStr", "CtlFlow", "astliteral"
            }:
                self._add_edge(
                    edges_, src=n, dst=attach_to, type=self._edge_types["inside"], scope=self.latest_scope,
                    position=positions.get(n, None)
                )

        return edges_

    def parse_FunctionDef(self, node):
        # need to create function name before generic_parse so that the scope is set up correctly
        # scope is used to create local mentions of variable and function names
        fdef_node = self._get_node(node=node)
        self._scope.append(fdef_node)

        to_parse = []
        if len(node.args.args) > 0 or node.args.vararg is not None:
            to_parse.append("args")
        if len("decorator_list") > 0:
            to_parse.append("decorator_list")

        edges, f_name = self.generic_parse(node, to_parse, with_name=fdef_node)

        edges = self._what_is_inside(edges, f_name, position_node=node)

        if node.returns is not None:
            # returns stores return type annotation
            # can contain quotes
            # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
            # https://www.python.org/dev/peps/pep-0484/#forward-references
            annotation_position = self._get_positions_from_node(node.returns)
            annotation_string = self._get_source_from_range(*annotation_position)
            annotation = self._get_node(
                name=annotation_string, type=self._node_types["type_annotation"]
            )
            self._add_edge(edges, src=annotation, dst=f_name, type=self._edge_types["returned_by"],
                           scope=self.latest_scope, position_node=node.returns)

        self.parse_in_context(f_name, self._edge_types["defined_in_function"], edges, node.body)

        self._scope.pop(-1)

        ext_edges, func_name = self.parse_as_mention(name=node.name)
        edges.extend(ext_edges)

        assert isinstance(node.name, str)
        self._add_edge(edges, src=func_name, dst=f_name, type=self._edge_types["function_name"],
                       scope=self.latest_scope)

        return edges, f_name

    def parse_collapsing_inside(self, node):
        edges, node_ = self.generic_parse(node, self._graph_definitions.overriden_collapsing_inside[node.__class__.__name__])
        edges = self._what_is_inside(edges, node_, position_node=node)
        return edges, node_

    def parse_Lambda(self, node):
        # this is too ambiguous
        edges, lmb_name = self.generic_parse(node, [])
        self.parse_and_add_operand(lmb_name, node.body, self._edge_types["lambda"], edges)
        edges = self._what_is_inside(edges, lmb_name, position_node=node)

        return edges, lmb_name

    def parse_Attribute(self, node):
        if node.attr is not None:
            # change attr name so that it does not mix with variable names
            node.attr += "@#attr#"
        edges, node_ = self.generic_parse(node, ["value", "attr"])
        edges = self._what_is_inside(edges, node_, position_node=node)
        return edges, node_

    def postprocess(self):
        super(PythonCFGraphBuilder, self).postprocess()
        for i in range(len(self._edges)):
            edge = self._edges[i]
            if self._node_pool[edge.src].type.name in {"mention", "instance"}:
                reverse = edge.make_reverse(self._graph_definitions, self._node_pool[edge.src])
                if reverse is not None:
                    self._edges.append(reverse)

    def to_df(self):
        self.postprocess()
        nodes, edges = nodes_edges_to_df(self._node_pool.values(), self._edges)
        nodes, edges.drop_duplicates(["type", "src", "dst"])
        edges = edges[edges['src'] != edges['dst']]  # remove self-loops
        remaining = set(edges['src']) | set(edges['dst'])
        nodes = nodes[nodes["id"].apply(lambda x: x in remaining)]

        edges, offsets = self._get_offsets(edges)
        return nodes, edges, offsets


def make_python_cf_graph(
        source_code, **kwargs
):
    return make_python_ast_graph(
        source_code, graph_builder_class=PythonCFGraphBuilder, node_edge_definition_class=PythonNodeEdgeCFDefinitions,
        **kwargs
    )


if __name__ == "__main__":
    for example in PythonCodeExamplesForNodes.examples.values():
        nodes, edges = make_python_cf_graph(example.lstrip(), add_reverse_edges=False, add_mention_instances=False)
    c = "def f(a=5): f(a=4)"
    g = make_python_ast_graph(c.lstrip())
    # import sys
    # f_bodies = pd.read_csv(sys.argv[1])
    # failed = 0
    # for ind, c in enumerate(f_bodies['body_normalized']):
    #     if isinstance(c, str):
    #         try:
    #             g = AstGraphGenerator(c.lstrip())
    #         except SyntaxError as e:
    #             print(e)
    #             continue
    #         failed += 1
    #         edges = g.get_edges()
    #         # edges.to_csv(os.path.join(os.path.dirname(sys.argv[1]), "body_edges.csv"), mode="a", index=False, header=(ind==0))
    #         print("\r%d/%d" % (ind, len(f_bodies['normalized_body'])), end="")
    #     else:
    #         print("Skipped not a string")
    #
    # print(" " * 30, end="\r")
    # print(failed, len(f_bodies['normalized_body']))
