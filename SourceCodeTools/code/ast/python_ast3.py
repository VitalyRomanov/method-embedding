import ast
import hashlib
import logging
from collections import defaultdict
from enum import Enum
from itertools import chain
from pprint import pprint
from collections.abc import Iterable
from typing import Optional, Type

import pandas as pd

from SourceCodeTools.code.IdentifierPool import IdentifierPool
from SourceCodeTools.code.annotator_utils import get_cum_lens, to_offsets
from SourceCodeTools.nlp.string_tools import get_byte_to_char_map


class PythonNodeEdgeDefinitions:
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
        "Import": ["names"],
        "alias": ["name", "asname"],
        "ImportFrom": ["module", "names"],
        "Delete": ["targets"],
        "Global": ["names"],
        "Nonlocal": ["names"],
        "withitem": ["context_expr", "optional_vars"],
        "Subscript": ["value", "slice", "ctx"],
        "Slice": ["lower", "upper", "step"],
        "ExtSlice": ["dims"],
        "Index": ["value"],
        "Starred": ["value", "ctx"],
        "Yield": ["value"],
        "ExceptHandler": ["type"],
        "Call": ["func", "args", "keywords"],
        "Compare": ["left", "ops", "comparators"],
        "BoolOp": ["values", "op"],
        "Assert": ["test", "msg"],
        "List": ["elts", "ctx"],
        "Tuple": ["elts", "ctx"],
        "Set": ["elts"],
        "UnaryOp": ["operand", "op"],
        "BinOp": ["left", "right", "op"],
        "Await": ["value"],
        "GeneratorExp": ["elt", "generators"],
        "ListComp": ["elt", "generators"],
        "SetComp": ["elt", "generators"],
        "DictComp": ["key", "value", "generators"],
        "Return": ["value"],
        "Raise": ["exc", "cause"],
        "YieldFrom": ["value"],
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
        "control_flow", "next", "instance", "inside"
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
        "CtlFlow", "CtlFlowInstance", "instance", "ctx"
        # "subword", "subword_instance"
    }

    @classmethod
    def regular_node_types(cls):
        return set(cls.ast_node_type_edges.keys())

    @classmethod
    def overridden_node_types(cls):
        return set(cls.overriden_node_type_edges.keys())

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
            set(chain(*cls.extra_node_type_edges.values())) |
            cls.scope_edges() | cls.extra_edge_types
            # | cls.named_nodes | cls.constant_nodes |
            # cls.operand_nodes | cls.control_flow_nodes | cls.extra_node_types
        )

        reverse_edges = list(cls.compute_reverse_edges(direct_edges))
        return direct_edges + reverse_edges

    @classmethod
    def make_node_type_enum(cls) -> Enum:
        if not cls.node_type_enum_initialized:
            cls.node_type_enum = Enum(
                "NodeTypes",
                " ".join(
                    cls.node_types()
                )
            )
            cls.node_type_enum_initialized = True
        return cls.node_type_enum

    @classmethod
    def make_edge_type_enum(cls) -> Enum:
        if not cls.edge_type_enum_initialized:
            cls.edge_type_enum = Enum(
                "EdgeTypes",
                " ".join(
                    cls.edge_types()
                )
            )
            cls.edge_type_enum_initialized = True
        return cls.edge_type_enum

    @classmethod
    def _initialize_shared_nodes(cls):
        node_types_enum = cls.make_node_type_enum()
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

        cls.shared_node_types = annotation_types | subword_types | tokenizable_types | python_token_types

        cls.shared_node_types_initialized = True

    @classmethod
    def get_shared_node_types(cls):
        if not cls.shared_node_types_initialized:
            cls._initialize_shared_nodes()
        return cls.shared_node_types

    @classmethod
    def is_shared_name_type(cls, name, type):
        if not cls.shared_node_types_initialized:
            cls._initialize_shared_nodes()

        if type in cls.shared_node_types:
            return True
        return False

    @classmethod
    def get_reverse_type(cls, type):
        if type.endswith("_rev"):
            return None

        if not cls.edge_type_enum_initialized:
            cls.make_edge_type_enum()

        reverse_type = cls.reverse_edge_exceptions.get(type, type + "_rev")
        if reverse_type is not None:
            return cls.edge_type_enum[reverse_type]
        return None


class PythonCodeExamplesForNodes:
    examples = {
        "FunctionDef":
            "def f(a):\n"
            "   return a\n",
        "ClassDef":
            "class C:\n"
            "   def m():\n"
            "       pass\n",
        "AnnAssign": "a: int = 5\n",
        "With":
            "with open(a) as b:\n"
            "   do_stuff(b)\n",
        "arg":
            "def f(a: int = 5):\n"
            "   return a\n",
        "Lambda": "lambda x: x + 3\n",
        "IfExp": "a = 5 if True else 0\n",
        "keyword": "fn(a=5, b=4)\n",
        "Attribute": "a.b.c\n",
        "If":
            "if d is True:\n"
            "   a = b\n"
            "else:\n"
            "   a = c\n",
        "For":
            "for i in list:\n"
            "   k = fn(i)\n"
            "   if k == 4:\n"
            "       fn2(k)\n"
            "       break\n"
            "else:\n"
            "   fn2(0)\n",
        "Try":
            "try:\n"
            "   a = b\n"
            "except Exception as e:\n"
            "   a = c\n"
            "else:\n"
            "   a = d\n"
            "finally:\n"
            "   print(a)\n",
        "While":
            "while b == c:\n"
            "   do_iter(b)\n",
        "Dict": "{a:b, c:d}\n",
        "comprehension": "[i for i in list if i != 5]\n",
        "BinOp": "c = a + b\n",
        "ImportFrom": "from module import Class\n",
        "alias": "import module as m\n",
        "List": "a = [1, 2, 3, 4]\n"
    }


def generate_available_edges(node_edge_definition_class: Type[PythonNodeEdgeDefinitions]):
    # node_types = PythonNodeEdgeDefinitions.node_types()
    node_types = node_edge_definition_class.node_types()
    for nt in sorted(node_types):
        if hasattr(ast, nt):
            fl = sorted(getattr(ast, nt)._fields)
            if len(fl) == 0:
                print(nt, )
            else:
                for f in fl:
                    print(nt, f, sep=" ")


def generate_utilized_edges(node_edge_definition_class: Type[PythonNodeEdgeDefinitions]):
    d = dict()
    # d.update(PythonNodeEdgeDefinitions.ast_node_type_edges)
    # d.update(PythonNodeEdgeDefinitions.overriden_node_type_edges)
    d.update(node_edge_definition_class.ast_node_type_edges)
    d.update(node_edge_definition_class.overriden_node_type_edges)
    for nt in sorted(d.keys()):
        if hasattr(ast, nt):
            fl = sorted(d[nt])
            if len(fl) == 0:
                print(nt, )
            else:
                for f in fl:
                    print(nt, f, sep=" ")


def nodes_edges_to_df(nodes, edges):
    edge_specification = {
        "id": ("hash_id", "int64", None),
        "src": ("src", "int64", None),
        "dst": ("dst", "int64", None),
        "type": ("type", "string", lambda x: x.name),
        "scope": ("scope", "Int64", None),
        "offset_start": ("positions", "Int64", lambda x: x[0] if isinstance(x, tuple) else x),
        "offset_end": ("positions", "Int64", lambda x: x[1] if isinstance(x, tuple) else x),
    }

    node_specification = {
        "id": ("hash_id", "int64", None),
        "name": ("name", "string", None),
        "type": ("type", "string", lambda x: x.name),
        "scope": ("scope", "Int64", None),
        "string": ("string", "string", None),
        "offset_start": ("positions", "Int64", lambda x: x[0] if isinstance(x, tuple) else x),
        "offset_end": ("positions", "Int64", lambda x: x[1] if isinstance(x, tuple) else x),
    }

    def create_table(collection, specification):
        entries = defaultdict(list)
        for record in collection:
            # entry = {}
            for trg_col, (src_col, _, preproc_fn) in specification.items():
                value = getattr(record, src_col)
                if value is None:
                    value = pd.NA
                entries[trg_col].append(value if preproc_fn is None else preproc_fn(value))
            # entries.append(entry)

        table = pd.DataFrame(entries)

        for trg_col, (_, dtype, _) in specification.items():
            table = table.astype({trg_col: dtype})
        return table

    nodes = create_table(nodes, node_specification)
    edges = create_table(edges, edge_specification)
    edges.drop_duplicates("id", inplace=True)

    return nodes, edges


class PythonAstGraphBuilder(object):
    def __init__(
            self, source, graph_definitions, add_reverse_edges=True, save_node_strings=True,
            add_mention_instances=False, parse_constants=False,
            # parse_ctx=False,
            **kwargs
    ):
        self._node_types = graph_definitions.make_node_type_enum()
        self._edge_types = graph_definitions.make_edge_type_enum()
        self._graph_definitions = graph_definitions
        self._original_source = source
        self._source_lines = source.split("\n")  # lines of the source code
        self._root = ast.parse(source)
        self._current_condition = []
        self._condition_status = []
        self._scope = []
        self._add_reverse_edges = add_reverse_edges
        self._add_mention_instances = add_mention_instances
        self._parse_constants = parse_constants
        # self._parse_ctx = parse_ctx
        self._save_node_strings = save_node_strings
        self._node_pool = dict()
        self._cum_lens = get_cum_lens(self._original_source, as_bytes=True)
        self._byte2char = get_byte_to_char_map(self._original_source)
        self._set_node_class()
        self._set_edge_class()

        self._identifier_pool = IdentifierPool()
        self._edges = self._parse(self._root)[0]

    def _set_node_class(self):
        class GNode:
            def __init__(self, name, type, string=None, **kwargs):
                self.name = name
                self.type = type
                self.string = string
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def __eq__(self, other):
                return self.name == other.name and self.type == other.type

            def __repr__(self):
                return self.__dict__.__repr__()

            def __hash__(self):
                return (self.name, self.type).__hash__()

            def setprop(self, key, value):
                setattr(self, key, value)

            @property
            def hash_id(self):
                if not hasattr(self, "node_hash"):
                    self.node_hash = abs(
                        int(hashlib.md5(f"{self.type.name.strip()}_{self.name.strip()}".encode('utf-8')).hexdigest()[
                            :8], 16))
                return self.node_hash

        self._make_node = GNode

    def _set_edge_class(builderself):
        class GEdge:
            def __init__(
                    self, src: int, dst: int, type, scope: Optional[int] = None,
            ):
                self.src = src
                self.dst = dst
                self.type = type
                self.scope = scope
                self.positions = None

            def assign_positions(self, positions, prefix: Optional[str] = None):
                if prefix is None:
                    self.positions = positions
                else:
                    if positions is not None:
                        setattr(self, f"{prefix}_positions", positions)

            def make_reverse(self, *args, **kwargs):
                reverse_type = builderself._graph_definitions.get_reverse_type(self.type.name)
                src_node = builderself._node_pool[self.src]
                if reverse_type is not None and not builderself._graph_definitions.is_shared_name_type(src_node.name,
                                                                                                   src_node.type):
                    return self.__class__(src=self.dst, dst=self.src, type=reverse_type, scope=self.scope)
                else:
                    return None

            def __getitem__(self, item):
                return self.__dict__[item]

            @property
            def hash_id(self):
                if not hasattr(self, "edge_hash"):
                    self.edge_hash = abs(
                        int(hashlib.md5(f"{self.src}_{self.dst}_{self.type}".encode('utf-8')).hexdigest()[:8], 16))
                return self.edge_hash

        builderself._make_edge = GEdge

    @property
    def latest_scope(self):
        if len(self._scope) > 0:
            return self._scope[-1]
        else:
            return None

    @property
    def latest_scope_name(self):
        if len(self._scope) > 0:
            scope = self._node_pool[self._scope[-1]]
            return scope.name
        else:
            return None

    def _into_offset(self, range):
        if isinstance(range, dict):
            range = (range["line"], range["end_line"], range["col_offset"], range["end_col_offset"])

        assert len(range) == 4

        try:
            return to_offsets(
                self._original_source, [(*range, None)], cum_lens=self._cum_lens, b2c=self._byte2char, as_bytes=True
            )[-1][:2]
        except:
            return None

    def _get_positions_from_node(self, node):
        if node is not None and hasattr(node, "lineno"):
            positions = {
                "line": node.lineno - 1,
                "end_line": node.end_lineno - 1,
                "col_offset": node.col_offset,
                "end_col_offset": node.end_col_offset
            }
            positions = self._into_offset(positions)
        else:
            positions = None
        return positions

    def _get_source_from_range(self, start, end):
        return self._original_source[start: end]

    def _get_node(
            self, *, node=None, name: Optional[str] = None, type=None,
            positions=None, scope=None, add_random_identifier: bool = False
    ) -> int:

        random_identifier = self._identifier_pool.get_new_identifier()

        if name is not None:
            assert name is not None and type is not None
            if add_random_identifier:
                name = f"{name}_{random_identifier}"
        else:
            assert node is not None
            name = f"{node.__class__.__name__}_{random_identifier}"
            type = self._node_types[node.__class__.__name__]

        if positions is None:
            positions = self._get_positions_from_node(node)
        if self._save_node_strings:
            node_string = self._get_source_from_range(*positions) if positions is not None else None
        else:
            node_string = None

        if scope is None and self._graph_definitions.is_shared_name_type(name, type) is False:
            scope = self.latest_scope

        new_node = self._make_node(name=name, type=type, scope=scope, string=node_string, positions=positions)
        self._node_pool[new_node.hash_id] = new_node
        return new_node.hash_id

    def _parse(self, node):
        n_type = type(node).__name__
        if n_type in self._graph_definitions.ast_node_type_edges:
            return self.generic_parse(
                node,
                self._graph_definitions.ast_node_type_edges[n_type]
            )
        elif n_type in self._graph_definitions.overriden_node_type_edges:
            method_name = "parse_" + n_type
            return self.__getattribute__(method_name)(node)
        elif n_type in self._graph_definitions.iterable_nodes:
            return self.parse_iterable(node)
        elif n_type in self._graph_definitions.named_nodes:
            return self.parse_name(node)
        elif n_type in self._graph_definitions.constant_nodes:
            return self.parse_Constant(node)
        elif n_type in self._graph_definitions.operand_nodes:
            return self.parse_op_name(node)
        elif n_type in self._graph_definitions.control_flow_nodes:
            return self.parse_control_flow(node)
        elif n_type in self._graph_definitions.ctx_nodes:
            return self.parse_ctx(node)
        else:
            print(type(node))
            print(ast.dump(node))
            print(node._fields)
            pprint(self._source_lines)
            return self.generic_parse(node, node._fields)

    def _add_edge(
            self, edges, src: int, dst: int, type, scope: Optional[int] = None,
            position_node=None, var_position_node=None, position=None
    ):
        new_edge = self._make_edge(src=src, dst=dst, type=type, scope=scope)
        new_edge.assign_positions(self._get_positions_from_node(position_node))
        new_edge.assign_positions(self._get_positions_from_node(var_position_node), prefix="var")
        if position is not None:
            assert position_node is None, "position conflict"
            new_edge.assign_positions(position)

        edges.append(new_edge)

        if self._add_reverse_edges is True:
            reverse = new_edge.make_reverse()
            if reverse is not None:
                edges.append(reverse)

    def parse_body(self, nodes):
        edges = []
        last_node = None
        for node in nodes:
            s = self._parse(node)
            if isinstance(s, tuple):
                if self._node_pool[s[1]].type == self._node_types["Constant"]:
                    # this happens when processing docstring, as a result a lot of nodes are connected to the node
                    # Constant_
                    continue  # in general, constant node has no affect as a body expression, can skip
                # some parsers return edges and names?
                edges.extend(s[0])

                if last_node is not None:
                    self._add_edge(edges, src=last_node, dst=s[1], type=self._edge_types["next"],
                                   scope=self.latest_scope)

                last_node = s[1]

                for cond_name, cond_stat in zip(self._current_condition[-1:], self._condition_status[-1:]):
                    self._add_edge(edges, src=last_node, dst=cond_name, type=cond_stat,
                                   scope=self.latest_scope)  # "defined_in_" +
            else:
                edges.extend(s)
        return edges

    def parse_in_context(self, cond_name, cond_stat, edges, body):
        if not isinstance(cond_name, list):
            cond_name = [cond_name]
            cond_stat = [cond_stat]

        for cn, cs in zip(cond_name, cond_stat):
            self._current_condition.append(cn)
            self._condition_status.append(cs)

        edges.extend(self.parse_body(body))

        for i in range(len(cond_name)):
            self._current_condition.pop(-1)
            self._condition_status.pop(-1)

    def parse_as_mention(self, name, ctx=None):
        mention_name = self._get_node(name=name + "@" + self.latest_scope_name, type=self._node_types["mention"])
        name_ = self._get_node(name=name, type=self._node_types["Name"])

        edges = []
        self._add_edge(edges, src=name_, dst=mention_name, type=self._edge_types["local_mention"],
                       scope=self.latest_scope)

        if self._add_mention_instances:
            mention_instance = self._get_node(
                name="instance", type=self._node_types["instance"], add_random_identifier=True
            )
            self._node_pool[mention_instance].string = name
            self._add_edge(
                edges, src=mention_name, dst=mention_instance, type=self._edge_types["instance"],
                scope=self.latest_scope
            )
            mention_name = mention_instance

            if ctx is not None:
                ctx_node = self._parse(ctx)
                self._add_edge(
                    edges, src=ctx_node, dst=mention_instance, type=self._edge_types["ctx"],
                    scope=self.latest_scope
                )
        return edges, mention_name

    def parse_operand(self, node):
        # need to make sure upper level name is correct; handle @keyword; type placeholder for sourcetrail nodes?
        edges = []
        if isinstance(node, str):
            # fall here when parsing attributes, they are given as strings; should attributes be parsed into subwords?
            if "@" in node:
                node_name, node_type = node.split("@")
                node = self._get_node(name=node_name, type=self._node_types[node_type])
            else:
                node = ast.Name(node)
                edges_, node = self._parse(node)
                edges.extend(edges_)
            iter_ = node
        elif isinstance(node, int) or node is None:
            iter_ = self._get_node(name=str(node), type=self._node_types["astliteral"])
        else:
            iter_e = self._parse(node)
            if type(iter_e) == str:
                iter_ = iter_e
            elif isinstance(iter_e, int):
                iter_ = iter_e
            elif type(iter_e) == tuple:
                ext_edges, name = iter_e
                assert isinstance(name, int) and name in self._node_pool
                edges.extend(ext_edges)
                iter_ = name
            else:
                # unexpected scenario
                print(node)
                print(ast.dump(node))
                print(iter_e)
                print(type(iter_e))
                pprint(self._source_lines)
                print(self._source_lines[node.lineno - 1].strip())
                raise Exception()

        return iter_, edges

    def parse_and_add_operand(self, node_name, operand, type, edges):

        operand_name, ext_edges = self.parse_operand(operand)
        edges.extend(ext_edges)

        if not isinstance(type, self._edge_types):
            type = self._edge_types[type]

        self._add_edge(edges, src=operand_name, dst=node_name, type=type, scope=self.latest_scope,
                       position_node=operand)

    def generic_parse(self, node, operands, with_name=None, ensure_iterables=False):

        edges = []

        if with_name is None:
            node_name = self._get_node(node=node)
        else:
            node_name = with_name

        for operand in operands:
            if operand in ["body", "orelse", "finalbody"]:
                logging.warning(f"Not clear which node is processed here {ast.dump(node)}")
                self.parse_in_context(node_name, operand, edges, node.__getattribute__(operand))
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
        # edges.append({"scope": copy(self._scope[-1]), "src": node.__class__.__name__, "dst": node_name, "type": "node_type"})

        return edges, node_name

    # def parse_type_node(self, node):
    #     if node.lineno == node.end_lineno:
    #         type_str = self._source_lines[node.lineno][node.col_offset - 1: node.end_col_offset]
    #     else:
    #         type_str = ""
    #         for ln in range(node.lineno - 1, node.end_lineno):
    #             if ln == node.lineno - 1:
    #                 type_str += self._source_lines[ln][node.col_offset - 1:].strip()
    #             elif ln == node.end_lineno - 1:
    #                 type_str += self._source_lines[ln][:node.end_col_offset].strip()
    #             else:
    #                 type_str += self._source_lines[ln].strip()
    #     return type_str

    def parse_Module(self, node):
        edges, module_name = self.generic_parse(node, [])
        self._scope.append(module_name)
        self.parse_in_context(module_name, self._edge_types["defined_in_module"], edges, node.body)
        self._scope.pop(-1)
        return edges, module_name

    def parse_FunctionDef(self, node):
        # need to create function name before generic_parse so that the scope is set up correctly
        # scope is used to create local mentions of variable and function names
        fdef_node = self._get_node(node=node)
        self._scope.append(fdef_node)

        to_parse = []
        if (
                len(node.args.posonlyargs) > 0 or
                len(node.args.args) > 0 or
                len(node.args.kwonlyargs) > 0 or
                node.args.vararg is not None or
                node.args.kwarg is not None
        ):
            to_parse.append("args")
        if len("decorator_list") > 0:
            to_parse.append("decorator_list")

        edges, f_name = self.generic_parse(node, to_parse, with_name=fdef_node)

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

    def parse_AsyncFunctionDef(self, node):
        return self.parse_FunctionDef(node)

    def parse_ClassDef(self, node):

        edges, class_node_name = self.generic_parse(node, [])

        self._scope.append(class_node_name)
        self.parse_in_context(class_node_name, self._edge_types["defined_in_class"], edges, node.body)
        self._scope.pop(-1)

        ext_edges, cls_name = self.parse_as_mention(name=node.name)
        edges.extend(ext_edges)
        self._add_edge(edges, src=class_node_name, dst=cls_name, type=self._edge_types["class_name"],
                       scope=self.latest_scope)

        return edges, class_node_name

    def parse_With(self, node):
        edges, with_name = self.generic_parse(node, self._graph_definitions.overriden_node_type_edges["With"])

        self.parse_in_context(with_name, self._edge_types["executed_inside_with"], edges, node.body)

        return edges, with_name

    def parse_AsyncWith(self, node):
        return self.parse_With(node)

    def parse_arg(self, node, default_value=None):
        # node.annotation stores type annotation
        # if node.annotation:
        #     print(self.source[node.lineno-1]) # can get definition string here
        #     print(node.arg)

        # # included mention
        name = self._get_node(node=node)
        edges, mention_name = self.parse_as_mention(node.arg)
        self._add_edge(
            edges, src=mention_name, dst=name, type=self._edge_types["arg"], scope=self.latest_scope,
            position_node=node
        )

        if node.annotation is not None:
            # can contain quotes
            # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
            # https://www.python.org/dev/peps/pep-0484/#forward-references
            positions = self._get_positions_from_node(node.annotation)
            annotation_string = self._get_source_from_range(*positions)
            annotation = self._get_node(name=annotation_string, type=self._node_types["type_annotation"])
            mention_name = self._get_node(
                name=node.arg + "@" + self.latest_scope_name, type=self._node_types["mention"],
                scope=self.latest_scope
            )
            self._add_edge(edges, src=annotation, dst=mention_name, type=self._edge_types["annotation_for"],
                           scope=self.latest_scope, position_node=node.annotation, var_position_node=node)

        if default_value is not None:
            deflt_ = self._parse(default_value)
            if isinstance(deflt_, tuple):
                edges.extend(deflt_[0])
                default_val = deflt_[1]
            else:
                default_val = deflt_
            self._add_edge(edges, default_val, name, type=self._edge_types["default"], position_node=default_value,
                          scope=self.latest_scope)
        return edges, name

    def parse_AnnAssign(self, node):
        # stores annotation information for variables
        #
        # paths: List[Path] = []
        # AnnAssign(target=Name(id='paths', ctx=Store()), annotation=Subscript(value=Name(id='List', ctx=Load()),
        #           slice=Index(value=Name(id='Path', ctx=Load())),
        #           ctx=Load()), value=List(elts=[], ctx=Load()), simple=1)

        # TODO
        # parse value??

        # can contain quotes
        # https://stackoverflow.com/questions/46458470/should-you-put-quotes-around-type-annotations-in-python
        # https://www.python.org/dev/peps/pep-0484/#forward-references
        positions = self._get_positions_from_node(node.annotation)
        annotation_string = self._get_source_from_range(*positions)
        annotation = self._get_node(
            name=annotation_string, type=self._node_types["type_annotation"]
        )
        edges, name = self.generic_parse(node, ["target", "value"])
        try:
            mention_name = self._get_node(
                name=node.target.id + "@" + self.latest_scope_name, type=self._node_types["mention"],
                scope=self.latest_scope
            )
        except Exception as e:
            mention_name = name

        self._add_edge(edges, src=annotation, dst=mention_name, type=self._edge_types["annotation_for"],
                       scope=self.latest_scope, position_node=node.annotation, var_position_node=node)
        return edges, name

    def parse_Lambda(self, node):
        # this is too ambiguous
        edges, lmb_name = self.generic_parse(node, [])
        self.parse_and_add_operand(lmb_name, node.body, self._edge_types["lambda"], edges)

        return edges, lmb_name

    def parse_IfExp(self, node):
        edges, ifexp_name = self.generic_parse(node, ["test"])
        self.parse_and_add_operand(ifexp_name, node.body, self._edge_types["if_true"], edges)
        self.parse_and_add_operand(ifexp_name, node.orelse, self._edge_types["if_false"], edges)
        return edges, ifexp_name

    def parse_ExceptHandler(self, node):
        # have missing fields. example:
        # not parsing "name" field
        # except handler is unique for every function
        return self.generic_parse(node, ["type"])

    def parse_keyword(self, node):
        if isinstance(node.arg, str):
            # change arg name so that it does not mix with variable names
            node.arg += "@#keyword#"
            return self.generic_parse(node, self._graph_definitions.overriden_node_type_edges["keyword"])
        else:
            return self.generic_parse(node, ["value"])

    def parse_name(self, node):
        edges = []
        # if type(node) == ast.Attribute:
        #     left, ext_edges = self.parse(node.value)
        #     right = node.attr
        #     return self.parse(node.value) + "___" + node.attr
        if type(node) == ast.Name:
            return self.parse_as_mention(str(node.id), ctx=node.ctx if hasattr(node, "ctx") else None)
        elif type(node) == ast.NameConstant:
            return self._get_node(name=str(node.value), type=self._node_types["NameConstant"])

    def parse_Attribute(self, node):
        if node.attr is not None:
            # change attr name so that it does not mix with variable names
            node.attr += "@#attr#"
        return self.generic_parse(node, self._graph_definitions.overriden_node_type_edges["Attribute"])

    def parse_Constant(self, node):
        # TODO
        # decide whether this name should be unique or not
        if self._parse_constants:
            name_ = str(node.value)
        else:
            value_type = type(node.value).__name__
            name_ = f"Constant_{value_type}"
        name = self._get_node(name=name_, type=self._node_types["Constant"])
        # name = "Constant_"
        # if node.kind is not None:
        #     name += ""
        return name

    def parse_op_name(self, node):
        return self._get_node(name=node.__class__.__name__, type=self._node_types["Op"])
        # return node.__class__.__name__

    def parse_Num(self, node):
        return str(node.n)

    def parse_Str(self, node):
        return self.generic_parse(node, [])
        # return node.s

    def parse_Bytes(self, node):
        return repr(node.s)

    def parse_If(self, node):

        edges, if_name = self.generic_parse(node, ["test"])

        self.parse_in_context(if_name, self._edge_types["executed_if_true"], edges, node.body)
        self.parse_in_context(if_name, self._edge_types["executed_if_false"], edges, node.orelse)

        return edges, if_name

    def parse_For(self, node):

        edges, for_name = self.generic_parse(node, ["target", "iter"])

        self.parse_in_context(for_name, self._edge_types["executed_in_for"], edges, node.body)
        self.parse_in_context(for_name, self._edge_types["executed_in_for_orelse"], edges, node.orelse)

        return edges, for_name

    def parse_AsyncFor(self, node):
        return self.parse_For(node)

    def parse_Try(self, node):

        edges, try_name = self.generic_parse(node, [])

        self.parse_in_context(try_name, self._edge_types["executed_in_try"], edges, node.body)

        for h in node.handlers:
            handler_name, ext_edges = self.parse_operand(h)
            edges.extend(ext_edges)
            self.parse_in_context(
                [handler_name],  # [try_name, handler_name],
                [self._edge_types["executed_with_try_handler"]],
                # [self._edge_types["executed_in_try_except"], self._edge_types["executed_with_try_handler"]],
                edges, h.body
            )
            self._add_edge(edges, src=handler_name, dst=try_name, type=self._edge_types["executed_in_try_except"])

        self.parse_in_context(try_name, self._edge_types["executed_in_try_final"], edges, node.finalbody)
        self.parse_in_context(try_name, self._edge_types["executed_in_try_else"], edges, node.orelse)

        return edges, try_name

    def parse_While(self, node):

        edges, while_name = self.generic_parse(node, ["test"])

        # cond_name, ext_edges = self.parse_operand(node.test)
        # edges.extend(ext_edges)

        self.parse_in_context(
            [while_name],  # [while_name, cond_name],
            [self._edge_types["executed_in_while"]],  # [self._edge_types["executed_in_while"], self._edge_types["executed_while_true"]],
            edges, node.body
        )

        return edges, while_name

    # def parse_Compare(self, node):
    #     return self.generic_parse(node, ["left", "ops", "comparators"])
    #
    # def parse_BoolOp(self, node):
    #     return self.generic_parse(node, ["values", "op"])

    def parse_Expr(self, node):
        edges = []
        expr_name, ext_edges = self.parse_operand(node.value)
        edges.extend(ext_edges)

        return edges, expr_name

    def parse_control_flow(self, node):
        edges = []
        ctrlflow_name = self._get_node(
            name="ctrl_flow", type=self._node_types["CtlFlowInstance"], node=node, add_random_identifier=True
        )
        self._add_edge(edges, src=self._get_node(name=node.__class__.__name__, type=self._node_types["CtlFlow"]),
                       dst=ctrlflow_name, type=self._edge_types["control_flow"], scope=self.latest_scope,
                       position_node=node)

        return edges, ctrlflow_name

    def parse_ctx(self, node):
        ctx_name = self._get_node(
            name=node.__class__.__name__, type=self._node_types["ctx"], node=node, scope=None
        )
        return ctx_name

    def parse_iterable(self, node):
        return self.generic_parse(node, ["elts", "ctx"], ensure_iterables=True)

    def parse_Dict(self, node):
        return self.generic_parse(node, ["keys", "values"], ensure_iterables=True)

    def parse_JoinedStr(self, node):
        joinedstr_name = self._get_node(
            name="JoinedStr_", type=self._node_types["JoinedStr"], node=node
        )
        return [], joinedstr_name
        # return self.generic_parse(node, [])
        # return self.generic_parse(node, ["values"])

    def parse_FormattedValue(self, node):
        # have missing fields. example:
        # FormattedValue(value=Subscript(value=Name(id='args', ctx=Load()), slice=Index(value=Num(n=0)), ctx=Load()),conversion=-1, format_spec=None)
        # 'conversion', 'format_spec' not parsed
        return self.generic_parse(node, ["value"])

    def parse_arguments(self, node):
        # have missing fields. example:
        #    arguments(args=[arg(arg='self', annotation=None), arg(arg='tqdm_cls', annotation=None), arg(arg='sleep_interval', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])

        edges, arguments = self.generic_parse(node, [])

        if node.vararg is not None:
            ext_edges_, vararg = self.parse_arg(node.vararg)
            edges.extend(ext_edges_)
            self._add_edge(edges, vararg, arguments, type=self._edge_types["vararg"], position_node=node.vararg,
                          scope=self.latest_scope)

        for i in range(len(node.posonlyargs)):
            ext_edges_, posarg = self.parse_arg(node.posonlyargs[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, posarg, arguments, type=self._edge_types["posonlyarg"], position_node=node.posonlyargs[i],
                          scope=self.latest_scope)

        without_default = len(node.args) - len(node.defaults)
        for i in range(without_default):
            ext_edges_, just_arg = self.parse_arg(node.args[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, just_arg, arguments, type=self._edge_types["arg"], position_node=node.args[i],
                          scope=self.latest_scope)

        for ind, i in enumerate(range(without_default, len(node.args))):
            ext_edges_, just_arg = self.parse_arg(node.args[i], default_value=node.defaults[ind])
            edges.extend(ext_edges_)
            self._add_edge(edges, just_arg, arguments, type=self._edge_types["arg"], position_node=node.args[i],
                          scope=self.latest_scope)

        for i in range(len(node.kwonlyargs)):
            ext_edges_, kw_arg = self.parse_arg(node.kwonlyargs[i], default_value=node.kw_defaults[i])
            edges.extend(ext_edges_)
            self._add_edge(edges, kw_arg, arguments, type=self._edge_types["kwonlyarg"], position_node=node.kwonlyargs[i],
                          scope=self.latest_scope)

        if node.kwarg is not None:
            ext_edges_, kwarg = self.parse_arg(node.kwarg)
            edges.extend(ext_edges_)
            self._add_edge(edges, kwarg, arguments, type=self._edge_types["kwarg"], position_node=node.kwarg, scope=self.latest_scope)

        return edges, arguments

        # vararg constains type annotations
        # return self.generic_parse(node, ["args", "vararg", "kwarg", "kwonlyargs", "posonlyargs"])

    def parse_comprehension(self, node):
        edges = []

        cph_name = self._get_node(
            name="comprehension", type=self._node_types["comprehension"], add_random_identifier=True
        )

        target, ext_edges = self.parse_operand(node.target)
        edges.extend(ext_edges)

        self._add_edge(edges, src=target, dst=cph_name, type=self._edge_types["target"], scope=self.latest_scope,
                       position_node=node.target)

        iter_, ext_edges = self.parse_operand(node.iter)
        edges.extend(ext_edges)

        self._add_edge(edges, src=iter_, dst=cph_name, type=self._edge_types["iter"], scope=self.latest_scope,
                       position_node=node.iter)

        for if_ in node.ifs:
            if_n, ext_edges = self.parse_operand(if_)
            edges.extend(ext_edges)
            self._add_edge(edges, src=if_n, dst=cph_name, type=self._edge_types["ifs"], scope=self.latest_scope,
                           position_node=if_)

        return edges, cph_name

    def postprocess(self):
        pass
        # if self._parse_ctx is False:
        #     ctx_edge_type = self._edge_types["ctx"]
        #     ctx_node_type = self._node_types["ctx"]
        #     self._edges = [edge for edge in self._edges if edge.type != ctx_edge_type]
        #     nodes_to_remove = [node.hash_id for node in self._node_pool.values() if node.type != ctx_node_type]
        #     for node_id in nodes_to_remove:
        #         self._node_pool.pop(node_id)

    def _get_offsets(self, edges):
        offsets = edges[["src", "offset_start", "offset_end", "scope"]] \
            .dropna() \
            .rename({
                "src": "node_id", "offset_start": "start", "offset_end": "end" #, "scope": "mentioned_in"
            }, axis=1)

        # assert len(offsets) == offsets["node_id"].nunique()  # there can be several offsets for constants

        edges = edges.drop(["offset_start", "offset_end"], axis=1)
        return edges, offsets

    def to_df(self):

        self.postprocess()

        nodes, edges = nodes_edges_to_df(self._node_pool.values(), self._edges)
        edges, offsets = self._get_offsets(edges)  # TODO should include offsets from table with nodes?
        return nodes, edges, offsets


def make_python_ast_graph(
        source_code, add_reverse_edges=False, save_node_strings=False, add_mention_instances=False,
        graph_builder_class=None, node_edge_definition_class=None, **kwargs
):
    if graph_builder_class is None:
        graph_builder_class = PythonAstGraphBuilder
    if node_edge_definition_class is None:
        node_edge_definition_class = PythonNodeEdgeDefinitions

    g = graph_builder_class(
        source_code, node_edge_definition_class, add_reverse_edges=add_reverse_edges,
        save_node_strings=save_node_strings, add_mention_instances=add_mention_instances,  **kwargs
    )
    return g.to_df()


if __name__ == "__main__":
    for example in PythonCodeExamplesForNodes.examples.values():
        nodes, edges, offsets = make_python_ast_graph(example.lstrip(), add_reverse_edges=False, add_mention_instances=True)
    c = "def f(a=5): a = f([1,2])"
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
