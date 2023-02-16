import logging
from functools import lru_cache

from SourceCodeTools.code.ast.python_ast3 import PythonAstGraphBuilder, make_python_ast_graph, PythonNodeEdgeDefinitions
from SourceCodeTools.code.data.ast_graph.draw_graph import visualize
from SourceCodeTools.nlp import create_tokenizer


@lru_cache
def make_node_edge_definitions_with_subwords(base_class):
    class PythonGraphWithSubwordsEdgeDefinitions(base_class):
        node_type_enum_initialized = False
        edge_type_enum_initialized = False
        shared_node_types_initialized = False

        node_type_enum = None
        edge_type_enum = None

        leaf_types = None
        named_leaf_types = None
        tokenizable_types_and_annotations = None
        shared_node_types = None

        extra_node_types = base_class.extra_node_types | {
            "subword", "subword_instance"
        }

        extra_edge_types = base_class.extra_edge_types | {
            "subword", "subword_instance", "next_subword", "prev_subword"
        }

        reverse_edge_exceptions = {
            **base_class.reverse_edge_exceptions, **{
                "subword": None,
                "subword_instance": None,
                "next_subword": None,
                "prev_subword": None
            }
        }

        @classmethod
        def get_tokenizable_types_and_annotations(cls):
            if not cls.shared_node_types_initialized:
                cls._initialize_shared_nodes()
            return cls.tokenizable_types_and_annotations

        @classmethod
        def _initialize_shared_nodes(cls):
            node_types_enum = cls.make_node_type_enum()
            subword_types = {"subword"}
            annotation_types = {"type_annotation"}
            tokenizable_types = {"Name", "#attr#", "#keyword#"}

            cls.shared_node_types = {
                node_types_enum[type_] for type_ in
                (
                        {type_.name for type_ in cls.__base__.get_shared_node_types()} | subword_types
                )
            }

            cls.tokenizable_types_and_annotations = {
                node_types_enum[type_] for type_ in (annotation_types | tokenizable_types)
            }

            cls.shared_node_types_initialized = True

        @classmethod
        def is_shared_name_type(cls, name, type):
            if not cls.shared_node_types_initialized:
                cls._initialize_shared_nodes()

            if type in cls.shared_node_types or \
                    (type == cls.node_type_enum["subword_instance"] and "0x" not in name):
                return True
            return False


    return PythonGraphWithSubwordsEdgeDefinitions


@lru_cache
def make_graph_bilder_class(base_class):
    class PythonWithSubwordsGraphBuilder(base_class):
        def __init__(
                self, *args, bpe_tokenizer_path=None, connect_subwords=False,
                create_subword_instances=False, **kwargs
        ):
            super(PythonWithSubwordsGraphBuilder, self).__init__(*args, **kwargs)

            if bpe_tokenizer_path is not None:
                tokenize = create_tokenizer("bpe", bpe_path=bpe_tokenizer_path)
                self._bpe = lambda x: tokenize(x)
            else:
                logging.info("No tokenizer binary provided for bpe tokenizer. Using CodeBERT.")
                tokenize = create_tokenizer("codebert")
                self._bpe = lambda x: tokenize(x).tokens[1:-1]

            self.create_subword_instances = create_subword_instances
            self.connect_subwords = connect_subwords

        def postprocess(self):
            super().postprocess()
            self.replace_mentions_with_subwords()

        def replace_mentions_with_subwords(self):
            if self.create_subword_instances:
                def produce_subw_edges(new_edges, subwords, dst):
                    return self.produce_subword_edges_with_instances(new_edges, subwords, dst)
            else:
                def produce_subw_edges(new_edges, subwords, dst):
                    return self.produce_subword_edges(new_edges, subwords, dst, self.connect_subwords)

            new_edges = []
            for edge in self._edges:
                if edge.type == self._edge_types["local_mention"]:
                    subwords = self._bpe(self._node_pool[edge.src].name)
                    produce_subw_edges(new_edges, subwords, edge.dst)
                elif self._node_pool[edge.src].type in self._graph_definitions.get_tokenizable_types_and_annotations():
                    new_edges.append(edge)
                    new_dst = edge.src
                    subwords = self._bpe(self._node_pool[new_dst].name)
                    produce_subw_edges(new_edges, subwords, new_dst)
                else:
                    new_edges.append(edge)

            self._edges = new_edges

        def connect_prev_next_subwords(self, edges, current, prev_subw, next_subw):
            if next_subw is not None:
                edges.append(self._make_edge(
                    src=current,
                    dst=next_subw,
                    type=self._edge_types["next_subword"]
                ))
            if prev_subw is not None:
                edges.append(self._make_edge(
                    src=current,
                    dst=prev_subw,
                    type=self._edge_types["prev_subword"]
                ))

        def produce_subword_edges(self, new_edges, subwords, dst, connect_subwords=False):
            subwords = [self._get_node(name=x, type=self._node_types["subword"]) for x in subwords]

            for ind, subword in enumerate(subwords):
                new_edges.append(self._make_edge(
                    src=subword,
                    dst=dst,
                    type=self._edge_types["subword"]
                ))
                if connect_subwords:
                    self.connect_prev_next_subwords(new_edges, subword, subwords[ind - 1] if ind > 0 else None,
                                                    subwords[ind + 1] if ind < len(subwords) - 1 else None)

        def produce_subword_edges_with_instances(self, new_edges, subwords, dst, connect_subwords=True):
            subwords = [self._get_node(name=x, type=self._node_types["subword"]) for x in subwords]
            dst_name = self._node_pool[dst].name
            instances = [
                self._get_node(
                    name=self._node_pool[x].name + "@" + dst_name,
                    type=self._node_types["subword_instance"]
                ) for x in subwords
            ]
            for ind, subword in enumerate(subwords):
                subword_instance = instances[ind]
                new_edges.append(self._make_edge(
                    src=subword,
                    dst=subword_instance,
                    type=self._edge_types["subword_instance"]
                ))
                new_edges.append(self._make_edge(
                    src=subword_instance,
                    dst=dst,
                    type=self._edge_types["subword"]
                ))
                if connect_subwords:
                    self.connect_prev_next_subwords(
                        new_edges, subword_instance,
                        instances[ind - 1] if ind > 0 else None,
                        instances[ind + 1] if ind < len(instances) - 1 else None
                    )
    return PythonWithSubwordsGraphBuilder


def make_python_graph_with_subwords(
        source_code, add_reverse_edges=False, save_node_strings=False, add_mention_instances=False,
        connect_subwords=False, create_subword_instances=False,
        graph_builder_base_class=None, node_edge_base_definition_class=None,
        **kwargs
):
    if graph_builder_base_class is None:
        graph_builder_base_class = PythonAstGraphBuilder
    if node_edge_base_definition_class is None:
        node_edge_base_definition_class = PythonNodeEdgeDefinitions

    graph_builder_class = make_graph_bilder_class(graph_builder_base_class)
    node_edge_definition_class = make_node_edge_definitions_with_subwords(node_edge_base_definition_class)

    return make_python_ast_graph(
        source_code, add_reverse_edges=add_reverse_edges, save_node_strings=save_node_strings,
        add_mention_instances=add_mention_instances,
        graph_builder_class=graph_builder_class,
        node_edge_definition_class=node_edge_definition_class,
        connect_subwords=connect_subwords, create_subword_instances=create_subword_instances, **kwargs
    )


if __name__ == "__main__":
    # for example in PythonCodeExamplesForNodes.examples.values():
    #     nodes, edges = make_python_ast_graph_with_subwords(example.lstrip(), add_reverse_edges=False)
    #     visualize(nodes, edges, "/Users/LTV/1.png")
    c = "def f(and_this_is_how_it_starts=5): f(a=4)"
    nodes, edges = make_python_graph_with_subwords(c.lstrip(), add_reverse_edges=True)
    visualize(nodes, edges, "/Users/LTV/1.png")