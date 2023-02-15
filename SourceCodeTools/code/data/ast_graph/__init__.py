
def source_code_to_graph(
        source_code, variety, bpe_tokenizer_path=None, reverse_edges=False, mention_instances=False
):
    if variety == "v1.0":
        raise NotImplementedError()
    elif variety == "v2.5":
        from SourceCodeTools.code.data.ast_graph.build_ast_graph import ast_graph_for_single_example
        nodes, edges, offsets = ast_graph_for_single_example(
            source_code=source_code, bpe_tokenizer_path=bpe_tokenizer_path, create_subword_instances=False, connect_subwords=False,
            track_offsets=False, reverse_edges=reverse_edges
        )
    elif variety == "v1.0_control_flow":
        from SourceCodeTools.code.data.ast_graph.build_ast_graph import ast_graph_for_single_example
        from SourceCodeTools.code.ast.python_ast_cf import AstGraphGenerator
        nodes, edges, offsets = ast_graph_for_single_example(
            source_code=source_code, bpe_tokenizer_path=None, create_subword_instances=False, connect_subwords=False,
            track_offsets=False, reverse_edges=reverse_edges,
            ast_generator_base_class=AstGraphGenerator
        )
    elif variety == "v3.5":
        from SourceCodeTools.code.ast.python_ast3 import make_python_ast_graph as make_python_ast_graph_without_subwords
        from SourceCodeTools.code.ast.python_graph_subwords import make_python_graph_with_subwords
        if bpe_tokenizer_path is None:
            nodes, edges, offsets = make_python_ast_graph_without_subwords(
                source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances
            )
        else:
            nodes, edges, offsets = make_python_graph_with_subwords(
                source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances,
                bpe_tokenizer_path=bpe_tokenizer_path
            )
    elif variety == "v3.5_control_flow":
        from SourceCodeTools.code.ast.python_ast3_cf import make_python_cf_graph
        from SourceCodeTools.code.ast.python_graph_subwords import make_python_graph_with_subwords
        if bpe_tokenizer_path is None:
            nodes, edges, offsets = make_python_cf_graph(
                source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances
            )
        else:
            from SourceCodeTools.code.ast.python_ast3_cf import PythonCFGraphBuilder
            from SourceCodeTools.code.ast.python_ast3_cf import PythonNodeEdgeCFDefinitions
            nodes, edges, offsets = make_python_graph_with_subwords(
                source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances,
                bpe_tokenizer_path=bpe_tokenizer_path, graph_builder_base_class=PythonCFGraphBuilder,
                node_edge_base_definition_class=PythonNodeEdgeCFDefinitions
            )
    else:
        raise ValueError()

    return {
        "nodes": nodes,
        "edges": edges,
        "offsets": offsets
    }