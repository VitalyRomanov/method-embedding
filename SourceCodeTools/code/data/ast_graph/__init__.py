
def source_code_to_graph(
        source_code, variety, bpe_tokenizer_path=None, reverse_edges=False, mention_instances=False
):
    if variety == "plain_ast":
        raise NotImplementedError()
    elif variety == "with_mention":
        from SourceCodeTools.code.data.ast_graph.build_ast_graph import ast_graph_for_single_example
        nodes, edges, _ = ast_graph_for_single_example(
            source_code=source_code, bpe_tokenizer_path=None, create_subword_instances=False, connect_subwords=False,
            track_offsets=False, reverse_edges=reverse_edges
        )
    elif variety == "cf":
        from SourceCodeTools.code.data.ast_graph.build_ast_graph import ast_graph_for_single_example
        from SourceCodeTools.code.ast.python_ast_cf import AstGraphGenerator
        nodes, edges, _ = ast_graph_for_single_example(
            source_code=source_code, bpe_tokenizer_path=None, create_subword_instances=False, connect_subwords=False,
            track_offsets=False, reverse_edges=reverse_edges,
            ast_generator_base_class=AstGraphGenerator
        )
    elif variety == "new_with_mention":
        from SourceCodeTools.code.ast.python_ast3 import make_python_ast_graph
        nodes, edges = make_python_ast_graph(
            source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances
        )
    elif variety == "new_cf":
        from SourceCodeTools.code.ast.python_ast3_cf import make_python_cf_graph
        nodes, edges = make_python_cf_graph(
            source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances
        )
    elif variety == "new_with_subwords":
        from SourceCodeTools.code.ast.python_graph_subwords import make_python_graph_with_subwords
        nodes, edges = make_python_graph_with_subwords(
            source_code, add_reverse_edges=reverse_edges, add_mention_instances=mention_instances,
            bpe_tokenizer_path=bpe_tokenizer_path
        )
    else:
        raise ValueError()

    return nodes, edges