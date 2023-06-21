from pathlib import Path

root_path = Path(__file__).parent.parent


def test_dataset_var_misuse_creator():
    from SourceCodeTools.code.data.ast_graph.build_ast_graph import AstDatasetCreator

    source_code_table_path = root_path.joinpath("res", "python_testdata", "cubert_benchmark", "variable_misuse.csv")
    output_directory = root_path.joinpath("res", "python_testdata", "var_misuse_graph")

    bpe_tokenizer = root_path.joinpath("examples", "sentencepiece_bpe.model")
    dataset = AstDatasetCreator(
        path=source_code_table_path, lang="python",
        bpe_tokenizer=bpe_tokenizer, create_subword_instances=False,
        connect_subwords=False, only_with_annotations=False,
        do_extraction=True, visualize=False, track_offsets=True, remove_type_annotations=True,
        recompute_l2g=False, chunksize=10000, keep_frac=1.0, seed=42,
        create_mention_instances=False, graph_format_version="v3.5"
    )
    dataset.merge(output_directory)


def test_variable_misuse_node_level_labels():
    from SourceCodeTools.code.data.cubert_python_benchmarks.variable_misuse_node_level_labels import get_node_labels
    input_path = root_path.joinpath("res", "python_testdata", "var_misuse_graph", "with_ast")

    get_node_labels(input_path)


def test_extract_partition():
    from SourceCodeTools.code.data.cubert_python_benchmarks.extract_partition import extract_partitions
    input_path = root_path.joinpath("res", "python_testdata", "var_misuse_graph", "with_ast", "common_filecontent.json")

    extract_partitions(input_path)


def test_make_edge_level_labels():
    from SourceCodeTools.code.data.cubert_python_benchmarks.make_edge_level_labels import create_edge_labels
    input_path = root_path.joinpath("res", "python_testdata", "var_misuse_graph", "with_ast")

    create_edge_labels(dataset_directory=input_path, use_mention_instances=False)


def test_create_text_dataset():
    from SourceCodeTools.code.data.cubert_python_benchmarks.create_text_dataset import create_text_dataset
    input_path = root_path.joinpath("res", "python_testdata", "var_misuse_graph", "with_ast")

    create_text_dataset(input_path)