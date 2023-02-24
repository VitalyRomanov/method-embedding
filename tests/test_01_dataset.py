from pathlib import Path

root_path = Path(__file__).parent.parent


def test_dataset_creator():
    from SourceCodeTools.code.data.sourcetrail.DatasetCreator2 import DatasetCreator

    indexed_environments = root_path.joinpath("res", "python_testdata", "example_environment")
    bpe_tokenizer = root_path.joinpath("examples", "sentencepiece_bpe.model")
    output_directory = root_path.joinpath("res", "python_testdata", "example_graph")
    dataset = DatasetCreator(
        path=indexed_environments, lang="python",
        bpe_tokenizer=bpe_tokenizer, create_subword_instances=False,
        connect_subwords=False, only_with_annotations=False,
        do_extraction=True, visualize=True, track_offsets=True, remove_type_annotations=True,
        recompute_l2g=False
    )
    dataset.merge(output_directory)


def test_type_annotation_dataset_envs():
    indexed_environments = root_path.joinpath("res", "python_testdata", "example_environment")
    output = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast", "annotations_dataset_no_default.json"
    )
    from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_dataset import \
        create_from_environments
    create_from_environments(**{
        "dataset_format": "envs",
        "dataset_path": indexed_environments,
        "output_path": output,
        "format": "json",
        "remove_default": True,
        "global_nodes": None,
        "require_labels": True
    })


def test_type_annotation_dataset_dataset():
    dataset_path = root_path.joinpath("res", "python_testdata", "example_graph", "with_ast")
    output = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast", "annotations_dataset_no_default_from_dataset.json"
    )
    from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_dataset import create_from_dataset
    create_from_dataset(**{
        "dataset_format": "dataset",
        "dataset_path": dataset_path,
        "output_path": output,
        "format": "json",
        "remove_default": True,
        "global_nodes": None,
        "require_labels": True
    })


def test_map_args_to_mentions():
    working_directory = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast"
    )
    output = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast", "annotations_dataset_no_default_args_mapped.json"
    )
    dataset_file = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast", "annotations_dataset_no_default.json"
    )
    from SourceCodeTools.code.data.type_annotation_dataset.map_args_to_mentions import map_args_to_mention
    map_args_to_mention(
        working_directory, output, dataset_file=dataset_file
    )


def test_split_dataset():
    data_path = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast", "annotations_dataset_no_default_args_mapped.json"
    )

    from SourceCodeTools.code.data.type_annotation_dataset.split_dataset import split_dataset
    split_dataset(
        data_path, random_seed=42, min_entity_count=0, name_suffix="no_default_args_mapped"
    )


def test_create_type_annotation_aware_graph_partition():
    working_directory = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast"
    )
    type_annotation_test_set = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast",
        "type_prediction_dataset_no_default_args_mapped_test.json"
    )
    output_path = root_path.joinpath(
        "res", "python_testdata", "example_graph", "with_ast",
        "partition_type_prediction.json"
    )
    from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_aware_graph_partition import \
        create_type_annotation_aware_graph_partition
    create_type_annotation_aware_graph_partition(working_directory, type_annotation_test_set, output_path)
