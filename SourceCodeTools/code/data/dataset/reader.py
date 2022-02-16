from pathlib import Path

from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.code.annotator_utils import source_code_graph_alignment


def load_data(node_path, edge_path, rename_columns=True):
    nodes = unpersist(node_path)
    edges = unpersist(edge_path)

    nodes = nodes.astype({
        'type': 'category', "serialized_name": "string", "mentioned_in": "Int64", "string": "string"
    })
    edges = edges.astype({
        'type': 'category', "mentioned_in": "Int64"
    })

    if rename_columns:
        nodes = nodes.rename(mapper={
            'serialized_name': 'name'
        }, axis=1)
        edges = edges.rename(mapper={
            'source_node_id': 'src',
            'target_node_id': 'dst'
        }, axis=1)

    return nodes, edges


def load_graph(dataset_directory, rename_columns=True):
    dataset_path = Path(dataset_directory)
    nodes = dataset_path.joinpath("common_nodes.bz2")
    edges = dataset_path.joinpath("common_edges.bz2")

    return load_data(nodes, edges)


def load_aligned_source_code(dataset_directory, tokenizer="codebert"):
    dataset_path = Path(dataset_directory)

    files = unpersist(dataset_path.joinpath("common_filecontent.bz2")).rename({"id": "file_id"}, axis=1)

    content = dict(zip(zip(files["package"], files["file_id"]), files["filecontent"]))
    pd_offsets = unpersist(dataset_path.joinpath("common_offsets.bz2"))

    seen = set()

    source_codes = []
    offsets = []

    for group, data in pd_offsets.groupby(by=["package", "file_id"]):
        source_codes.append(content[group])
        offsets.append(list(zip(data["start"], data["end"], data["node_id"])))
        seen.add(group)

    for key, val in content.items():
        if key not in seen:
            source_codes.append(val)
            offsets.append([])

    return source_code_graph_alignment(source_codes, offsets, tokenizer=tokenizer)


if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    for tokens, node_tags in load_aligned_source_code(data_path):
        for t, tt in zip(tokens, node_tags):
            print(t, tt, sep="\t")
        print()

