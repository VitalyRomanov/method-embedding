import sys

from SourceCodeTools.code.data.file_utils import *


def get_node_name(full_name):
    """
    Used for processing java nodes
    """
    return full_name.split(".")[-1].split("___")[0]


def extract_node_names(nodes, min_count):


    # some cells are empty, probably because of empty strings in AST
    # data = nodes.dropna(axis=0)
    # data = data[data['type'] != 262144]
    data = nodes
    data['src'] = data['id']
    data['dst'] = data['serialized_name'].apply(get_node_name)

    corrected_names = []
    for type_, name_ in data[["type", "dst"]].values:
        if type_ == "mention":
            corrected_names.append(name_.split("@")[0])
        else:
            corrected_names.append(name_)

    data["dst"] = corrected_names

    def not_contains(name):
        return "0x" not in name

    data = data[
        data["dst"].apply(not_contains)
    ]

    counts = data['dst'].value_counts()

    data['counts'] = data['dst'].apply(lambda x: counts[x])
    data = data.query(f"counts >= {min_count}")

    if len(data) > 0:
        return data[['src', 'dst']]
    else:
        return None


if __name__ == "__main__":
    nodes_path = sys.argv[1]
    out_path = sys.argv[2]
    try:
        min_count = int(sys.argv[3])
    except:
        min_count = 1

    nodes = unpersist_or_exit(nodes_path)

    names = extract_node_names(nodes, min_count)

    if names is not None:
        persist(names, out_path)
