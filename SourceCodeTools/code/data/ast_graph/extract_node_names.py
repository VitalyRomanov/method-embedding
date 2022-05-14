def extract_node_names(nodes, min_count):

    data = nodes.copy().rename({"id": "src", "serialized_name": "dst", "name": "dst"}, axis=1)

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