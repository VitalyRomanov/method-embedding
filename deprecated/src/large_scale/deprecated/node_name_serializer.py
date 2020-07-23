# currently not used

def serialize_node_name(name):
    name = name.replace("___", "srstrlnd_space")
    name = name.replace("__", "srstrlnd_pthn_dundrscr")
    name = name.replace("_", "srstrlnd_shrt_spc")
    name = name.replace(")", "srstrlnd_rrbr")
    name = name.replace("(", "srstrlnd_lrbr")
    name = name.replace(">", "srstrlnd_rtbr")
    name = name.replace("<", "srstrlnd_ltbr")
    name = name.replace("?", "srstrlnd_qmark")
    name = name.replace("@", "srstrlnd_at")
    name = name.replace('.', 'srstrlnd_dot')
    name = f"stn_____{name}"
    return name


def deserialize_node_name(name):
    if name.startswith("stn_____"):
        new_name = name.replace("stn_____", "") \
            .replace("srstrlnd_dot", ".")\
            .replace("srstrlnd_at", "@") \
            .replace("srstrlnd_qmark", "?") \
            .replace("srstrlnd_ltbr", "<") \
            .replace("srstrlnd_rtbr", ">") \
            .replace("srstrlnd_lrbr", "(") \
            .replace("srstrlnd_rrbr", ")") \
            .replace("srstrlnd_shrt_spc", "_")\
            .replace("srstrlnd_pthn_dundrscr", "__")\
            .replace("srstrlnd_space", "___")
        return new_name
    else:
        return name

if __name__ == "__main__":

    import pandas as pd
    import sys

    nodes = pd.read_csv(sys.argv[1])

    for ind, row in nodes.iterrows():
        serialized = serialize_node_name(row.serialized_name)
        deserialized = deserialize_node_name(serialized)
        assert deserialized == row.serialized_name, f"""ERROR: serialization strategy failed
{row.serialized_name} != {deserialized}
serialized = {serialized}
deserialized = {deserialized}"""