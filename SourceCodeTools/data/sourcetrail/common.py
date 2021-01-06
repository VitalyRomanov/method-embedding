from SourceCodeTools.data.sourcetrail.file_utils import *


DEFINITION_TYPE = 1
UNRESOLVED_SYMBOL = "unsolved_symbol"


def get_occurrence_groups(working_directory):
    source_location = read_source_location(working_directory)
    occurrence = read_occurrence(working_directory)
    nodes = read_nodes(working_directory)
    edges = read_edges(working_directory)

    # merge nodes and edges, some references in code point to edges, not to nodes
    node_edge = pd.concat([nodes, edges], sort=False).astype({"target_node_id": "Int32", "source_node_id": "Int32"})
    assert len(node_edge["id"].unique()) == len(node_edge), f"{len(node_edge['id'].unique())} != {len(node_edge)}"

    # rename columns
    source_location.rename(columns={'id': 'source_location_id', 'type': 'occ_type'}, inplace=True)
    node_edge.rename(columns={'id': 'element_id'}, inplace=True)

    # join tables
    occurrences = occurrence.merge(source_location, on='source_location_id', )
    nodes = node_edge.merge(occurrences, on='element_id')
    occurrence_groups = nodes.groupby("file_node_id")

    return occurrence_groups


def sort_occurrences(occurrences):
    return occurrences.sort_values(by=["start_line", "end_column"], ascending=[True, False])


def get_function_definitions(occurrences):
    return occurrences.query(f"occ_type == {DEFINITION_TYPE} and (type == 'function' or type == 'class_method')")
    # return occurrences.query(f"occ_type == {DEFINITION_TYPE} and (type == 4096 or type == 8192)")


def get_occurrences_from_range(occurrences, start, end) -> pd.DataFrame:
    return occurrences.query(
        f"start_line >= {start} and end_line <= {end} and occ_type != {DEFINITION_TYPE} and start_line == end_line")


def create_node_repr(nodes):
    return list(zip(nodes['serialized_name'], nodes['type']))


def map_id_columns(df, column_names, mapper):
    df = df.copy()
    for col in column_names:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: mapper.get(x, -1))
    return df


def merge_with_file_if_exists(df, merge_with_file):
    if os.path.isfile(merge_with_file):
        original_data = unpersist(merge_with_file)
        data = pd.concat([original_data, df], axis=0)
    else:
        data = df
    return data


def create_local_to_global_id_map(local_nodes, global_nodes):
    local_nodes = local_nodes.copy()
    global_nodes = global_nodes.copy()

    global_nodes['node_repr'] = create_node_repr(global_nodes)
    local_nodes['node_repr'] = create_node_repr(local_nodes)

    rev_id_map = dict(zip(
        global_nodes['node_repr'].tolist(), global_nodes['id'].tolist()
    ))
    id_map = dict(zip(
        local_nodes["id"].tolist(), map(
            lambda x: rev_id_map[x], local_nodes["node_repr"].tolist()
        )
    ))

    return id_map