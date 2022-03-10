import hashlib

from SourceCodeTools.code.data.file_utils import *
from tqdm import tqdm

DEFINITION_TYPE = 1
UNRESOLVED_SYMBOL = "unsolved_symbol"


def get_occurrence_groups(nodes, edges, source_location, occurrence):
    """
    Group nodes based on file id. Return dataset that contains node ids and their offsets in the source code.
    :param nodes: dataframe with nodes
    :param edges: dataframe with edges
    :param source_location: dataframe with sources
    :param occurrence: dataframe with with offsets
    :return: Result of group by file id
    """
    edges = edges.rename(columns={'type': 'e_type'})
    edges = edges.query("id >= 0")  # filter reverse edges

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


def sql_get_function_definitions(occurrences):
    return occurrences.query(
        f"select * from {occurrences.table_name} where occ_type = {DEFINITION_TYPE} and (type = 'function' or type = 'class_method')")
    # return occurrences.query(f"occ_type == {DEFINITION_TYPE} and (type == 4096 or type == 8192)")


def sql_get_occurrences_from_range(occurrences, start, end) -> pd.DataFrame:
    df = occurrences.query(
        f"select * from {occurrences.table_name} where start_line >= {start} and end_line <= {end} and occ_type != {DEFINITION_TYPE} and start_line = end_line")
    df = df.astype({"source_node_id": "Int32", "target_node_id": "Int32"})
    return df
