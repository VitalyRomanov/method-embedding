from SourceCodeTools.code.data.sourcetrail.common import *

import sys

import pandas as pd

pd.options.mode.chained_assignment = None


def get_function_calls(occurrences):
    return occurrences['target_node_id'].dropna().tolist()


def get_function_calls_from_range(occurrences, start, end):
    return occurrences.query(f"start_line >= {start} and end_line <= {end} and occ_type != {DEFINITION_TYPE} and e_type == 'calls'")


def sql_get_function_calls_from_range(occurrences, start, end):
    df = occurrences.query(f"select * from {occurrences.table_name} where start_line >= {start} and end_line <= {end} and occ_type != {DEFINITION_TYPE} and e_type = 'calls'")
    df = df.astype({"source_node_id": "Int32", "target_node_id": "Int32"})
    return df


def extract_call_seq(nodes, edges, source_location, occurrence):

    occurrence_groups = get_occurrence_groups(nodes, edges, source_location, occurrence)

    call_seq = []

    for grp_ind, (file_id, occurrences) in custom_tqdm(
            enumerate(occurrence_groups), message="Extracting call sequences", total=len(occurrence_groups)
    ):

        sql_occurrences = SQLTable(occurrences, "/tmp/sourcetrail_occurrences.db", "occurrences")

        function_definitions = sql_get_function_definitions(sql_occurrences)

        if len(function_definitions):
            for ind, f_def in function_definitions.iterrows():
                local_occurrences = sql_get_function_calls_from_range(sql_occurrences, start=f_def.start_line, end=f_def.end_line)
                local_occurrences = sort_occurrences(local_occurrences)

                all_calls = get_function_calls(local_occurrences)

                for i in range(len(all_calls) - 1):
                    call_seq.append({
                        'src': all_calls[i],
                        'dst': all_calls[i+1]
                    })

        del sql_occurrences

        # print(f"\r{grp_ind}/{len(occurrence_groups)}", end="")
    # print(" " * 30, end="\r")

    if len(call_seq) > 0:
        call_seq = pd.DataFrame(call_seq).astype({
            'src': 'int',
            'dst': 'int'
        })
        return call_seq
    else:
        return None


if __name__ == "__main__":
    working_directory = sys.argv[1]
    source_location = read_source_location(working_directory)
    occurrence = read_occurrence(working_directory)
    nodes = read_nodes(working_directory)
    edges = read_edges(working_directory)

    call_seq = extract_call_seq(nodes, edges, source_location, occurrence)

    if call_seq is not None:
        persist(call_seq, os.path.join(working_directory, filenames["call_seq"]))
