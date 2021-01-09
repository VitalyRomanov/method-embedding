from SourceCodeTools.data.sourcetrail.common import *
from SourceCodeTools.data.sourcetrail.common import get_occurrence_groups

import sys

import pandas as pd

pd.options.mode.chained_assignment = None


def get_function_calls(occurrences):
    return occurrences['target_node_id'].dropna().tolist()


def get_function_calls_from_range(occurrences, start, end):
    return occurrences.query(f"start_line >= {start} and end_line <= {end} and occ_type != {DEFINITION_TYPE} and e_type == 'calls'")


def extract_call_seq(nodes, edges, source_location, occurrence):

    occurrence_groups = get_occurrence_groups(nodes, edges, source_location, occurrence)

    call_seq = []

    for grp_ind, (file_id, occurrences) in enumerate(occurrence_groups):

        function_definitions = get_function_definitions(occurrences)

        if len(function_definitions):
            for ind, f_def in function_definitions.iterrows():
                local_occurrences = get_function_calls_from_range(occurrences, start=f_def.start_line, end=f_def.end_line)
                local_occurrences = sort_occurrences(local_occurrences)

                all_calls = get_function_calls(local_occurrences)

                for i in range(len(all_calls) - 1):
                    call_seq.append({
                        'src': all_calls[i],
                        'dst': all_calls[i+1]
                    })

        print(f"\r{grp_ind}/{len(occurrence_groups)}", end="")
    print(" " * 30, end="\r")

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
        persist(call_seq, os.path.join(working_directory, filenames["call-seq"]))
