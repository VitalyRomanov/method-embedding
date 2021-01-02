from SourceCodeTools.data.sourcetrail.common import *
from SourceCodeTools.data.sourcetrail.common import get_occurrence_groups

import sys

import pandas as pd

pd.options.mode.chained_assignment = None


def get_function_calls(occurrences):
    return list(map(lambda x: not pd.isna, occurrences['target_node_id'].tolist()))


def main(working_directory):

    occurrence_group = get_occurrence_groups(working_directory)

    call_seq = []

    for grp_ind, (file_id, occurrences) in enumerate(occurrence_group):

        function_definitions = get_function_definitions(occurrences)

        if len(function_definitions):
            for ind, f_def in function_definitions.iterrows():
                local_occurrences = get_occurrences_from_range(occurrences, start=f_def.start_line, end=f_def.end_line)
                local_occurrences = sort_occurrences(local_occurrences)

                all_calls = get_function_calls(local_occurrences)

                for i in range(len(all_calls) - 1):
                    call_seq.append({
                        'src': all_calls[i],
                        'dst': all_calls[i+1]
                    })

        print(f"\r{grp_ind}/{len(occurrence_group)}", end="")
    print(" " * 30, end="\r")

    call_seq = pd.DataFrame(call_seq).astype({
        'src': 'int',
        'dst': 'int'
    })

    persist(call_seq, os.path.join(working_directory, filenames["call-seq"]))


if __name__ == "__main__":
    main(sys.argv[1])
