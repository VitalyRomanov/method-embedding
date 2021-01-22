#%%
import sys

from SourceCodeTools.code.data.sourcetrail.file_utils import *

def filter_ambiguous_edges(edges, ambiguous_edges):

    if ambiguous_edges is None or len(ambiguous_edges) == 0:
        return edges

    # as of january 2020, it seems that all elements in element_component table are ambiguous
    ambiguous_edges = set(ambiguous_edges['element_id'].tolist())

    edges = edges[
        edges["id"].apply(lambda x: x not in ambiguous_edges)
    ]

    return edges

if __name__ == "__main__":
    working_directory = sys.argv[1]

    edges = read_edges(working_directory)
    element_component = read_element_component(working_directory)

    edges = filter_ambiguous_edges(edges, element_component)

    if len(edges) > 0:
        write_edges(edges, working_directory)