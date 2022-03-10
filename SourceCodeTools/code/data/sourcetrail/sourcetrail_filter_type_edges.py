import sys
from os.path import join

from SourceCodeTools.code.data.ast_graph.filter_type_edges import filter_type_edges
from SourceCodeTools.code.data.file_utils import *


def main():
    working_directory = sys.argv[1]
    nodes = unpersist(join(working_directory, "nodes.bz2"))
    edges = unpersist(join(working_directory, "edges.bz2"))
    out_annotations = sys.argv[2]
    out_no_annotations = sys.argv[3]

    no_annotations, annotations = filter_type_edges(nodes, edges)

    persist(annotations, out_annotations)
    persist(no_annotations, out_no_annotations)

if __name__ == "__main__":
    main()