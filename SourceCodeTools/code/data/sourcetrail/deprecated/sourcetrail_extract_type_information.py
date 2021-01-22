import sys

from SourceCodeTools.code.data.sourcetrail.file_utils import *


def main():
    edges = unpersist(sys.argv[1])
    out_annotations = sys.argv[2]
    out_no_annotations = sys.argv[3]

    annotations = edges.query(f"type == 'annotation_for' or type == 'returned_by'")
    no_annotations = edges.query(f"type != 'annotation_for' and type != 'returned_by'")

    persist(annotations, out_annotations)
    persist(no_annotations, out_no_annotations)

if __name__ == "__main__":
    main()