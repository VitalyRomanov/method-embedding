#!/usr/bin/env python
import pandas as pd
import sys, os

def verify_files(working_dir):

    fileheaders = {
        "edges.csv": "id,type,source_node_id,target_node_id\n",
        "nodes.csv": "id,type,serialized_name\n",
        "element_component.csv": "id,element_id,type,data\n",
        "source_location.csv": "id,file_node_id,start_line,start_column,end_line,end_column,type\n",
        "occurrence.csv": "element_id,source_location_id\n",
        "filecontent.csv": "id,content\n"
    }

    for filename, header in fileheaders.items():
        file_path = os.path.join(working_dir, filename)
        try:
            pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            with open(file_path, "w") as sink:
                sink.write(header)


if __name__ == "__main__":
    working_dir = sys.argv[1]
    verify_files(working_dir)