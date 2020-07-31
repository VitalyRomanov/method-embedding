conda activate SourceCodeTools

python ../../SourceCodeTools/data/sourcetrail/extract_node_names.py common_nodes.csv
python ../../SourceCodeTools/data/sourcetrail/edge_types_to_int.py common_edges_with_ast.csv
python ../../SourceCodeTools/data/sourcetrail/extract_type_information.py common_nodes_with_ast.csv common_edges_with_types_with_ast.csv -18 -19
python process_returns.py
cat edges_component_0_annotations.csv| awk -F"," '{print $3","$4}' > type_links.csv
python join_type_names.csv common_nodes_with_ast.csv type_links.csv
python filter_ast_edges.py

conda deactivate


#import pandas as pd
#>>> ann = pd.read_csv("common_edges_with_types_with_ast_annotations.csv")
#>>> ann = ann.query("type == -19")
#>>> ann.shape
#(3178, 4)
#>>> ann.to_csv("only_returns.csv", index=False)