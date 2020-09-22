conda activate SourceCodeTools

# prepare data to classify nodes
#
sourcetrail-extract-node-names.py common_nodes.csv
sourcetrail-edge-types-to-int.py common_edges_with_ast.csv
sourcetrail-extract-type-information.py common_nodes_with_ast.csv common_edges_with_types_with_ast.csv common_type_maps_with_ast.csv
python process_returns.py common_nodes_with_ast.csv common_edges_with_types_with_ast.csv common_edges_with_types_with_ast_annotations.csv
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