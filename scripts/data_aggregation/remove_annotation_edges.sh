conda activate SourceCodeTools

sourcetrail-extract-node-names.py common_nodes.csv
sourcetrail-edge-types-to-int.py common_edges_with_ast.csv
sourcetrail-extract-type-information.py common_nodes_with_ast.csv common_edges_with_types_with_ast.csv common_type_maps_with_ast.csv

conda deactivate