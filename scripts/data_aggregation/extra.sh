conda activate SourceCodeTools

python ../../SourceCodeTools/data/sourcetrail/extract_node_names.py common_nodes.csv
python ../../SourceCodeTools/data/sourcetrail/edge_types_to_int.py common_edges_with_ast.csv
python ../../SourceCodeTools/data/sourcetrail/extract_type_information.py common_nodes_with_ast.csv common_edges_with_types_with_ast.csv -18 -19

conda deactivate