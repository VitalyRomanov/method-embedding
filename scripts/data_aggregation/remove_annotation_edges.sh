conda activate SourceCodeTools

OUT_DIR=$1

sourcetrail-extract-type-information.py $OUT_DIR/common_nodes.csv $OUT_DIR/common_edges_ast_type_as_int.csv $OUT_DIR/ast_types_int_to_str.csv $OUT_DIR/common_edges_annotations.csv $OUT_DIR/common_edges_no_annotations.csv
cp $OUT_DIR/common_edges_no_annotations.csv $OUT_DIR/common_edges.csv

conda deactivate