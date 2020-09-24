conda activate SourceCodeTools

ENVS_DIR=$1
OUT_DIR=$2

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      sourcetrail-node-local2global.py $OUT_DIR/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/local2global_with_ast.csv
      sourcecodetools-extract-type-annotations.py $ENVS_DIR/$dir/bodies_with_replacements.csv $OUT_DIR/functions_with_annotations.jsonl $ENVS_DIR/$dir/local2global_with_ast.csv $OUT_DIR/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv
    fi
  fi
done

conda deactivate
