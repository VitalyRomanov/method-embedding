conda activate python37

ENVS_DIR=$1


for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      python $ENVS_DIR/$dir/source-graph-bodies_global.csv $ENVS_DIR/functions_with_annotations.jsonl
    fi
  fi
done



conda deactivate
