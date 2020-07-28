conda activate SourceCodeTools

ENVS_DIR=$1
#RUN_DIR=$(pwd)
#SQL_Q=$RUN_DIR/extract.sql

#echo $RUN_DIR
#echo $SQL_Q
#echo $ENVS_DIR

#get directories
#https://stackoverflow.com/questions/2107945/how-to-loop-over-directories-in-linux

#for dir in $(ls $ENVS_DIR); do
#  if [ -d $ENVS_DIR/$dir ]; then
#    cd $ENVS_DIR/$dir
#    echo "Found package $dir"
#    if [ -f $dir.srctrldb ]; then
#      sqlite3 $dir.srctrldb < $SQL_Q
#      cd $RUN_DIR
#      python sourcetrail-verify-files.py $ENVS_DIR/$dir
#      python ../sourcetrail/sourcetrail-node-name-merge.py $ENVS_DIR/$dir/nodes.csv
#      python sourcetrail-parse-bodies.py $ENVS_DIR/$dir
#      python sourcetrail-ast-edges.py $ENVS_DIR/$dir
#      python ../code_processing/sourcetrail-call-seq-extractor.py $ENVS_DIR/$dir
#      python ../code_processing/sourcetrail-extract-variable-names.py py $ENVS_DIR/$dir
#    else
#      echo "Package not indexed"
#    fi
#  fi
#done

#rm $ENVS_DIR/common*
#
#for dir in $(ls $ENVS_DIR); do
#  if [ -d $ENVS_DIR/$dir ]; then
#    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
#      python ../sourcetrail/sourcetrail-merge-graphs.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv
#    fi
#  fi
#done

rm $ENVS_DIR/common*

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      sourcetrail-merge-graphs.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv
    fi
  fi
done


for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      sourcetrail-map-id-columns-only-annotations.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/edges_with_ast.csv $ENVS_DIR/common_edges_with_ast.csv target_node_id source_node_id
      sourcetrail-map-id-columns.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/source-graph-bodies.csv $ENVS_DIR/common_bodies_with_ast.csv id
      sourcetrail-map-id-columns.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/source-graph-function-variable-pairs.csv $ENVS_DIR/common-function-variable-pairs_with_ast.csv src
      sourcetrail-map-id-columns.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/call_seq.csv $ENVS_DIR/common-call-seq_with_ast.csv src dst
    fi
  fi
done



conda deactivate
