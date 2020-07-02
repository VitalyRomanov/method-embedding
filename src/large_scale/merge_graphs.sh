conda activate python37

ENVS_DIR=$1
RUN_DIR=$(pwd)
SQL_Q=$RUN_DIR/extract.sql

#echo $RUN_DIR
#echo $SQL_Q
#echo $ENVS_DIR

#get directories
#https://stackoverflow.com/questions/2107945/how-to-loop-over-directories-in-linux

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    cd $ENVS_DIR/$dir
    echo "Found package $dir"
    if [ -f $dir.srctrldb ]; then
      sqlite3 $dir.srctrldb < $SQL_Q
      cd $RUN_DIR
      python verify_files.py $ENVS_DIR/$dir
      python ../sourcetrail/sourcetrail-node-name-merge.py $ENVS_DIR/$dir/nodes.csv
      python parse_bodies.py $ENVS_DIR/$dir
      python get_ast_edges.py $ENVS_DIR/$dir
    else
      echo "Package not indexed"
    fi
  fi
done



for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      python ../sourcetrail/merge_graphs.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv
      python ../sourcetrail/merge_graphs.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv
    fi
  fi
done

#echo -e "type,source_node_id,target_node_id" > $ENVS_DIR/common_edges.csv
#echo -e "type,source_node_id,target_node_id" > $ENVS_DIR/common_edges_with_ast.csv
#echo -e "id,body,docstring,normalized_body" > $ENVS_DIR/common_bodies.csv
#echo -e "id,body,docstring,normalized_body" > $ENVS_DIR/common_bodies_with_ast.csv

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
#      python map_ids.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir

      python map_id_columns.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/edges.csv $ENVS_DIR/common_edges.csv target_node_id source_node_id
      python map_id_columns.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/source-graph-bodies.csv $ENVS_DIR/common_bodies.csv id

      python map_id_columns.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/edges_with_ast.csv $ENVS_DIR/common_edges_with_ast.csv target_node_id source_node_id
      python map_id_columns.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/source-graph-bodies.csv $ENVS_DIR/common_bodies_with_ast.csv id

#      python map_ids.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/edges.csv $ENVS_DIR/$dir/source-graph-bodies.csv edges_global.csv bodies_global.csv
#      python map_ids.py $ENVS_DIR/common_nodes_with_ast.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/edges_with_ast.csv $ENVS_DIR/$dir/source-graph-bodies.csv edges_with_ast_global.csv bodies_with_ast_global.csv
#      cat $ENVS_DIR/$dir/edges_global.csv >> $ENVS_DIR/common_edges.csv
#      cat $ENVS_DIR/$dir/edges_with_ast_global.csv >> $ENVS_DIR/common_edges_with_ast.csv
#      cat $ENVS_DIR/$dir/bodies_global.csv >> $ENVS_DIR/common_bodies.csv
#      cat $ENVS_DIR/$dir/bodies_with_ast_global.csv >> $ENVS_DIR/common_bodies_with_ast.csv
  #  python ../sourcetrail/merge_graphs.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv
    fi
  fi
done



conda deactivate