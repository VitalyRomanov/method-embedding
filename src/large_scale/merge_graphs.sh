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
      ans=$(python ../sourcetrail/sourcetrail-node-name-merge.py $ENVS_DIR/$dir/nodes.csv)
      if [ -n "$ans" ]; then
        echo "Could not extract nodes: $dir"
      fi
    else
      echo "Package not indexed"
    fi
  fi
done



for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      ans=$(python ../sourcetrail/merge_graphs.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv)
    fi
  fi
done

echo -e "type,source_node_id,target_node_id" > $ENVS_DIR/common_edges.csv
echo -e "id,body,docstring,normalized_body" > $ENVS_DIR/common_bodies.csv
for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      python parse_bodies.py $ENVS_DIR/$dir
      python map_ids.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir
      cat $ENVS_DIR/$dir/edges_global.csv >> $ENVS_DIR/common_edges.csv
      cat $ENVS_DIR/$dir/source-graph-bodies_global.csv >> $ENVS_DIR/common_bodies.csv
  #  python ../sourcetrail/merge_graphs.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv
    fi
  fi
done



conda deactivate