conda activate python37

ENVS_DIR=$1
RUN_DIR=$(pwd)
SQL_Q=$RUN_DIR/extract.sql

#echo $RUN_DIR
#echo $SQL_Q
#echo $ENVS_DIR

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    cd $ENVS_DIR/$dir
    echo "Found package $dir"
    if [ -f $dir.srctrldb ]; then
      sqlite3 $dir.srctrldb < $SQL_Q
      cd $RUN_DIR
      python ../sourcetrail/sourcetrail-node-name-merge.py $ENVS_DIR/$dir/nodes.csv
    else
      echo "Package not indexed"
    fi
  fi
done



for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    python ../sourcetrail/merge_graphs.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv
  fi
done

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
#  python ../sourcetrail/merge_graphs.py $ENVS_DIR/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv
  fi
done



conda deactivate