conda activate python38

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


    if [ -f $ENVS_DIR/$dir/source-graph-bodies.csv ]; then
      rm $ENVS_DIR/$dir/source-graph-bodies.csv
    fi
    if [ -f $ENVS_DIR/$dir/nodes_with_ast.csv ]; then
      rm $ENVS_DIR/$dir/nodes_with_ast.csv
    fi
    if [ -f $ENVS_DIR/$dir/edges_with_ast.csv ]; then
      rm $ENVS_DIR/$dir/edges_with_ast.csv
    fi
    if [ -f $ENVS_DIR/$dir/call_seq.csv ]; then
      rm $ENVS_DIR/$dir/call_seq.csv
    fi
    if [ -f $ENVS_DIR/$dir/source-graph-function-variable-pairs.csv ]; then
      rm $ENVS_DIR/$dir/source-graph-function-variable-pairs.csv
    fi


    if [ -f $dir.srctrldb ]; then
      sqlite3 $dir.srctrldb < $SQL_Q
      cd $RUN_DIR
      python verify_files.py $ENVS_DIR/$dir
      python ../sourcetrail/sourcetrail-node-name-merge.py $ENVS_DIR/$dir/nodes.csv
      python parse_bodies.py $ENVS_DIR/$dir
      python get_ast_edges.py $ENVS_DIR/$dir
      python ../code_processing/call_seq_extractor.py $ENVS_DIR/$dir
      python ../code_processing/extract_variable_names.py py $ENVS_DIR/$dir
    else
      echo "Package not indexed"
    fi
  fi
done

conda deactivate