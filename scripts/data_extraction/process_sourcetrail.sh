conda activate SourceCodeTools

ENVS_DIR=$(realpath $1)
RUN_DIR=$(realpath $(dirname "$0"))
SQL_Q=$(realpath $RUN_DIR/extract.sql)

#echo $RUN_DIR
#echo $SQL_Q
#echo $ENVS_DIR

#get directories
#https://stackoverflow.com/questions/2107945/how-to-loop-over-directories-in-linux

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
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


    if [ -f $ENVS_DIR/$dir/$dir.srctrldb ]; then
      cd $ENVS_DIR/$dir
      sqlite3 $ENVS_DIR/$dir/$dir.srctrldb < $SQL_Q
      cd $RUN_DIR
      sourcetrail-verify-files.py $ENVS_DIR/$dir
      sourcetrail-node-name-merge.py $ENVS_DIR/$dir/nodes.csv
      sourcetrail-parse-bodies.py $ENVS_DIR/$dir
      sourcetrail-ast-edges.py $ENVS_DIR/$dir
      sourcetrail-call-seq-extractor.py $ENVS_DIR/$dir
      sourcetrail-extract-variable-names.py py $ENVS_DIR/$dir
    else
      echo "Package not indexed"
    fi
  fi
done

conda deactivate