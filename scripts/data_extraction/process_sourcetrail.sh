conda activate SourceCodeTools

ENVS_DIR=$(realpath "$1")
RUN_DIR=$(realpath "$(dirname "$0")")
SQL_Q=$(realpath "$RUN_DIR/extract.sql")

#echo $RUN_DIR
#echo $SQL_Q
#echo $ENVS_DIR

#get directories
#https://stackoverflow.com/questions/2107945/how-to-loop-over-directories-in-linux

for dir in "$ENVS_DIR"/*; do
  if [ -d "$dir" ]; then
    package_name="$(basename "$dir")"

    echo "Found package $package_name"

    if [ -f "$dir/source-graph-bodies.csv" ]; then
      rm "$dir/source-graph-bodies.csv"
    fi
    if [ -f "$dir/nodes_with_ast.csv" ]; then
      rm "$dir/nodes_with_ast.csv"
    fi
    if [ -f "$dir/edges_with_ast.csv" ]; then
      rm "$dir/edges_with_ast.csv"
    fi
    if [ -f "$dir/call_seq.csv" ]; then
      rm "$dir/call_seq.csv"
    fi
    if [ -f "$dir/source-graph-function-variable-pairs.csv" ]; then
      rm "$dir/source-graph-function-variable-pairs.csv"
    fi


    if [ -f "$dir/$package_name.srctrldb" ]; then
      cd "$dir"
      sqlite3 "$dir/$package_name.srctrldb" < "$SQL_Q"
      cd "$RUN_DIR"
      sourcetrail-verify-files.py "$dir"
      sourcetrail-node-name-merge.py "$dir/nodes.csv"
      sourcetrail-decode-edge-types.py "$dir/edges.csv"
      sourcetrail-parse-bodies.py "$dir"
      sourcetrail-call-seq-extractor.py "$dir"


      sourcetrail-add-reverse-edges.py "$dir/edges.bz2"
      sourcetrail-ast-edges.py "$dir"
      sourcetrail-extract-variable-names.py py "$dir"

      if [ -f "$dir"/edges_with_ast_temp.csv ]; then
        rm "$dir"/edges_with_ast_temp.csv
      fi

    else
      echo "Package not indexed"
    fi
  fi
done

conda deactivate