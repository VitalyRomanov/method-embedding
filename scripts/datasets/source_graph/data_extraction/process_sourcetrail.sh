conda activate SourceCodeTools

BPE_PATH="$2"
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

#    if [ -f "$dir/source_graph_bodies.csv" ]; then
#      rm "$dir/source_graph_bodies.csv"
#    fi
#    if [ -f "$dir/nodes_with_ast.csv" ]; then
#      rm "$dir/nodes_with_ast.csv"
#    fi
#    if [ -f "$dir/edges_with_ast.csv" ]; then
#      rm "$dir/edges_with_ast.csv"
#    fi
#    if [ -f "$dir/call_seq.csv" ]; then
#      rm "$dir/call_seq.csv"
#    fi
#    if [ -f "$dir/source_graph_function_variable_pairs.csv" ]; then
#      rm "$dir/source_graph_function_variable_pairs.csv"
#    fi


    if [ -f "$dir/$package_name.srctrldb" ]; then
      cd "$dir"
      sqlite3 "$dir/$package_name.srctrldb" < "$SQL_Q"
      cd "$RUN_DIR"
      sourcetrail_verify_files.py "$dir"
#      sourcetrail_node_name_merge.py "$dir/nodes.csv"
#      sourcetrail_decode_edge_types.py "$dir/edges.csv"
#      sourcetrail_filter_ambiguous_edges.py $dir
#      sourcetrail_parse_bodies.py "$dir"
#      sourcetrail_call_seq_extractor.py "$dir"


#      sourcetrail_add_reverse_edges.py "$dir/edges.bz2"
#      if [ -n "$BPE_PATH" ]; then
#        BPE_PATH=$(realpath "$BPE_PATH")
#        sourcetrail_ast_edges.py "$dir" -bpe $BPE_PATH --create_subword_instances
#      else
#        sourcetrail_ast_edges.py "$dir"
#      fi
#      sourcetrail_extract_variable_names.py python "$dir"

#      if [ -f "$dir"/edges_with_ast_temp.csv ]; then
#        rm "$dir"/edges_with_ast_temp.csv
#      fi

    else
      echo "Package not indexed"
    fi
  fi
done

conda deactivate