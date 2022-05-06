conda activate SourceCodeTools

ENVS_DIR=$(realpath "$1")
RUN_DIR=$(realpath "$(dirname "$0")")
OUT_DIR=$(realpath "$2")

SPARK_GRAPH_ANALYSIS="$RUN_DIR/../../spark-SourceGraphAnalysis/out/artifacts/GraphAnalysis_jar/GraphAnalysis.jar"
if [ ! -f "$SPARK_GRAPH_ANALYSIS" ]; then
  echo "Graph Analysis executable not found:" "$SPARK_GRAPH_ANALYSIS"
  exit
fi

if [ -d "$OUT_DIR/with_ast" ]; then
  rm -r "$OUT_DIR/with_ast"
fi

if [ ! -d "$OUT_DIR/with_ast" ]; then
  mkdir "$OUT_DIR/with_ast"
fi

for dir in "$ENVS_DIR"/*; do
  if [ -d "$dir" ]; then
    if [ -f "$dir/nodes.bz2" ]; then
      sourcetrail_merge_graphs.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2"
    fi
  fi
done


for dir in "$ENVS_DIR"/*; do
  if [ -d "$dir" ]; then
    package_name="$(basename "$dir")"
    
    echo "Process $package_name"
    if [ -f "$dir/nodes.bz2" ]; then
      sourcetrail_node_local2global.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/local2global_with_ast.bz2"

      sourcetrail_map_id_columns_only_annotations.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/edges_with_ast.bz2" "$OUT_DIR/with_ast/common_edges.bz2" target_node_id source_node_id mentioned_in
      sourcetrail_map_id_columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/source_graph_bodies.bz2" "$OUT_DIR/with_ast/common_bodies.bz2" id
      sourcetrail_map_id_columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/function_variable_pairs.bz2" "$OUT_DIR/with_ast/common_function_variable_pairs.bz2" src
      sourcetrail_map_id_columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/call_seq.bz2" "$OUT_DIR/with_ast/common_call_seq.bz2" src dst
    fi
  fi
done

sourcetrail_extract_node_names.py "$OUT_DIR/with_ast/common_nodes.bz2" "$OUT_DIR/with_ast/node_names.bz2"

if [ -f "$OUT_DIR/with_ast/common_edges.bz2" ]; then

#  sourcetrail_extract_type_information.py "$OUT_DIR/with_ast/common_edges.bz2" "$OUT_DIR/with_ast/common_edges_annotations.bz2" "$OUT_DIR/with_ast/common_edges_no_annotations.bz2"
#  mv "$OUT_DIR/with_ast/common_edges.bz2" "$OUT_DIR/with_ast/common_edges_with_types.bz2"
#  mv "$OUT_DIR/with_ast/common_edges_no_annotations.bz2" "$OUT_DIR/with_ast/common_edges.bz2"

  echo "Computing connected components"

  $RUN_DIR/component_analysis.sh "$OUT_DIR/with_ast" "$OUT_DIR/with_ast/largest_component"
#  process_connected_components "$OUT_DIR/with_ast" "$OUT_DIR/with_ast/largest_component"

else
  echo "No edges found"
fi

conda deactivate
