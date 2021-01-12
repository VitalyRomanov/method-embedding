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

if [ -d "$OUT_DIR/no_ast" ]; then
  rm -r "$OUT_DIR/no_ast"
fi

if [ ! -d "$OUT_DIR/with_ast" ]; then
  mkdir "$OUT_DIR/with_ast"
fi

if [ ! -d "$OUT_DIR/no_ast" ]; then
  mkdir "$OUT_DIR/no_ast"
fi

echo "Merging nodes"
for dir in "$ENVS_DIR"/*; do
  if [ -d "$dir" ]; then
    sourcetrail_merge_graphs.py "$OUT_DIR/no_ast/common_nodes.bz2" "$dir/nodes.bz2"
    sourcetrail_merge_graphs.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2"
  fi
done

for dir in "$ENVS_DIR"/*; do
  if [ -d "$dir" ]; then
    package_name="$(basename "$dir")"

    echo "Process $package_name"
#    if [ -f "$dir/nodes.bz2" ]; then

      sourcetrail_node_local2global.py "$OUT_DIR/no_ast/common_nodes.bz2" "$dir/nodes.bz2" "$dir/local2global.bz2"
      sourcetrail_node_local2global.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/local2global_with_ast.bz2"

      sourcetrail_map_id_columns.py "$OUT_DIR/no_ast/common_nodes.bz2" "$dir/nodes.bz2" "$dir/edges.bz2" "$OUT_DIR/no_ast/common_edges.bz2" target_node_id source_node_id
      sourcetrail_map_id_columns.py "$OUT_DIR/no_ast/common_nodes.bz2" "$dir/nodes.bz2" "$dir/source_graph_bodies.bz2" "$OUT_DIR/no_ast/common_bodies.bz2" id
      sourcetrail_map_id_columns.py "$OUT_DIR/no_ast/common_nodes.bz2" "$dir/nodes.bz2" "$dir/function_variable_pairs.bz2" "$OUT_DIR/no_ast/common_function_variable_pairs.bz2" src
      sourcetrail_map_id_columns.py "$OUT_DIR/no_ast/common_nodes.bz2" "$dir/nodes.bz2" "$dir/call_seq.bz2" "$OUT_DIR/no_ast/common_call_seq.bz2" src dst

      sourcetrail_map_id_columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/edges_with_ast.bz2" "$OUT_DIR/with_ast/common_edges.bz2" target_node_id source_node_id mentioned_in
      sourcetrail_map_id_columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/source_graph_bodies.bz2" "$OUT_DIR/with_ast/common_bodies.bz2" id
      sourcetrail_map_id_columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/function_variable_pairs.bz2" "$OUT_DIR/with_ast/common_function_variable_pairs.bz2" src
      sourcetrail_map_id_columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/call_seq.bz2" "$OUT_DIR/with_ast/common_call_seq.bz2" src dst

#    fi
  fi
done

sourcetrail_extract_node_names.py "$OUT_DIR/no_ast/common_nodes.bz2" "$OUT_DIR/no_ast/node_names.bz2"
sourcetrail_extract_node_names.py "$OUT_DIR/with_ast/common_nodes.bz2" "$OUT_DIR/with_ast/node_names.bz2"

#sourcetrail_edge_types_to_int.py "$OUT_DIR/with_ast/common_edges_ast_type_as_str.csv" "$OUT_DIR/with_ast/common_edges_ast_type_as_int.csv" "$OUT_DIR/with_ast/ast_types_int_to_str.csv"
#cp "$OUT_DIR/with_ast/common_edges_ast_type_as_int.csv" "$OUT_DIR/with_ast/common_edges.csv"


echo "Computing connected components"

$RUN_DIR/component_analysis.sh "$OUT_DIR/no_ast" "$OUT_DIR/no_ast/largest_component"
$RUN_DIR/component_analysis.sh "$OUT_DIR/with_ast" "$OUT_DIR/with_ast/largest_component"
#process_connected_components "$OUT_DIR/no_ast" "$OUT_DIR/no_ast/largest_component"
#process_connected_components "$OUT_DIR/with_ast" "$OUT_DIR/with_ast/largest_component"

conda deactivate
