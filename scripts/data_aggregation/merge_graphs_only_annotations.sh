conda activate SourceCodeTools

ENVS_DIR=$(realpath "$1")
RUN_DIR=$(realpath "$(dirname "$0")")
OUT_DIR=$(realpath "$2")

process_connected_components () {
  BASE_LOCATION=$1
  OUTPUT_PATH=$2

  if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir "$OUTPUT_PATH"
  fi

  pandas-format-converter.py "$BASE_LOCATION/common_nodes.bz2" parquet
  pandas-format-converter.py "$BASE_LOCATION/common_edges.bz2" parquet

  echo "Analyzing with spark: $BASE_LOCATION"
  spark-submit "$SPARK_GRAPH_ANALYSIS" "$BASE_LOCATION/common_nodes.parquet" "$BASE_LOCATION/common_edges.parquet" "$OUTPUT_PATH" &> "$OUTPUT_PATH"/spark_analysis.log

  mv "$OUTPUT_PATH"/component_0_node_type_count/*.csv  "$OUTPUT_PATH"/node_type_count.csv
  mv "$OUTPUT_PATH"/component_0_edge_type_count/*.csv  "$OUTPUT_PATH"/edge_type_count.csv
  mv "$OUTPUT_PATH"/component_0_node_in_degrees/*.csv  "$OUTPUT_PATH"/node_in_degrees.csv
  mv "$OUTPUT_PATH"/component_0_node_out_degrees/*.csv  "$OUTPUT_PATH"/node_out_degrees.csv
  mv "$OUTPUT_PATH"/component_0_nodes/*.parquet  "$OUTPUT_PATH"/nodes.parquet
  mv "$OUTPUT_PATH"/component_0_edges/*.parquet  "$OUTPUT_PATH"/edges.parquet

  rm -r "$OUTPUT_PATH"/component_0_node_type_count
  rm -r "$OUTPUT_PATH"/component_0_edge_type_count
  rm -r "$OUTPUT_PATH"/component_0_node_in_degrees
  rm -r "$OUTPUT_PATH"/component_0_node_out_degrees
  rm -r "$OUTPUT_PATH"/component_0_nodes
  rm -r "$OUTPUT_PATH"/component_0_edges

  pandas-format-converter.py "$OUTPUT_PATH/nodes.parquet" bz2
  pandas-format-converter.py "$OUTPUT_PATH/edges.parquet" bz2

  rm "$BASE_LOCATION/common_nodes.parquet"
  rm "$BASE_LOCATION/common_edges.parquet"
  rm "$OUTPUT_PATH/nodes.parquet"
  rm "$OUTPUT_PATH/edges.parquet"
  rm -r temp

  bzip2 "$OUTPUT_PATH"/node_in_degrees.csv
  bzip2 "$OUTPUT_PATH"/node_out_degrees.csv
}

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
      sourcetrail-merge-graphs.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2"
    fi
  fi
done


for dir in "$ENVS_DIR"/*; do
  if [ -d "$dir" ]; then
    package_name="$(basename "$dir")"
    
    echo "Process $package_name"
    if [ -f "$dir/nodes.bz2" ]; then
      sourcetrail-node-local2global.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/local2global_with_ast.bz2"

      sourcetrail-map-id-columns-only-annotations.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/edges_with_ast.bz2" "$OUT_DIR/with_ast/common_edges.bz2" target_node_id source_node_id mentioned_in
      sourcetrail-map-id-columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/source-graph-bodies.bz2" "$OUT_DIR/with_ast/common_bodies.bz2" id
      sourcetrail-map-id-columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/function-variable-pairs.bz2" "$OUT_DIR/with_ast/common_function_variable_pairs.bz2" src
      sourcetrail-map-id-columns.py "$OUT_DIR/with_ast/common_nodes.bz2" "$dir/nodes_with_ast.bz2" "$dir/call-seq.bz2" "$OUT_DIR/with_ast/common-call-seq.bz2" src dst
    fi
  fi
done

sourcetrail-extract-node-names.py "$OUT_DIR/with_ast/common_nodes.bz2" "$OUT_DIR/with_ast/node_names.bz2"

if [ -f "$OUT_DIR/with_ast/common_edges.bz2" ]; then

  sourcetrail-extract-type-information.py "$OUT_DIR/with_ast/common_edges.bz2" "$OUT_DIR/with_ast/common_edges_annotations.bz2" "$OUT_DIR/with_ast/common_edges_no_annotations.bz2"
  mv "$OUT_DIR/with_ast/common_edges.bz2" "$OUT_DIR/with_ast/common_edges_with_types.bz2"
  mv "$OUT_DIR/with_ast/common_edges_no_annotations.bz2" "$OUT_DIR/with_ast/common_edges.bz2"

  echo "Computing connected components"

  process_connected_components "$OUT_DIR/with_ast" "$OUT_DIR/with_ast/largest_component"

else
  echo "No edges found"
fi

conda deactivate
