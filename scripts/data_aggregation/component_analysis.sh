conda activate SourceCodeTools || { echo 'Could not activate conda environment: SourceCodeTools' ; exit 1; }

RUN_DIR=$(realpath "$(dirname "$0")")

SPARK_GRAPH_ANALYSIS="$RUN_DIR/../../spark-SourceGraphAnalysis/out/artifacts/GraphAnalysis_jar/GraphAnalysis.jar"
if [ ! -f "$SPARK_GRAPH_ANALYSIS" ]; then
  echo "Graph Analysis executable not found:" "$SPARK_GRAPH_ANALYSIS"
  exit 1
fi

process_connected_components () {
  BASE_LOCATION=$1
  OUTPUT_PATH=$2

  if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir "$OUTPUT_PATH"
  fi

  pandas_format_converter.py "$BASE_LOCATION/common_nodes.bz2" parquet
  pandas_format_converter.py "$BASE_LOCATION/common_edges.bz2" parquet

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

  pandas_format_converter.py "$OUTPUT_PATH/nodes.parquet" bz2
  pandas_format_converter.py "$OUTPUT_PATH/edges.parquet" bz2

  rm "$BASE_LOCATION/common_nodes.parquet"
  rm "$BASE_LOCATION/common_edges.parquet"
  rm "$OUTPUT_PATH/nodes.parquet"
  rm "$OUTPUT_PATH/edges.parquet"
  rm -r temp

  bzip2 "$OUTPUT_PATH"/node_in_degrees.csv
  bzip2 "$OUTPUT_PATH"/node_out_degrees.csv
}

process_connected_components "$1" "$2"

conda deactivate