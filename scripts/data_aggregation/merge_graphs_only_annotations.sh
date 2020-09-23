conda activate SourceCodeTools

ENVS_DIR=$1
RUN_DIR=$(dirname "$0")
OUT_DIR=$2

SPARK_GRAPH_ANALYSIS="$RUN_DIR/../../spark-SourceGraphAnalysis/out/artifacts/GraphAnalysis_jar"
if [ ! -f $SPARK_GRAPH_ANALYSIS/GraphAnalysis.jar ]; then
  echo "Graph Analysis executable not found:" "$SPARK_GRAPH_ANALYSIS/GraphAnalysis.jar"
fi

if [ ! -d $OUT_DIR/with_ast ]; then
  mkdir $OUT_DIR/with_ast
fi

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      sourcetrail-merge-graphs.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv
    fi
  fi
done


for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      sourcetrail-node-local2global.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/local2global_with_ast.csv

      sourcetrail-map-id-columns-only-annotations.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/edges_with_ast.csv $OUT_DIR/with_ast/common_edges_ast_type_as_str.csv target_node_id source_node_id
      sourcetrail-map-id-columns.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/source-graph-bodies.csv $OUT_DIR/with_ast/common.csv id
      sourcetrail-map-id-columns.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/source-graph-function-variable-pairs.csv $OUT_DIR/with_ast/common-function-variable-pairs.csv src
      sourcetrail-map-id-columns.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/call_seq.csv $OUT_DIR/with_ast/common-call-seq.csv src dst
    fi
  fi
done

if [ -f $OUT_DIR/with_ast/common_edges_ast_type_as_str.csv ]; then

  sourcetrail-extract-node-names.py $OUT_DIR/with_ast/common_nodes.csv $OUT_DIR/with_ast/node_names.csv
  sourcetrail-edge-types-to-int.py $OUT_DIR/with_ast/common_edges_ast_type_as_str.csv $OUT_DIR/with_ast/common_edges_ast_type_as_int.csv $OUT_DIR/with_ast/ast_types_int_to_str.csv
  sourcetrail-extract-type-information.py $OUT_DIR/with_ast/common_nodes.csv $OUT_DIR/with_ast/common_edges_ast_type_as_int.csv $OUT_DIR/with_ast/ast_types_int_to_str.csv $OUT_DIR/with_ast/common_edges_annotations.csv $OUT_DIR/with_ast/common_edges_no_annotations.csv
  cp $OUT_DIR/with_ast/common_edges_no_annotations.csv $OUT_DIR/with_ast/common_edges.csv





  if [ ! -d $OUT_DIR/with_ast/02_largest_component ]; then
    mkdir $OUT_DIR/with_ast/02_largest_component
  fi

  CURR_DIR="$(pwd)"
  cd $OUT_DIR/with_ast/02_largest_component
  echo "Analyzing with spark: $OUT_DIR/with_ast/"
  spark-submit $SPARK_GRAPH_ANALYSIS/GraphAnalysis.jar python ../common_nodes.csv ../common_edges.csv &> spark_analysis.log
  bash $RUN_DIR/../../spark-SourceGraphAnalysis/merging-spark-results.sh
  echo "done"
  cd "$CURR_DIR"

else
  echo "No edges found"
fi

conda deactivate
