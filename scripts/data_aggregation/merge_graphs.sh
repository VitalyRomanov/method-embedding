conda activate SourceCodeTools

ENVS_DIR=$1
RUN_DIR=$(dirname "$0")
OUT_DIR=$2
#SQL_Q=$RUN_DIR/extract.sql

SPARK_GRAPH_ANALYSIS="$RUN_DIR/../../spark-SourceGraphAnalysis/out/artifacts/GraphAnalysis_jar"
if [ ! -f $SPARK_GRAPH_ANALYSIS/GraphAnalysis.jar ]; then
  echo "Graph Analysis executable not found:" "$SPARK_GRAPH_ANALYSIS/GraphAnalysis.jar"
fi

#echo $RUN_DIR
#echo $SQL_Q
#echo $ENVS_DIR

#get directories
#https://stackoverflow.com/questions/2107945/how-to-loop-over-directories-in-linux

#for dir in $(ls $ENVS_DIR); do
#  if [ -d $ENVS_DIR/$dir ]; then
#    cd $ENVS_DIR/$dir
#    echo "Found package $dir"
#    if [ -f $dir.srctrldb ]; then
#      sqlite3 $dir.srctrldb < $SQL_Q
#      cd $RUN_DIR
#      python sourcetrail-verify-files.py $ENVS_DIR/$dir
#      python ../sourcetrail/sourcetrail-node-name-merge.py $ENVS_DIR/$dir/nodes.csv
#      python sourcetrail-parse-bodies.py $ENVS_DIR/$dir
#      python sourcetrail-ast-edges.py $ENVS_DIR/$dir
#      python ../code_processing/sourcetrail-call-seq-extractor.py $ENVS_DIR/$dir
#      python ../code_processing/sourcetrail-extract-variable-names.py py $ENVS_DIR/$dir
#    else
#      echo "Package not indexed"
#    fi
#  fi
#done

#if [ -n "$(find "$OUT_DIR" -name 'common*')" ]; then
#  rm $OUT_DIR/common*
#fi

if [ ! -d $OUT_DIR/with_ast ]; then
  mkdir $OUT_DIR/with_ast
fi

if [ ! -d $OUT_DIR/no_ast ]; then
  mkdir $OUT_DIR/no_ast
fi

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then
      sourcetrail-merge-graphs.py $OUT_DIR/no_ast/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv
      sourcetrail-merge-graphs.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv
    fi
  fi
done

for dir in $(ls $ENVS_DIR); do
  if [ -d $ENVS_DIR/$dir ]; then
    echo "Process $dir"
    if [ -f $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv ]; then

      sourcetrail-node-local2global.py $OUT_DIR/no_ast/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/local2global.csv
      sourcetrail-node-local2global.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/local2global_with_ast.csv

      sourcetrail-map-id-columns.py $OUT_DIR/no_ast/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/edges.csv $OUT_DIR/no_ast/common_edges.csv target_node_id source_node_id
      sourcetrail-map-id-columns.py $OUT_DIR/no_ast/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/source-graph-bodies.csv $OUT_DIR/no_ast/common-bodies.csv id
      sourcetrail-map-id-columns.py $OUT_DIR/no_ast/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/source-graph-function-variable-pairs.csv $OUT_DIR/no_ast/common-function-variable-pairs.csv src
      sourcetrail-map-id-columns.py $OUT_DIR/no_ast/common_nodes.csv $ENVS_DIR/$dir/normalized_sourcetrail_nodes.csv $ENVS_DIR/$dir/call_seq.csv $OUT_DIR/no_ast/common-call-seq.csv src dst

      sourcetrail-map-id-columns.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/edges_with_ast.csv $OUT_DIR/with_ast/common_edges_ast_type_as_str.csv target_node_id source_node_id
      sourcetrail-map-id-columns.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/source-graph-bodies.csv $OUT_DIR/with_ast/common-bodies.csv id
      sourcetrail-map-id-columns.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/source-graph-function-variable-pairs.csv $OUT_DIR/with_ast/common-function-variable-pairs.csv src
      sourcetrail-map-id-columns.py $OUT_DIR/with_ast/common_nodes.csv $ENVS_DIR/$dir/nodes_with_ast.csv $ENVS_DIR/$dir/call_seq.csv $OUT_DIR/with_ast/common-call-seq.csv src dst

    fi
  fi
done

sourcetrail-extract-node-names.py $OUT_DIR/no_ast/common_nodes.csv $OUT_DIR/no_ast/node_names.csv
sourcetrail-extract-node-names.py $OUT_DIR/with_ast/common_nodes.csv $OUT_DIR/with_ast/node_names.csv

sourcetrail-edge-types-to-int.py $OUT_DIR/with_ast/common_edges_ast_type_as_str.csv $OUT_DIR/with_ast/common_edges_ast_type_as_int.csv $OUT_DIR/with_ast/ast_types_int_to_str.csv
cp $OUT_DIR/with_ast/common_edges_ast_type_as_int.csv $OUT_DIR/with_ast/common_edges.csv




if [ ! -d $OUT_DIR/no_ast/02_largest_component ]; then
  mkdir $OUT_DIR/no_ast/02_largest_component
fi

CURR_DIR="$(pwd)"
cd $OUT_DIR/no_ast/02_largest_component
echo "Analyzing with spark: $OUT_DIR/no_ast/"
spark-submit $SPARK_GRAPH_ANALYSIS/GraphAnalysis.jar python ../common_nodes.csv ../common_edges.csv > spark_analysis.log
bash $RUN_DIR/../../spark-SourceGraphAnalysis/merging-spark-results.sh
echo "done"
cd "$CURR_DIR"

if [ ! -d $OUT_DIR/with_ast/02_largest_component ]; then
  mkdir $OUT_DIR/with_ast/02_largest_component
fi

CURR_DIR="$(pwd)"
cd $OUT_DIR/with_ast/02_largest_component
echo "Analyzing with spark: $OUT_DIR/with_ast/"
spark-submit $SPARK_GRAPH_ANALYSIS/GraphAnalysis.jar python ../common_nodes.csv ../common_edges.csv > spark_analysis.log
bash $RUN_DIR/../../spark-SourceGraphAnalysis/merging-spark-results.sh
echo "done"
cd "$CURR_DIR"

conda deactivate
