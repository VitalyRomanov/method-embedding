CUBERT_DATA_PATH=$(realpath "$1")
BPE_TOKENIZER=$(realpath "$2")
OUTPUT_PATH=$(realpath "$3")
RUN_DIR=$(realpath "$(dirname "$0")")
GRAPH_FORMAT=$4
USE_INSTANCE=$5

KEEP_FRAC=0.10

if [ -z "$USE_INSTANCE" ]; then
  MENTION_INST_FLAG=""
  echo "Not using instances"
else
  MENTION_INST_FLAG="--use_mention_instances"
  echo "Using instances"
fi

#python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/prepare_for_ast_parser.py" $CUBERT_DATA_PATH &&
#python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/variable_misuse_detect_spans.py" $CUBERT_DATA_PATH/variable_misuse.json $CUBERT_DATA_PATH/variable_misuse_with_spans.json &&
#python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/convert_for_ast_graph_builder.py" $CUBERT_DATA_PATH/variable_misuse_with_spans.json $CUBERT_DATA_PATH/variable_misuse.csv --keep_frac $KEEP_FRAC &&
python "$RUN_DIR/../../../SourceCodeTools/code/data/ast_graph/build_ast_graph.py" --bpe_tokenizer $BPE_TOKENIZER --track_offsets $CUBERT_DATA_PATH/variable_misuse.csv $OUTPUT_PATH --chunksize 25000 --keep_frac 1.0 --seed 42 --do_extraction --graph_format_version $GRAPH_FORMAT $MENTION_INST_FLAG &&
python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/variable_misuse_node_level_labels.py" $OUTPUT_PATH/with_ast &&
python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/extract_partition.py" $OUTPUT_PATH/with_ast/common_filecontent.json &&
python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/make_edge_level_labels.py" $OUTPUT_PATH/with_ast/ $MENTION_INST_FLAG &&
python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/create_text_dataset.py" $OUTPUT_PATH/with_ast/ &&
for file in $OUTPUT_PATH/with_ast/*.json; do bzip2 $file; done
