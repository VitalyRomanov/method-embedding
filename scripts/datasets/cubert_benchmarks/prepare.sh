CUBERT_DATA_PATH=$(realpath "$1")
BPE_TOKENIZER=$(realpath "$2")
OUTPUT_PATH=$(realpath "$3")
RUN_DIR=$(realpath "$(dirname "$0")")

KEEP_FRAC=0.10

python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/prepare_for_ast_parser.py" $CUBERT_DATA_PATH
python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/convert_for_ast_graph_builder.py" $CUBERT_DATA_PATH/variable_misuse.json $CUBERT_DATA_PATH/variable_misuse.csv
python "$RUN_DIR/../../../SourceCodeTools/code/data/ast_graph/build_ast_graph.py" --bpe_tokenizer $BPE_TOKENIZER --track_offsets $CUBERT_DATA_PATH/variable_misuse.csv $OUTPUT_PATH --chunksize 100000 --keep_frac $KEEP_FRAC --seed 42 --do_extraction
python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/variable_misuse_node_level_labels.py" $OUTPUT_PATH/with_ast
python "$RUN_DIR/../../../SourceCodeTools/code/data/cubert_python_benchmarks/extract_partition.py" $OUTPUT_PATH/with_ast/common_filecontent.json
