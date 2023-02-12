ENVIRONMENTS_PATH=$(realpath "$1")
BPE_TOKENIZER=$(realpath "$2")
OUTPUT_PATH=$(realpath "$3")
RUN_DIR=$(realpath "$(dirname "$0")")

python $RUN_DIR/../../../SourceCodeTools/code/data/sourcetrail/DatasetCreator2.py $ENVIRONMENTS_PATH $OUTPUT_PATH --do_extraction --track_offsets --bpe_tokenizer $BPE_TOKENIZER
python $RUN_DIR/../../../SourceCodeTools/code/data/type_annotation_dataset/create_type_annotation_dataset.py envs $ENVIRONMENTS_PATH $OUTPUT_PATH/with_ast/annotations_dataset_no_default.json --remove_default --require_labels
python $RUN_DIR/../../../SourceCodeTools/code/data/type_annotation_dataset/map_args_to_mentions.py $OUTPUT_PATH/with_ast $OUTPUT_PATH/with_ast/annotations_dataset_no_default_args_mapped.json --dataset_file $OUTPUT_PATH/with_ast/annotations_dataset_no_default.json
python $RUN_DIR/../../../SourceCodeTools/code/data/type_annotation_dataset/split_dataset.py --data_path $OUTPUT_PATH/with_ast/annotations_dataset_no_default_args_mapped.json --random_seed 42 --name_suffix no_default_args_mapped
python $RUN_DIR/../../../SourceCodeTools/code/data/type_annotation_dataset/create_type_annotation_aware_graph_partition.py $OUTPUT_PATH/with_ast $OUTPUT_PATH/with_ast/type_prediction_dataset_no_default_args_mapped_test.json $OUTPUT_PATH/with_ast/partition_type_prediction.json

for file in $OUTPUT_PATH/with_ast/*.json; do
  bzip2 $file
done

bunzip2 $OUTPUT_PATH/with_ast/type_prediction_dataset_no_default_args_mapped_test.json.bz2
bunzip2 $OUTPUT_PATH/with_ast/type_prediction_dataset_no_default_args_mapped_train.json.bz2