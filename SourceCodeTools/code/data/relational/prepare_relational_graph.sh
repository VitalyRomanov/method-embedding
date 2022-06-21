DATASET_PATH=$(realpath "$1")
OUTPUT_PATH=$(realpath "$2")
CREATE_DATA=$3
RUN_DIR=$(realpath "$(dirname "$0")")

SCRIPTS_PATH=$RUN_DIR/../../../../SourceCodeTools/code/data/relational

if [ ! -z $CREATE_DATA ]; then
  python $SCRIPTS_PATH/prepare_dglke_format2.py $DATASET_PATH $DATASET_PATH/dglke2 --eval_frac 0.001 --node_type_edges
  python $SCRIPTS_PATH/k_hop_graph.py $DATASET_PATH/dglke2 2
fi

remove_if_exists () {
  if [ -d $1 ]; then
    rm -r $1;
  fi
}

remove_if_exists $DATASET_PATH/dglke2/transr_u
remove_if_exists $DATASET_PATH/dglke2/RESCAL_u
remove_if_exists $DATASET_PATH/dglke2/DistMult_u
remove_if_exists $DATASET_PATH/dglke2/ComplEx_u
remove_if_exists $DATASET_PATH/dglke2/RotatE_u

zsh -i $RUN_DIR/../../../../scripts/training/train_all_emb_types.sh $DATASET_PATH/dglke2 $DATASET_PATH/dglke2 > $DATASET_PATH/dglke2/training.log

process_if_exists () {
  VECTORS=$1;
  EMBEDDER=$2;
  TB_OUT=$3;
  if [ -d $DATASET_PATH/dglke2/transr_u ]; then
    python $RUN_DIR/../../../../SourceCodeTools/models/graph/utils/dglke_to_embedder2.py $DATASET_PATH/dglke2/entities.tsv $VECTORS $EMBEDDER
    python $RUN_DIR/../../../../SourceCodeTools/models/graph/utils/export4visualization2.py $DATASET_PATH $EMBEDDER $TB_OUT --into_groups
#    python $RUN_DIR/../../../../SourceCodeTools/models/graph/utils/dglke_to_embedder2.py $DATASET_PATH/dglke2/entities.tsv $DATASET_PATH/dglke2/transr_u/TransR_code_0/code_TransR_entity.npy $DATASET_PATH/dglke2/transr_u/TransR_code_0/embedder.pkl
#    python $RUN_DIR/../../../../SourceCodeTools/models/graph/utils/export4visualization2.py $DATASET_PATH $DATASET_PATH/dglke2/transr_u/TransR_code_0/embedder.pkl $DATASET_PATH/dglke2/transr_u/TransR_code_0/ --into_groups
  fi;
}

process_if_exists "$DATASET_PATH/dglke2/transr_u/TransR_code_0/code_TransR_entity.npy" "$DATASET_PATH/dglke2/transr_u/TransR_code_0/embedder.pkl" "$DATASET_PATH/dglke2/transr_u/TransR_code_0/"
process_if_exists "$DATASET_PATH/dglke2/RESCAL_u/RESCAL_code_0/code_RESCAL_entity.npy" "$DATASET_PATH/dglke2/RESCAL_u/RESCAL_code_0/embedder.pkl" "$DATASET_PATH/dglke2/RESCAL_u/RESCAL_code_0/"
process_if_exists "$DATASET_PATH/dglke2/DistMult_u/DistMult_code_0/code_DistMult_entity.npy" "$DATASET_PATH/dglke2/DistMult_u/DistMult_code_0/embedder.pkl" "$DATASET_PATH/dglke2/DistMult_u/DistMult_code_0/"
process_if_exists "$DATASET_PATH/dglke2/ComplEx_u/ComplEx_code_0/code_ComplEx_entity.npy" "$DATASET_PATH/dglke2/ComplEx_u/ComplEx_code_0/embedder.pkl" "$DATASET_PATH/dglke2/ComplEx_u/ComplEx_code_0/"
process_if_exists "$DATASET_PATH/dglke2/RotatE_u/RotatE_code_0/code_RotatE_entity.npy" "$DATASET_PATH/dglke2/RotatE_u/RotatE_code_0/embedder.pkl" "$DATASET_PATH/dglke2/RotatE_u/RotatE_code_0/"

python $RUN_DIR/../../../../scripts/training/dglke_log_parser.py $DATASET_PATH/dglke2/training.log