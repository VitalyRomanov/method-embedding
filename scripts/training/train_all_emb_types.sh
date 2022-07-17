DATASET=$1
DGLKE_OUT=$2
GRAPH_FILE=$3

if [ -z $GRAPH_FILE ]; then
  GRAPH_FILE="edges_train_dglke.tsv"
fi

#conda activate SourceCodeTools
#prepare_dglke_format.py $DATASET $DGLKE_OUT
#conda deactivate

conda activate dglke

MAX_STEPS=1000000
HDIM=100
LR=0.001

echo "Training TransR"
dglke_train --model_name TransR --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/transr_u --hidden_dim $HDIM --num_thread 4 --max_step $MAX_STEPS -adv --lr $LR
echo "Training RESCAL"
dglke_train --model_name RESCAL --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/RESCAL_u --hidden_dim $HDIM --num_thread 4 --max_step $MAX_STEPS -adv --lr $LR
echo "Training DistMult"
dglke_train --model_name DistMult --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/DistMult_u --hidden_dim $HDIM --num_thread 4 --max_step $MAX_STEPS -adv --lr $LR
#echo "Training ComplEx"
#dglke_train --model_name ComplEx --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/ComplEx_u --hidden_dim $HDIM --num_thread 4 --max_step $MAX_STEPS -adv --lr $LR
#echo "Training RotatE"
#dglke_train --model_name RotatE --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/RotatE_u --hidden_dim $HDIM --num_thread 4 --max_step $MAX_STEPS -de -adv --lr $LR

echo "Evaluating TransR"
dglke_eval --model_name TransR --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --model_path $DGLKE_OUT/transr_u/TransR_code_0 --hidden_dim $HDIM --num_thread 4 --batch_size_eval 16
echo "Evaluating RESCAL"
dglke_eval --model_name RESCAL --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --model_path $DGLKE_OUT/RESCAL_u/RESCAL_code_0 --hidden_dim $HDIM --num_thread 4 --batch_size_eval 16
echo "Evaluating DistMult"
dglke_eval --model_name DistMult --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --model_path $DGLKE_OUT/DistMult_u/DistMult_code_0 --hidden_dim $HDIM --num_thread 4 --batch_size_eval 16
#echo "Evaluating ComplEx"
#dglke_eval --model_name ComplEx --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --model_path $DGLKE_OUT/ComplEx_u/ComplEx_code_0 --hidden_dim $HDIM --num_thread 4 --batch_size_eval 16
#echo "Evaluating RotatE"
#dglke_eval --model_name RotatE --dataset code --data_path $DGLKE_OUT --data_files $GRAPH_FILE edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --model_path $DGLKE_OUT/RotatE_u/RotatE_code_0 --hidden_dim $HDIM --num_thread 4 --batch_size_eval 16


conda deactivate
