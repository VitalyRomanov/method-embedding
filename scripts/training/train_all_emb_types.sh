DATASET=$1
DGLKE_OUT=$2

#conda activate SourceCodeTools
#prepare_dglke_format.py $DATASET $DGLKE_OUT
#conda deactivate

conda activate dglke

dglke_train --model_name TransR --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/transr_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 80000 -adv --test
dglke_train --model_name TransR --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/transr_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 160000 -adv --test

dglke_train --model_name RESCAL --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/RESCAL_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 160000 -adv --test
dglke_train --model_name RESCAL --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/RESCAL_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 240000 -adv --test

dglke_train --model_name DistMult --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/DistMult_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 400000 -adv --test
dglke_train --model_name DistMult --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/DistMult_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 800000 -adv --test

dglke_train --model_name ComplEx --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/ComplEx_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 400000 -adv --test
dglke_train --model_name ComplEx --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/ComplEx_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 800000 -adv --test

dglke_train --model_name RotatE --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/RotatE_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 200000 -de -adv --test
dglke_train --model_name RotatE --dataset code --data_path $DGLKE_OUT --data_files edges_train_dglke.tsv edges_eval_dglke_10000.tsv edges_eval_dglke_10000.tsv --format raw_udd_htr --save_path $DGLKE_OUT/RotatE_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 400000 -de -adv --test

conda deactivate
