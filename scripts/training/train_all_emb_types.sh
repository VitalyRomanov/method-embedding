conda activate dglke
dglke_train --model_name TransR --dataset code --data_path . --data_files edges_train_dglke.tsv held_dglkg.tsv held_dglkg.tsv --format raw_udd_htr --save_path transr_u --hidden_dim 100 --num_thread 4 --gpu 0
dglke_train --model_name RESCAL --dataset code --data_path . --data_files edges_train_dglke.tsv held_dglkg.tsv held_dglkg.tsv --format raw_udd_htr --save_path RESCAL_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 160000
dglke_train --model_name DistMult --dataset code --data_path . --data_files edges_train_dglke.tsv held_dglkg.tsv held_dglkg.tsv --format raw_udd_htr --save_path DistMult_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 400000
dglke_train --model_name ComplEx --dataset code --data_path . --data_files edges_train_dglke.tsv held_dglkg.tsv held_dglkg.tsv --format raw_udd_htr --save_path ComplEx_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 400000
dglke_train --model_name RotatE --dataset code --data_path . --data_files edges_train_dglke.tsv held_dglkg.tsv held_dglkg.tsv --format raw_udd_htr --save_path RotatE_u --hidden_dim 100 --num_thread 4 --gpu 0 --max_step 200000 -de
conda deactivate
