conda activate dgl-l

edge_types=("none")
iters=(1 2 3 4 5)

for edge_type in $edge_types; do
  for i in $iters; do
    echo "$edge_type $i"
#    echo "filter $edge_type ADAM lr 0.001 iter $i" "main_filter_($edge_type)_($i).log"
#    python main.py --training_mode multitask --node_path /Users/LTV/Documents/subsample/common/02_largest_component/nodes_component_0.csv.bz2 --edge_path /Users/LTV/Documents/subsample/common/02_largest_component/edges_component_0.csv.bz2 --fname_file /Users/LTV/Documents/subsample/common/node_names.csv --varuse_file /Users/LTV/Documents/subsample/common/common-function-variable-pairs_with_ast.csv --call_seq_file /Users/LTV/Documents/subsample/common/common-call-seq_with_ast.csv --note "filter $edge_type ADAM lr 0.01 iter $i" --epochs 100
    python main.py --training_mode multitask --node_path /Users/LTV/Documents/subsample/graph_emb/unified_data/nodes.csv --edge_path /Users/LTV/Documents/subsample/graph_emb/unified_data/edges_train.csv --holdout /Users/LTV/Documents/subsample/graph_emb/unified_data/held.csv --fname_file /Users/LTV/Documents/subsample/common/node_names.csv --varuse_file /Users/LTV/Documents/subsample/common/common-function-variable-pairs_with_ast.csv --call_seq_file /Users/LTV/Documents/subsample/common/common-call-seq_with_ast.csv --note "filter $edge_type ADAM lr 0.01 iter $i" --epochs 100 --train_frac 1.0
  done
done

conda deactivate