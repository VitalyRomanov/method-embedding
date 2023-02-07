mkdir all_types pop_types
python ~/dev/method-embeddings/SourceCodeTools/code/experiments/run_experiment.py --base_path ../../v2_subsample_no_spacy_v3/with_ast --experiment typeann --type_ann ../../v2_subsample_no_spacy_v3/with_ast/type_annotations.json.bz2 --embeddings embeddings.pkl --trials 5 --out_path all_types > perform_experiment_all_types.log && python ~/dev/method-embeddings/SourceCodeTools/code/experiments/run_experiment.py --base_path ../../v2_subsample_no_spacy_v3/with_ast --experiment typeann --type_ann ../../v2_subsample_no_spacy_v3/with_ast/type_annotations.json.bz2 --embeddings embeddings.pkl --trials 5 --out_path pop_types --only_popular_types --popular_types str,int,bool,Callable,Dict,bytes,float,Description,List,Namespace,HTTPServerRequest,Future,Matcher > perform_experiment_pop_types.log
python ../parse_type_ann_experiment_log.py .