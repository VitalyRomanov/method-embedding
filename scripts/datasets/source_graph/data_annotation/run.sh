bash -i merge_annotation_texts.sh envs

python ft_bin_to_vec.py python-ftskip-dim100-ws4.bin > python-ft-vectors.txt
python -m spacy init-model en spacy-python -v python-ft-vectors.txt
python sourcecodetools-replace-tokenizer.py spacy-python spacy-python-with-tok
python sourcecodetools-spacy-ner.py spacy-python-with-tok functions_with_annotations.jsonl