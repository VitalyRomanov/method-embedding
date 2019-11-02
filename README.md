This project contains the code for preprocessing the [code-docstring-corpus](https://github.com/EdinburghNLP/code-docstring-corpus) dataset.

###Usage
## `method-extraction.py` 
Exports function call grpah in the format of adjacency list or json
```
method-extraction.py [plain/json] path/to/code-docstring-corpus
```

## `tfidf.py`

Exports code docstrings in the format of tdidf vectors
```
tfidf.py path/to/code-docstring-corpus
```

## `lda.py`
Train LDA model for the functions specified by input file. Descriptions for functions with the same name concatenated together.

## `spearman.py`

Calculate spearman coefficient for two types of embeddings.

## `lda_filter.py`

Filter LDA embeddings using external files. The file provides the order of indexing used for embeddings created with diffent method.