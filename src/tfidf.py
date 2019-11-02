import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from pprint import pprint
import logging
import sys
import os

# corpus_path = "/Volumes/External/dev/code-translation/code-docstring-corpus/parallel-corpus/data_ps.descriptions.train"
corpus_path = os.path.join(sys.argv[1], "parallel-corpus/data_ps.descriptions.train")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import numpy as np

corpus = open(corpus_path, encoding='utf8', errors='ignore').read().strip().split("\n")

# dataset = api.load("text8")
dct = Dictionary(line.split() for line in corpus)
dct.filter_extremes(no_below=5, no_above=0.6)#, keep_n=956464)
dct.compactify()  # remove gaps in id sequence after words that were removed
common_corpus = [dct.doc2bow(line.split()) for line in corpus]

model = TfidfModel(common_corpus)  # fit model


# from gensim.similarities import MatrixSimilarity
# index = MatrixSimilarity(model[common_corpus], num_features=len(dct))

np_docs = np.zeros((1000, len(dct)))

for ind, doc in enumerate(common_corpus):
    # print(ind)
    if ind >= 1000: break
    sparse_vector = model[common_corpus[ind]]  # apply model to the first corpus document    
    for loc, val in sparse_vector:
        np_docs[ind, loc] = val

np.savetxt("tfidf_1000.txt", np_docs, delimiter="\t")


# vector = model[common_corpus[0]]  # apply model to the first corpus document

# pprint(vector.todense())