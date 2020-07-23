#%%

import sys
import gensim
from nltk import RegexpTokenizer
from gensim.corpora.dictionary import Dictionary
from gensim.similarities import Similarity
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tok = RegexpTokenizer("\w+|[^\w\s][0-9]+")

corpus_path = "/Volumes/External/dev/code-translation/code-docstring-corpus/parallel-corpus/data_ps.descriptions.train"#sys.argv[1]

def get_texts(path):
    with open(path, encoding='latin') as corpus_file:
        for line in corpus_file:
            yield tok.tokenize(line.strip())

#%% 

texts = [line.strip() for line in open(corpus_path, encoding='latin')]
#%%

common_dictionary = Dictionary(get_texts(corpus_path))
common_dictionary.filter_extremes()
common_dictionary.compactify()

#%%

common_corpus = [common_dictionary.doc2bow(text) for text in get_texts(corpus_path)]

#%% 
from gensim.test.utils import get_tmpfile

index_tmpfile = get_tmpfile("index")
index = Similarity(index_tmpfile, common_corpus, num_features=len(common_dictionary))

#%%
# index.save("docstringdescription")

#%%

def perm_sort(x):
    return sorted(range(len(x)), key=lambda k: x[k], reverse=True)

with open("similaritites_description_train", "w") as sink:
    for query_index in range(len(common_corpus)):
        scores = index[common_corpus[query_index]]
        order = perm_sort(scores)
        for pos in order:
            if scores[pos] < .6: break
            if query_index == pos: continue
            sink.write("%d\t%d\t%.4f\n" % (query_index, pos, scores[pos]))
        if query_index % 1000 == 0:
            print("\r%d/%d" % (query_index, len(common_corpus)))


#%%
