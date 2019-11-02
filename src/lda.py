from gensim import corpora, models
import sys
from gensim.test.utils import datapath
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
from sklearn.model_selection import train_test_split
import numpy as np

# corpus_path = sys.argv[1]
# corpus_path = "/Volumes/External/dev/code-translation/code-docstring-corpus/parallel-corpus/data_ps.descriptions.train"
corpus_path = os.path.join(sys.argv[1], "parallel-corpus/data_ps.descriptions.train")
func_names_to_embed_path = sys.argv[2] # file with the list of fuctions to use during training: parent_only.txt
func_names = open(func_names_to_embed_path).read().strip().split()

corpus = open(corpus_path, encoding='utf8', errors='ignore').read().strip().split("\n")

dict_name = "deerwester.dict"
corp_name = "corpus.mm"

from six import iteritems
if os.path.isfile(dict_name):
    dictionary = corpora.Dictionary.load(dict_name)
else:
    dictionary = corpora.Dictionary(line.split() for line in corpus)
    # remove stop words and words that appear only once
    # stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                #  if stopword in dictionary.token2id]
    # once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq < 5]
    # dictionary.filter_tokens(once_ids)  # remove stop words and words that appear only once
    dictionary.filter_extremes(no_below=1, no_above=0.6)#, keep_n=956464)
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    # print(dictionary)
    # dictionary.save(dict_name)


if os.path.isfile(corp_name):
    common_corpus = corpora.MmCorpus(corp_name)
else:
    # common_corpus = [dictionary.doc2bow(line.split()) for line in corpus]
    # corpora.MmCorpus.serialize(corp_name, common_corpus)
    f_names_desc = dict()
    for ind, text in enumerate(corpus):
        if func_names[ind] in f_names_desc:
            f_names_desc[func_names[ind]] += " " + text
        else:
            f_names_desc[func_names[ind]] = text

    f_names, desc = zip(*f_names_desc.items())

    common_corpus = [dictionary.doc2bow(line.split()) for line in desc]

    with open("compacted_for_desc.txt", "w") as cfd:
        for n in f_names:
            cfd.write("%s\n" % n) 



empty = set(ind for ind, doc in enumerate(common_corpus) if len(doc) == 0)
# for ind, doc in enumerate(common_corpus):
#     if len(doc) == 0: 
#         print("Doc %d is empty" % ind)

# print("Total docs %d" % len(common_corpus))
 
# common_corpus, test = train_test_split(common_corpus)

n_topics = [500]#, 100, 120]
alpha = ['auto']#,'symmetric']
eta = ['auto']

for nt in n_topics:
    for a in alpha:
        for e in eta:
            lda = models.ldamulticore.LdaMulticore(common_corpus, id2word=dictionary, num_topics=nt, passes=5)
            # lda = models.LdaModel(common_corpus, id2word=dictionary, num_topics=nt, passes=5, alpha=a, eta=e)
            doc_toplics = lda.get_document_topics(common_corpus, minimum_probability=0., minimum_phi_value=None, per_word_topics=False)
            from pprint import pprint

            # doc_topic_matrix = np.zeros((len(common_corpus), nt))

            # print(len(common_corpus), len(corpus))

            # for ind, doc in enumerate(common_corpus):
            #     if ind not in empty:
            #         doc_topics = lda.get_document_topics([doc], minimum_probability=0., minimum_phi_value=None, per_word_topics=False)[0]
            #         topics, values = zip(*doc_topics)
            #         doc_topic_matrix[ind] = np.array(values)

            doc_topic_matrix = np.zeros((len(common_corpus), nt))

            for ind, doc in enumerate(doc_toplics):
                topics, values = zip(*doc)
                doc_topic_matrix[ind] = np.array(values)

            print(len(f_names), len(desc))

            np.savetxt("doc_topic_matrix_%d_compacted.txt" % nt, doc_topic_matrix, delimiter="\t")
            # lda = models.LdaModel(common_corpus, id2word=dictionary, num_topics=nt, passes=10, alpha=a, eta=e)
            # lp = lda.log_perplexity(test)
            # with open("res.txt", "a") as res:
            #     res.write("%d %s %s %f\n" % (nt, a, e, lp))

# temp_file = datapath("model")
# lda.save("model")
