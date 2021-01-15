import logging
import os
from copy import copy
import pandas as pd
from gensim import corpora, models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def read_corpus(path, data_field):
    if path.endswith("bz2") or path.endswith("parquet") or path.endswith("csv"):
        from SourceCodeTools.data.sourcetrail.file_utils import unpersist
        data = unpersist(path)[data_field].tolist()
    elif path.endswith("jsonl"):
        import json
        data = []
        with open(path) as data_source:
            for ind, line in enumerate(data_source):
                if line.strip():
                    d = json.loads(line.strip())
                    if data_field in data:
                        data.append(d)
                    else:
                        logging.warning(f"No data field '{data_field}' on line {ind}")
    else:
        data = []
        with open(path) as data_source:
            for ind, line in enumerate(data_source):
                if line.strip():
                    data.append(line.strip())

    return data


class LdaTrainer:
    def __init__(self, corpus, tokenizer, output_path):
        self.corpus = corpus
        self.output_path = output_path

        if tokenizer == "default":
            logging.info("Loading regex tokenizer")
            from nltk import RegexpTokenizer
            tok = RegexpTokenizer("[\\w]+|[^\\w\\s]|[0-9]+")

            def tokenize(text):
                return tok.tokenize(text)

            self.tokenize = tokenize
        elif os.path.isfile(tokenizer):
            logging.info("Loading bpe tokenizer")
            from SourceCodeTools.embed.bpe import load_bpe_model, make_tokenizer
            self.tokenize = make_tokenizer(load_bpe_model(tokenizer))

        self.dictionary = self.load_dictionary()

        self.corpus_mm = self.load_corpus()
        self.train, self.test = train_test_split(self.corpus_mm, test_size=0.1)

    @property
    def vocab_path(self):
        return os.path.join(self.output_path, "vocab.dict")

    @property
    def corpus_path(self):
        return os.path.join(self.output_path, "corpus.mm")

    def model_path(self, params=None):
        if params is None:
            add = ""
        else:
            add = f"""_{"_".join(f"{key}_{val}" for key, val in params.items())}_"""
        return os.path.join(self.output_path, f"model{add}.lda")

    @property
    def results_path(self):
        return os.path.join(self.output_path, "results.csv")

    def load_dictionary(self):
        if os.path.isfile(self.vocab_path):
            dictionary = corpora.Dictionary.load(self.vocab_path)
        else:
            dictionary = corpora.Dictionary(self.tokenize(doc) for doc in self.corpus)
            # remove stop words and words that appear only once
            # stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            #  if stopword in dictionary.token2id]
            # once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq < 5]
            # dictionary.filter_tokens(once_ids)  # remove stop words and words that appear only once
            dictionary.filter_extremes(no_below=1, no_above=0.6)  # , keep_n=956464)
            dictionary.compactify()  # remove gaps in id sequence after words that were removed
            dictionary.save(self.vocab_path)
        return dictionary

    def load_corpus(self):
        if os.path.isfile(self.corpus_path):
            common_corpus = corpora.MmCorpus(self.corpus_path)
        else:
            common_corpus = [self.dictionary.doc2bow(self.tokenize(doc)) for doc in self.corpus]
            corpora.MmCorpus.serialize(self.corpus_path, common_corpus)

        return common_corpus

    def fit(self):

        all_results = []

        param_grid = {
            'num_topics': [50, 100, 150],
            'alpha': ['auto', 'symmetric'],
            'eta': ['auto'],
            'passes': [5, 10, 15],
            'decay': [0.55, 0.7, 0.95],
            'offset': [1, 4, 16]
        }

        for params in ParameterGrid(param_grid):
            # lda = models.ldamulticore.LdaMulticore(
            #     self.train, id2word=self.dictionary, **params
            # )
            lda = models.LdaModel(
                self.train, id2word=self.dictionary, **params
            )

            log_perplexity = lda.log_perplexity(self.test)

            results = copy(params)
            results['log_perplexity'] = log_perplexity

            logging.info(f"{results}")
            all_results.append(results)

            # lda.save(self.model_path(params))
            topics = lda.print_topics(num_topics=params['num_topics'], num_words=40)
            with open(self.model_path(params) + ".txt", "w") as topics_sink:
                for t in topics:
                    topics_sink.write(f"{t}\n")


        pd.DataFrame(all_results).to_csv(self.results_path, index=False)

    def save_as_vectors(self):
        # doc_toplics = lda.get_document_topics(
        #     self.corpus_mm, minimum_probability=0., minimum_phi_value=None, per_word_topics=False
        # )
        # from pprint import pprint
        #
        # # doc_topic_matrix = np.zeros((len(common_corpus), nt))
        #
        # # print(len(common_corpus), len(corpus))
        #
        # # for ind, doc in enumerate(common_corpus):
        # #     if ind not in empty:
        # #         doc_topics = lda.get_document_topics([doc], minimum_probability=0., minimum_phi_value=None, per_word_topics=False)[0]
        # #         topics, values = zip(*doc_topics)
        # #         doc_topic_matrix[ind] = np.array(values)
        #
        # doc_topic_matrix = np.zeros((len(common_corpus), nt))
        #
        # for ind, doc in enumerate(doc_toplics):
        #     topics, values = zip(*doc)
        #     doc_topic_matrix[ind] = np.array(values)
        #
        # print(len(f_names), len(desc))
        #
        # np.savetxt("doc_topic_matrix_%d_compacted.txt" % nt, doc_topic_matrix, delimiter="\t")
        pass


def main(args):
    corpus = read_corpus(args.corpus, args.data_field)
    trainer = LdaTrainer(corpus, args.tokenizer, args.output)
    trainer.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("output")
    parser.add_argument("--tokenizer", dest="tokenizer", default="regex", type=str)
    parser.add_argument("--data_field", dest="data_field", default=None, type=str)
    # parser.add_argument("--n_topics", dest="n_topics", default=100, type=int)

    args = parser.parse_args()

    main(args)
