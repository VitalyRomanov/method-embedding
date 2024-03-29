import argparse
import os
import time
import logging

from nltk import RegexpTokenizer


_tokenizer = RegexpTokenizer("[\w]+|[^\w\s]|[0-9]+")


def default_tokenizer(text):
    return _tokenizer.tokenize(text)
# default_tokenizer = lambda text: _tokenizer.tokenize(text)


class Corpus(object):
    def __init__(self, file_name, tokenizer=None):
        self.dirname = file_name
        self.tok = default_tokenizer if tokenizer is None else tokenizer

    def __iter__(self):
        line_buf = ""
        ind = 0
        max_lines = 10
        for line in open(self.dirname):
            if ind < max_lines:
                line_buf += "\n" + line.lower()
                ind += 1
            else:
                tokens = self.tok(line_buf)
                yield tokens
                ind = 0
                line_buf = ""
        if line_buf:
            yield self.tok(line_buf)


# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_embedding_model(model, params, corpus_path, output_path, tokenizer=None):

    sentences = Corpus(corpus_path, tokenizer=tokenizer)

    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    print("FastText training started, ", time.strftime("%Y-%m-%d %H:%M"))
    model = model(sentences, **params)

    print("FastText training finished, ", time.strftime("%Y-%m-%d %H:%M"))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    model.wv.save_word2vec_format(output_path + "/" + 'emb.txt')
    print("Embeddings saved, ", time.strftime("%Y-%m-%d %H:%M"))
    model.save(output_path + "/" + 'model')
    print("Model saved, ", time.strftime("%Y-%m-%d %H:%M"))


def train_fasttext(corpus_path, output_path, tokenizer=None):
    from gensim.models import FastText
    
    params = {
        'size': 100,
        'window': 15,
        'min_count': 1,
        'workers': 4,
        'sg': 1,
        'negative': 15,
        'ns_exponent': 0.75,
        'sample': 1e-4,
        'iter': 20,
        'alpha': 0.1,
        'min_alpha': 5e-3,
        'sorted_vocab': 1,
        'max_vocab_size': 2000000,
        'word_ngrams': 1,
        'bucket': 200000,
        'min_n': 3,
        'max_n': 5
    }

    train_embedding_model(FastText, params, corpus_path, output_path, tokenizer)


def train_wor2vec(corpus_path, output_path, tokenizer=None, emb_size=100):
    from gensim.models import Word2Vec

    params = {
        'size': emb_size,
        'window': 15,
        'min_count': 1,
        'workers': 4,
        'sg': 1,
        'negative': 15,
        'ns_exponent': 0.75,
        'sample': 1e-4,
        'iter': 20,
        'alpha': 0.1,
        'min_alpha': 5e-3,
        'sorted_vocab': 1,
        'max_vocab_size': 2000000,
    }

    train_embedding_model(Word2Vec, params, corpus_path, output_path, tokenizer)


def export_w2v_for_tensorboard(embs_path, tb_meta_path, tb_embs_path, sep="\t"):
    with open(embs_path) as embeddings:
        embeddings.readline()
        with open(tb_meta_path, "w") as tb_meta:
            with open(tb_embs_path, "w") as tb_embs:
                for line in embeddings:
                    e_ = line.split(" ")
                    tb_meta.write(f"{e_[0]}\n")
                    tb_embs.write(f"{sep.join(e_[1:])}")


def load_w2v_map(w2v_path):
    """
    Load embeddings for words in w2v txt format
    :param w2v_path:
    :return: Embedder
    """
    embs = []
    w_map = dict()

    with open(w2v_path) as w2v:
        n_vectors, n_dims = map(int, w2v.readline().strip().split())
        for ind in range(n_vectors):
            line = w2v.readline()
            e = line.rstrip("\n").split(" ")

            word = e[0]
            w_map[word] = len(w_map)

            assert len(e[1:]) == n_dims
            embs.append(list(map(float, e[1:])))

    from SourceCodeTools.models.Embedder import Embedder
    import numpy as np

    return Embedder(w_map, np.array(embs))


def char_ngram_window(x, gram_size):
    x = "<" + x + ">"
    length = len(x)
    return (x[i:i + gram_size] for i in range(0, length) if i+gram_size <= length)


def main():
    parser = argparse.ArgumentParser(description='Train word vectors')
    parser.add_argument('input_file', type=str, default=150, help='Path to text file')
    parser.add_argument('output_dir', type=str, default=5, help='Output saving directory')
    args = parser.parse_args()

    corpus_path = args.input_file
    output_path = args.output_dir

    train_fasttext(corpus_path, output_path)


if __name__ == "__main__":
    main()
