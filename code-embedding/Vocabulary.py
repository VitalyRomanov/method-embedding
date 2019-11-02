from collections import Counter
import numpy as np
import pickle

class Vocabulary:
    """
    Vocabulary stores the mapping from words to IDs in word2id,
    the inverse mapping in id2word
    frequency counts for words in id_count

    The class has method for sampling from the vocabulary according to some distribution

    """
    word2id = None
    id2word = None
    id_count = None
    unigram_weights = None
    noise_weight = None

    # Used to implement subsampling described in skipgram negative sampling paper
    # discard_prob = None
    # alpha = 1e-3 # used for caclulating discard probability

    def __init__(self):
        self.word2id = {}
        self.word_count = Counter() # used temporary to simplify the process of counting
        self.id_count = Counter()
        self.id2word = {}
        self.negative_buffer = []
        self.top = set()

        self.negative_buffer_position = 0

    def _update(self):
        """
        Called before executing operations with vocabulary. Make sure all variables are up to date
        :return:
        """

        if len(self.word2id) != len(self.id2word):

            words, word_ids = zip(*self.word2id.items())
            self.id2word = dict(zip(word_ids, words))

            counts = np.array([self.id_count[id_] for id_ in sorted(self.id_count.keys())])
            self.unigram_weights = counts / sum(counts)
            noise_weight = self.unigram_weights**(3/4)
            self.noise_weight = noise_weight / sum(noise_weight)

    def get_id(self, word):
        """
        Return vocabulary ID of a word. Returns -1 if word is not in the vocabulary
        :param word:
        :return: None
        """
        return self.word2id.get(word, -1)

    def add_words(self, tokens):
        """
        Add new words to vocabulary. Updates only word_count
        :param tokens: list of string tokens
        :return: None
        """

        for t in tokens:
            if t in self.word_count:
                self.word_count[t] += 1
            else:
                self.word_count[t] = 1

        # TODO
        # set a flag that shows that vocabulary is updated
        # empty negative sampler buffer

    def prune(self, top_n):
        """
        Keep only top words in the vocabulary
        :param top_n:
        :return: None
        """
        id_count = Counter()
        word2id = {}
        for ind, (word, count) in enumerate(self.word_count.most_common(top_n)):
            word2id[word] = ind
            id_count[ind] = count

        self.word2id = word2id
        self.id_count = id_count

        self._update()

    def export_vocabulary(self, top_n, filename):
        """
        Save words and their counts in TSV file
        :param top_n: how many words to export
        :param filename: where to export
        :return: None
        """

        self._update()

        with open(filename, "w") as voc_exp:
            voc_exp.write("{}\t{}\n".format("Word", "Count"))
            for word_id, count in self.id_count.most_common(top_n):
                voc_exp.write("{}\t{}\n".format(self.id2word[word_id], count))

    def save(self, path):
        """
        Save vocabulary on disk
        :param path:
        :return: None
        """
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(path):
        """
        Load from disk
        :return: None
        """
        return pickle.load(open(path, "wb"))

    def total_tokens(self):
        """
        Return total numbe of tokens ovserved
        :return:
        """
        return sum(self.id_count.values())

    def __len__(self):
        return len(self.word2id)

    # def in_top(self, word_id):
    #     return word_id in self.id_count

    def sample_negative(self, k):
        """
        Sample words according to noise distribution.
        :param k: Number of samples
        :return: sample as list
        """
        self._update()

        # calling np.random is slow. bufferize 100000 random samples and get slices every time the method is called
        if self.negative_buffer_position + k > len(self.negative_buffer):
            self.negative_buffer = np.random.choice(np.array(list(self.id_count.keys())), 100000, p=self.noise_weight)
            self.negative_buffer_position = 0

        sample = self.negative_buffer[self.negative_buffer_position : self.negative_buffer_position + k].tolist()
        self.negative_buffer_position += k

        return sample


if __name__ == "__main__":
    import sys
    from nltk import word_tokenize

    corpus_path = sys.argv[1]
    output_path = sys.argv[2]

    voc = Vocabulary()

    counter = 0

    with open(corpus_path, "r") as reader:
        line = reader.readline()
        while line:
            tokens = word_tokenize(line.strip(), preserve_line=True)
            voc.add_words(tokens)
            line = reader.readline()
            counter += 1
            if counter % 10000 == 0:
                print(counter)

    print("Total {} tokens, unique {}".format(voc.total_tokens(), len(voc)))

    counter = 0
    new_voc = Vocabulary()

    with open(corpus_path, "r") as reader:
        with open(output_path, "w") as writer:
            line = reader.readline()
            while line:
                tokens = np.array(word_tokenize(line.strip(), preserve_line=True))

                if len(tokens) > 0:
                    discard = voc.discard(tokens)
                    
                    new_tokens = tokens[np.logical_not(discard)]
                    new_voc.add_words(new_tokens.tolist())
                    for token in new_tokens:
                        writer.write("{} ".format(token))
                    writer.write("\n")

                line = reader.readline()
                counter += 1
                if counter % 10000 == 0:
                    print(counter)
            
            print("Total {} tokens, unique {}".format(new_voc.total_tokens(), len(new_voc)))
            new_voc.save("voc.pkl")


