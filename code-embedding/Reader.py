import numpy as np

class Reader:
    """
    Reader for implementing skipgram negative sampling
    Class exposes method next_minibatch

    takes a plain text file as input
    """
    def __init__(self, path, vocabulary, n_contexts, window_size, k, ):
        """

        :param path: Location of training file
        :param vocabulary: Vocabulary that can generate negative samples
        :param n_contexts: coxtexts to include in minibatch
        :param window_size: window size per context. full window span is window_size*2 + 1
        :param k: number of negative samples per context
        """
        self.file = None
        self.path = path
        self.voc = vocabulary
        self.tokens = []

        self.window_size = window_size
        self.n_contexts = n_contexts
        self.k = k

        self.position = window_size

        self.init()

    def init(self):
        # If file is opened
        if self.file is not None:
            self.file.close()
        self.file = open(self.path, "r")
        # read initial set of tokens
        # self.tokens = self.file.readline().strip().split()
        self.read_everything()

    def read_everything(self):
        lines = self.file.read().strip().split("\n")
        pairs = map(lambda line: line.split(), lines)
        self.pairs_ids = list(map(lambda pair: (self.voc.get_id(pair[0]), self.voc.get_id(pair[1])), pairs))
        print("Read everything")


    def get_tokens(self):
        new_tokens = self.file.readline().strip().split()
        if len(new_tokens) == 0:
            # at the end of the file
            self.tokens = None
        else:
            # discard old tokens and append new ones
            self.tokens = self.tokens[self.position - self.window_size: -1] + new_tokens
            self.position = self.window_size

    def batches(self, b_size = 128):
        """
        Generate next minibatch. Only words from vocabulary are included in minibatch
        :return: batches for (context_word, second_word, label)
        """


        position = 0

        # batch_size = 0

        # while be

        # while position < len(self.pairs_ids):
        #     slices = self.pairs_ids[position: min(position + b_size, len(self.pairs_ids))]

        batch = []

        for c_token_id, c_target_id in self.pairs_ids:

            if c_target_id == -1: continue
            if c_token_id == -1: continue
            batch.append([c_token_id, c_target_id, 1.])

            neg = self.voc.sample_negative(self.k)

            for n in neg:
                # if word is the same as central word, the pair is omitted
                if n != c_token_id:
                    batch.append([c_token_id, n, 0.])

            if len(batch) > b_size:

                # position += b_size
                bb = np.array(batch)
                yield bb[:,0], bb[:,1], bb[:,2]
                batch = []

        

        # batch = []
        # context_count = 0

        # while self.tokens is not None and context_count < self.n_contexts:
        #     # get more tokens if necessary
        #     while self.tokens is not None and self.position + self.window_size + 1 > len(self.tokens):
        #         self.get_tokens()

        #     # re-initialize if at the end of the file
        #     if self.tokens is None:
        #         self.init()
        #         return None

        #     c_token = self.tokens[self.position]
        #     c_token_id = self.voc.get_id(c_token)

        #     if c_token_id != -1:
        #         # generate positive samples
        #         for i in range(-self.window_size, self.window_size + 1, 1):
        #             if i == 0:
        #                 continue

        #             c_pair = self.tokens[self.position + i]
        #             c_pair_id = self.voc.get_id(c_pair)

        #             # if word not in vocabulary, the pair ommited
        #             if c_pair_id != -1:
        #                 batch.append([c_token_id, c_pair_id, 1.])

        #         # generate negative samples
        #         neg = self.voc.sample_negative(self.k)
        #         for n in neg:
        #             # if word is the same as central word, the pair is ommited
        #             if n != c_token_id:
        #                 batch.append([c_token_id, n, 0.])

        #         context_count += 1

        #     self.position += 1

        # bb = np.array(batch)
        # return bb[:,0], bb[:,1], bb[:,2]