import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Embedding, concatenate
from tensorflow.keras import Model
from tensorflow.keras import regularizers


class LRClassifier(Model):
    def __init__(self, input_size=None):
        super(LRClassifier, self).__init__()
        self.logits = Dense(2,
                            kernel_regularizer=regularizers.l2(0.001))
                            # input_shape=(input_size,))

    def __call__(self, x, training=False, **kwargs):
        return self.logits(x)


class NNClassifier(Model):
    def __init__(self, input_size=None, h_size=None):
        super(NNClassifier, self).__init__()

        if h_size is None:
            h_size = [20]

        self.layers_ = []
        for ind, l in enumerate(h_size):
            if ind == 0:
                self.layers_.append(
                    Dense(l,
                            # kernel_regularizer=regularizers.l2(0.001),
                            input_shape=(input_size,), activation=tf.nn.relu)
                )
            else:
                self.layers_.append(Dense(l, activation=tf.nn.relu))
        self.layers_.append(Dense(2))

        # self.l1 = Dense(h_size[0],
        #                     # kernel_regularizer=regularizers.l2(0.001),
        #                     input_shape=(input_size,), activation=tf.nn.relu)
        # self.l2 = Dense(h_size[1], activation=tf.nn.relu)
        # self.l3 = Dense(h_size[2], activation=tf.nn.relu)
        # self.logits = Dense(2)


    def __call__(self, x, **kwargs):
        h = x
        for l in self.layers_:
            h = l(h)
        return h
        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # return self.logits(x)


class ElementPredictor(Model):
    def __init__(self, node_emb_size, n_emb, emb_size, h_size=50):
        super(ElementPredictor, self).__init__()
        self.emb_layer = Embedding(n_emb, emb_size)

        self.l1 = Dense(h_size, activation="relu",
                        input_shape=(node_emb_size + emb_size,))
        self.logits = Dense(2)

    def __call__(self, x, elements, **kwargs):

        element_embeddings = self.emb_layer(elements)
        x = concatenate([x, element_embeddings])
        x = self.l1(x)
        x = self.logits(x)
        return x


class NodeClassifier(Model):
    def __init__(self, node_emb_size, n_classes, h_size=None):
        super(NodeClassifier, self).__init__()

        if h_size is None:
            h_size = [30, 15]

        self.layers_ = []
        for ind, l in enumerate(h_size):
            if ind == 0:
                self.layers_.append(
                    Dense(l,
                          # kernel_regularizer=regularizers.l2(0.001),
                          input_shape=(node_emb_size,), activation=tf.nn.relu)
                )
            else:
                self.layers_.append(Dense(l, activation=tf.nn.relu))
        self.layers_.append(Dense(n_classes))

        # self.l1 = Dense(h_size[0], input_shape=(node_emb_size,))
        # self.l2 = Dense(h_size[2], input_shape=(h_size[0],))
        # self.logits = Dense(n_classes, input_shape=(h_size[1],))

    def __call__(self, x, **kwargs):
        h = x
        for l in self.layers_:
            h = l(h)
        return h
        # x = self.l1(x)
        # x = self.l2(x)
        # return self.logits(x)


