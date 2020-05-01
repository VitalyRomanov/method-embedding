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
    def __init__(self, input_size=None):
        super(NNClassifier, self).__init__()
        self.l1 = Dense(20,
                            # kernel_regularizer=regularizers.l2(0.001),
                            input_shape=(input_size,), activation=tf.nn.relu)
        self.l2 = Dense(30, activation=tf.nn.relu)
        self.l3 = Dense(10, activation=tf.nn.relu)
        self.logits = Dense(2)


    def __call__(self, x, **kwargs):
        x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        return self.logits(x)

class ElementPredictor(Model):
    def __init__(self, node_emb_size, n_emb, emb_size):
        super(ElementPredictor, self).__init__()
        self.emb_layer = Embedding(n_emb, emb_size)

        self.l1 = Dense(50, activation="relu",
                        input_shape=(node_emb_size + emb_size,))
        self.logits = Dense(2)

    def __call__(self, x, elements, **kwargs):

        element_embeddings = self.emb_layer(elements)
        x = concatenate([x, element_embeddings])
        x = self.l1(x)
        x = self.logits(x)
        return x


class NodeClassifier(Model):
    def __init__(self, node_emb_size, n_classes):
        super(NodeClassifier, self).__init__()

        self.l1 = Dense(30, input_shape=(node_emb_size,))
        self.l2 = Dense(15, input_shape=(30,))
        self.logits = Dense(n_classes, input_shape=(15,))

    def __call__(self, x, **kwargs):
        x = self.l1(x)
        x = self.l2(x)
        return self.logits(x)


