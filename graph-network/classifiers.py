import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras import Model
from tensorflow.keras import regularizers


class LRClassifier(Model):
    def __init__(self, input_size=None):
        super(LRClassifier, self).__init__()
        self.logits = Dense(2,
                            kernel_regularizer=regularizers.l2(0.001))
                            # input_shape=(input_size,))

    def __call__(self, x, training=False):
        return self.logits(x)


class NNClassifier(Model):
    def __init__(self, input_size=None):
        super(NNClassifier, self).__init__()
        self.l1 = Dense(50,
                            kernel_regularizer=regularizers.l2(0.001),
                            input_shape=(input_size,))
        self.l2 = Dense(30, input_shape=(50,))
        self.l3 = Dense(10, input_shape=(30,))
        self.logits = Dense(2, input_shape=(10,))


    def __call__(self, x, training=False):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return self.logits(x)