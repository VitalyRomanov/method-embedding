import tensorflow as tf


def to_numpy(tensor):
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    else:
        return tf.make_ndarray(tf.make_tensor_proto(tensor))