import numpy as np


def batch_split(input_tensor, batch_size):
    """Split an input tensor of size N x (dims) to a list of tensors of size batch_size x (dims).
    This function works with both numpy and torch tensors provided that dim 0 is the sampling dimension
    """

    return np.array_split(input_tensor, np.arange(batch_size, len(input_tensor), batch_size))


