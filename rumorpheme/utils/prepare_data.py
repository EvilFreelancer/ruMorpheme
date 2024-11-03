import numpy as np

from rumorpheme.const import PAD, BEGIN, END, UNKNOWN


def prepare_data(data, symbol_codes, max_word_length=64):
    """
    Function to prepare data for training and inference.
    """
    batch_size = len(data)
    inputs = np.full((batch_size, max_word_length), PAD, dtype=int)
    inputs[:, 0] = BEGIN
    for i, word in enumerate(data):
        word_codes = [symbol_codes.get(char, UNKNOWN) for char in word]
        inputs[i, 1:1 + len(word_codes)] = word_codes
        inputs[i, 1 + len(word_codes)] = END
    return inputs
