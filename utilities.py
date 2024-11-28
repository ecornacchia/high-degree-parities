import itertools
import numpy as np


def create_test_matrix_11(rows, cols, p=0.5):
    return 1 - 2 * np.random.binomial(1, p, size=(rows, cols)).astype(np.float32)
