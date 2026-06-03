import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    result_array = np.exp(-x)
    result=1 / (1 + result_array)
    return result