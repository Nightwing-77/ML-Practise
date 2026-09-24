import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    arr = np.array(x, dtype=np.float64)
    
    high = np.max(arr, axis=-1, keepdims=True)
    arr = arr - high
    
    arr = np.exp(arr)
    total = np.sum(arr, axis=-1, keepdims=True)
    arr = arr / total
    
    return arr