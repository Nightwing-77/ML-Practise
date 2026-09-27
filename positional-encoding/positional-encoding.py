import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    arr = np.zeros((seq_len, d_model))
    for pos in range(seq_len)   :
        for j in range((d_model + 1) // 2):
            denom=base ** (2 * j/ d_model)
            arr[pos,2*j]=result = np.sin(pos / denom) 
            if 2 * j + 1 < d_model:
                arr[pos, 2 * j + 1] = np.cos(pos / denom)

    return arr