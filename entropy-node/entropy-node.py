import numpy as np
import math
def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    counts = np.bincount(y)
    probs = counts / len(y)
    total=0.0
    for prob in probs:
        if prob>0:
            entropy=prob*math.log2(prob)
            total=total+entropy
    return -total