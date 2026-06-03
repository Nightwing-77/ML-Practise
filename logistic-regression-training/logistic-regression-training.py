import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    x = np.array(X)
    y = np.array(y)
    
    num_samples = x.shape[0]
    num_features = x.shape[1]
    
    # 1. Initialize weights based on the number of FEATURES, not samples
    bias = 0.0
    weight = np.zeros(num_features) 
    
    # 2. Gradient Descent Loop
    for _ in range(steps):
        # Forward pass: Calculate predictions for ALL samples at once
        # Using np.dot(x, weight) handles matrix-vector multiplication
        z = np.dot(x, weight) + bias
        output = _sigmoid(z)
        
        # Calculate the error vector (predictions - true labels)
        error = output - y
        
        # Compute gradients for the entire batch
        dw = (1 / num_samples) * np.dot(x.T, error)
        db = (1 / num_samples) * np.sum(error)
        
        # Update parameters
        weight = weight - lr * dw
        bias = bias - lr * db
        
    return weight, bias