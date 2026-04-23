import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores, dtype=float)
    shifted = scores - np.max(scores)
    
    exps = np.exp(shifted)
    probs = exps / np.sum(exps)
    
    return probs.tolist()