import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    predicted_probs = np.array(predicted_probs)
    true_labels = np.array(true_labels)
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    log_probs = np.log(predicted_probs)
    sample_losses = -np.sum(true_labels * log_probs, axis=1)
    return float(np.mean(sample_losses))