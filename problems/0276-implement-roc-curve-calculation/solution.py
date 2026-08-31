import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)

    if total_positives == 0 or total_negatives == 0:
        return [0.0, 1.0], [0.0, 1.0]

    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]

    fpr_list = np.r_[0.0, fps / total_negatives].tolist()
    tpr_list = np.r_[0.0, tps / total_positives].tolist()

    return fpr_list, tpr_list