import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def NMI(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)


def ARI(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def ACC(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(np.max(w) - w)
    return sum([w[idx[0], idx[1]] for idx in ind]) * 1.0 / y_pred.size
