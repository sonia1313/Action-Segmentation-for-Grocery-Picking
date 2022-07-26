import numpy as np
from numba import njit


from utils.metrics_utils import _get_preds_and_labels, _segment_labels


@njit
#@jit(float64(int64[:], int64[:], boolean), nopython=True)
#@jit(nopython=True)
def _levenshtein_distance(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros((m_row + 1, n_col + 1)).astype(np.float64)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,
                              D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def edit_score(logits, labels, norm=True):
    predictions, labels = _get_preds_and_labels(logits, labels)

    pred_labels = _segment_labels(predictions)
    true_labels = _segment_labels(labels)

    score = _levenshtein_distance(pred_labels, true_labels, norm)

    print(f"edit score: {score}")

    return score
