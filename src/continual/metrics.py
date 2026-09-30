"""Success-rate CL metrics; missing measurements never become zeros.

Adapted from original Continual World results_processing/tables.py at 7841d77:
endpoint success means; forgetting = task-end minus latest success (not max-past);
normalized FWT = (mean learning-curve success - reference mean)/(1-reference mean).
Original tables smooth endpoints over five evaluations. Here an R cell is one
fixed-seed endpoint evaluation; no implicit smoothing. FWT needs separate curves.
"""

import numpy as np


def _matrix(matrix, stage):
    r = np.asarray(matrix, dtype=float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not 0 <= stage < len(r):
        raise ValueError("Need square R and valid completed-stage index.")
    if np.isinf(r).any() or ((r[np.isfinite(r)] < 0) | (r[np.isfinite(r)] > 1)).any():
        raise ValueError("Success rates must be [0,1]; NaN denotes unmeasured.")
    return r


def average_performance(matrix, stage):
    """AP_k = mean_{j=0..k} R[k,j]; None if any required cell is missing."""
    values = _matrix(matrix, stage)[stage, : stage + 1]
    return float(values.mean()) if np.isfinite(values).all() else None


def forgetting(matrix, stage):
    """F_k = mean_{j=0..k}(R[j,j]-R[k,j]); includes current task's zero.

    CW endpoint adaptation; negative values mean improvement. Not max-history
    forgetting. None for missing endpoints; zero for a measured first stage.
    """
    r = _matrix(matrix, stage)
    values = np.diag(r)[: stage + 1] - r[stage, : stage + 1]
    return float(values.mean()) if np.isfinite(values).all() else None


def forward_transfer(learning_curves=None, reference_curves=None):
    """CW normalized FWT across explicitly supplied completed tasks.

    Dictionaries must have identical task IDs; each pair contains success rates
    at the SAME equally spaced task-local training budgets. FWT_i=(A_i-B_i)/(1-B_i)
    with A/B arithmetic means (the original implementation's sampled-AUC proxy).
    Supply task0 too to reproduce CW inclusion of all active tasks. Caller is
    responsible for paired budgets/evaluation protocols. Missing data or B=1
    returns None, never a fabricated score. Endpoint R alone is insufficient.
    """
    if not learning_curves or not reference_curves:
        return None
    if set(learning_curves) != set(reference_curves):
        return None
    scores = []
    for key in learning_curves:
        a, b = (
            np.asarray(learning_curves[key], float),
            np.asarray(reference_curves[key], float),
        )
        if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
            raise ValueError("Need aligned nontrivial curves.")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        if ((a < 0) | (a > 1)).any() or ((b < 0) | (b > 1)).any():
            raise ValueError("Success rates outside [0,1].")
        baseline = float(b.mean())
        if baseline >= 1:
            return None
        scores.append((float(a.mean()) - baseline) / (1 - baseline))
    return float(np.mean(scores))
