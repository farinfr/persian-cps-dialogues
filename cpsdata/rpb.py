"""Question-selection scores: RPB and entropy.

All scores take a feature column ``codes`` (value index per product, -1 for the
default group of products that lack the feature) and, for the dynamic scores,
the current belief ``b`` over the cluster's products (sums to one).
See Section 3.3 of the paper for the definitions.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

NEGLIGIBLE_SHARE = 0.01  # values below this share do not count towards m_P


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def value_mass(codes: np.ndarray, b: np.ndarray, m: int) -> np.ndarray:
    has = codes >= 0
    return np.bincount(codes[has], weights=b[has], minlength=m)


def rpb(codes: np.ndarray, b: np.ndarray, m: int, delta_hat: float = 0.0,
        alpha: float = 0.1, use_coverage: bool = True,
        use_answerability: bool = True) -> float:
    """Regularized Partition Balance score, Eq. (2) of the paper.

    RPB(P|b) = (1 - delta_hat) * kappa * [H(P) - alpha * max(0, log2 m_P - 0.5 log2 N_P)]_+
    """
    has = codes >= 0
    mass = np.bincount(codes[has], weights=b[has], minlength=m)
    kappa = float(mass.sum())
    if kappa <= 0:
        return 0.0
    p = mass / kappa
    h = _entropy(p)
    if h <= 0:
        return 0.0
    w = b[has]
    n_eff = kappa ** 2 / float((w ** 2).sum())          # Kish's effective sample size
    m_p = max(2, int((p >= NEGLIGIBLE_SHARE).sum()))
    penalty = max(0.0, math.log2(m_p) - 0.5 * math.log2(max(n_eff, 1.0)))
    core = max(0.0, h - alpha * penalty)
    score = core
    if use_coverage:
        score *= kappa
    if use_answerability:
        score *= (1.0 - delta_hat)
    return score


def entropy_default_group(codes: np.ndarray, b: np.ndarray, m: int) -> float:
    """Entropy of the partition in which the default group counts as a value."""
    mass = np.bincount(np.where(codes >= 0, codes, m), weights=b, minlength=m + 1)
    total = mass.sum()
    return _entropy(mass / total) if total > 0 else 0.0


def entropy_real_values(codes: np.ndarray, b: np.ndarray, m: int) -> float:
    """Entropy over the real values only (ignores coverage)."""
    mass = value_mass(codes, b, m)
    total = mass.sum()
    return _entropy(mass / total) if total > 0 else 0.0



def update_belief(b: np.ndarray, codes: np.ndarray, m: int, answer: int,
                  eps: float = 0.1, p_hat: Optional[float] = None) -> np.ndarray:
    """Soft belief update, Eq. (5) of the paper.

    Products with the answered value keep weight 1-eps, products with another
    value get eps/(m-1), products lacking the feature get the catalog
    frequency p_hat of the answered value.
    """
    if p_hat is None:
        counts = np.bincount(codes[codes >= 0], minlength=m)
        p_hat = counts[answer] / max(1, counts.sum())
    like = np.where(codes == answer, 1.0 - eps,
                    np.where(codes >= 0, eps / max(1, m - 1), p_hat))
    nb = b * like
    s = nb.sum()
    return nb / s if s > 0 else b
