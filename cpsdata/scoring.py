"""Vectorised scores for all features of a cluster at once.

``ClusterScorer`` computes the same quantities as the scalar functions in
``cpsdata.rpb`` (tests/test_rpb.py checks that they agree) but for every
feature in one pass, which makes the simulation fast.
"""

from __future__ import annotations

import numpy as np

from .features import ClusterTable
from .rpb import NEGLIGIBLE_SHARE, initial_score, update_belief


def _xlogx(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = x[pos] * np.log2(x[pos])
    return out


class ClusterScorer:
    def __init__(self, table: ClusterTable):
        self.table = table
        self.codes = table.codes()                       # (n, F)
        self.m = np.array([f.m for f in table.features], dtype=int)
        self.F = self.codes.shape[1]
        n = table.n
        self.offsets = np.concatenate([[0], np.cumsum(self.m)[:-1]]).astype(int) if self.F else np.zeros(0, int)
        M = int(self.m.sum())
        self.onehot = np.zeros((n, M))
        self.has = self.codes >= 0
        for j in range(self.F):
            rows = np.nonzero(self.has[:, j])[0]
            self.onehot[rows, self.offsets[j] + self.codes[rows, j]] = 1.0
        self.seg = np.repeat(np.arange(self.F), self.m)
        counts = self.onehot.sum(axis=0)
        tot = np.add.reduceat(counts, self.offsets) if self.F else np.zeros(0)
        self.catalog_freq = counts / np.maximum(tot[self.seg], 1)  # p_hat per value

    # -- per-feature quantities -------------------------------------------
    def _mass(self, b: np.ndarray):
        mass = b @ self.onehot
        kappa = np.add.reduceat(mass, self.offsets)
        return mass, kappa

    def entropy_real(self, b: np.ndarray) -> np.ndarray:
        mass, kappa = self._mass(b)
        p = mass / np.maximum(kappa[self.seg], 1e-300)
        return -np.add.reduceat(_xlogx(p), self.offsets)

    def entropy_default(self, b: np.ndarray) -> np.ndarray:
        mass, kappa = self._mass(b)
        d = np.clip(1.0 - kappa, 0.0, 1.0)
        return -(np.add.reduceat(_xlogx(mass), self.offsets) + _xlogx(d))

    def rpb(self, b: np.ndarray, delta_hat: np.ndarray, alpha: float,
            use_coverage: bool = True, use_answerability: bool = True) -> np.ndarray:
        mass, kappa = self._mass(b)
        safe = np.maximum(kappa, 1e-300)
        p = mass / safe[self.seg]
        h = -np.add.reduceat(_xlogx(p), self.offsets)
        sumsq = (b ** 2) @ self.has
        n_eff = np.where(sumsq > 0, kappa ** 2 / np.maximum(sumsq, 1e-300), 1.0)
        m_p = np.maximum(2, np.add.reduceat((p >= NEGLIGIBLE_SHARE).astype(float), self.offsets))
        penalty = np.maximum(0.0, np.log2(m_p) - 0.5 * np.log2(np.maximum(n_eff, 1.0)))
        score = np.maximum(0.0, h - alpha * penalty)
        score = np.where((kappa > 0) & (h > 1e-12), score, 0.0)
        if use_coverage:
            score = score * kappa
        if use_answerability:
            score = score * (1.0 - delta_hat)
        return score

    def initial(self, alpha: float) -> np.ndarray:
        return np.array([initial_score(self.codes[:, j], int(self.m[j]), alpha) for j in range(self.F)])

    def update(self, b: np.ndarray, j: int, answer: int, eps: float) -> np.ndarray:
        p_hat = float(self.catalog_freq[self.offsets[j] + answer])
        return update_belief(b, self.codes[:, j], int(self.m[j]), answer, eps, p_hat)
