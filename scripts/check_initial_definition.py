#!/usr/bin/env python3
"""Pairwise check of the initial score D(P) against entropy (Section 3.3).

Generates 600 random features with 2-20 values over a cluster of 76 products
(the average cluster size) and counts, over all feature pairs, how often a
score ranks the feature that leaves MORE products on average (sum_i x_i^2 / n)
above the one that leaves fewer.  Also reproduces example (b) of the paper.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.rpb import entropy_real_values, initial_score  # noqa: E402

N = 76
FEATURES = 600


def main() -> None:
    rng = np.random.default_rng(0)
    feats = []
    for _ in range(FEATURES):
        m = int(rng.integers(2, 21))
        sizes = rng.multinomial(N - m, rng.dirichlet(np.ones(m))) + 1   # every value used
        codes = np.repeat(np.arange(m), sizes)
        feats.append((codes, m, float((sizes ** 2).sum() / N)))
    uniform = np.full(N, 1.0 / N)
    results = {}
    scorers = {"entropy": lambda c, m: entropy_real_values(c, uniform, m)}
    for a in (0.1, 0.25, 0.5, 1.0, 2.0):
        scorers[f"initial alpha={a}"] = (lambda a: lambda c, m: initial_score(c, m, a))(a)
    for name, fn in scorers.items():
        s = np.array([fn(c, m) for c, m, _ in feats])
        rem = np.array([r for _, _, r in feats])
        wrong = total = 0
        for i, j in itertools.combinations(range(FEATURES), 2):
            if rem[i] == rem[j] or s[i] == s[j]:
                continue
            total += 1
            wrong += (s[i] > s[j]) != (rem[i] < rem[j])
        results[name] = round(100.0 * wrong / total, 1)
        print(f"{name:22s} discordant pairs: {results[name]:.1f}%")

    binary = np.repeat(np.arange(2), 38)
    dominant = np.repeat(np.arange(5), [40, 9, 9, 9, 9])
    ex = {a: initial_score(binary, 2, a) / initial_score(dominant, 5, a) for a in (0.5, 1.0)}
    print("example (b): D(binary)/D(five-valued) =", {k: round(v, 2) for k, v in ex.items()},
          "| remaining products:", 38, "vs", round((np.bincount(dominant) ** 2).sum() / N, 1))
    out = Path("results") / "initial_definition_check.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"discordant_percent": results, "example_b_ratio": ex}, indent=2))


if __name__ == "__main__":
    main()
