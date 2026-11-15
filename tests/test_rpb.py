"""Checks of the RPB score's properties and of the vectorised scorer."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.features import ClusterTable, Feature  # noqa: E402
from cpsdata.rpb import entropy_default_group, entropy_real_values, rpb  # noqa: E402
from cpsdata.scoring import ClusterScorer  # noqa: E402


def uniform(n):
    return np.full(n, 1.0 / n)


def test_single_value_scores_zero_even_with_default_group():
    codes = np.array([0] * 38 + [-1] * 38)
    assert rpb(codes, uniform(76), 1, alpha=0.1) == 0.0


def test_equals_entropy_in_ideal_case():
    codes = np.repeat(np.arange(4), 19)
    assert math.isclose(rpb(codes, uniform(76), 4, delta_hat=0, alpha=0), 2.0)


def test_coverage_scales_score():
    full = np.repeat(np.arange(2), 38)
    half = np.concatenate([np.repeat(np.arange(2), 19), -np.ones(38, int)])
    assert math.isclose(rpb(half, uniform(76), 2, alpha=0), 0.5 * rpb(full, uniform(76), 2, alpha=0))


def test_vectorised_matches_scalar():
    rng = np.random.default_rng(0)
    n = 60
    feats = []
    for j in range(8):
        m = int(rng.integers(2, 9))
        codes = rng.integers(-1, m, size=n).astype(np.int32)
        codes[: m] = np.arange(m)  # every value used
        feats.append(Feature(f"f{j}", codes, [str(i) for i in range(m)]))
    table = ClusterTable(0, [{}] * n, np.ones(n), feats)
    sc = ClusterScorer(table)
    b = rng.random(n)
    b /= b.sum()
    dh = rng.random(len(feats)) * 0.5
    vec = sc.rpb(b, dh, 0.25)
    ent_d = sc.entropy_default(b)
    ent_r = sc.entropy_real(b)
    for j, f in enumerate(feats):
        assert math.isclose(vec[j], rpb(f.codes, b, f.m, dh[j], 0.25), rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(ent_d[j], entropy_default_group(f.codes, b, f.m), rel_tol=1e-9)
        assert math.isclose(ent_r[j], entropy_real_values(f.codes, b, f.m), rel_tol=1e-9)
