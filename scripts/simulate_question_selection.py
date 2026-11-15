#!/usr/bin/env python3
"""Evaluate question-selection strategies on the Digikala catalog (Section 4.3).

The catalog structure (features, values, coverage, popularity) is real; the
user's answers are simulated.  Every strategy sees the same targets and the
same random numbers for the user's answers (common random numbers), so they
differ only in the feature they ask about.

    python scripts/simulate_question_selection.py --out results/simulation
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.catalog import by_cluster, load_catalog  # noqa: E402
from cpsdata.features import build_cluster_table  # noqa: E402
from cpsdata.scoring import ClusterScorer  # noqa: E402

MAX_TURNS = 10
FAIL_TURNS = 11
TARGETS_PER_CLUSTER = 15
TUNE_SHARE = 0.25
PILOT_QUESTIONS = 10
BELIEF_EPS = 0.1
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]
INITIAL_ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]


def true_delta(m: np.ndarray, assumption: int) -> np.ndarray:
    if assumption == 1:
        return np.where(m <= 12, 0.05, 0.5)
    return np.full(m.shape, 0.1)


def error_rate(m: np.ndarray) -> np.ndarray:
    return np.minimum(0.4, 0.04 * np.log2(m))


class Cluster:
    """A cluster with its scorer, simulated user parameters and targets."""

    def __init__(self, table, assumption: int, seed: int):
        self.table = table
        self.sc = ClusterScorer(table) if table.features else None
        self.F = self.sc.F if self.sc else 0
        rng = np.random.default_rng(seed)
        self.prior = table.popularity / table.popularity.sum()
        self.targets = rng.choice(table.n, size=TARGETS_PER_CLUSTER, replace=True, p=self.prior)
        m = self.sc.m if self.sc else np.zeros(0, int)
        self.delta = true_delta(m, assumption)
        self.eps = error_rate(m) if self.F else np.zeros(0)
        # delta_hat from pilot questions with Beta(1,1) smoothing
        u = rng.binomial(PILOT_QUESTIONS, self.delta) if self.F else np.zeros(0)
        self.delta_hat = (u + 1.0) / (PILOT_QUESTIONS + 2.0)
        # common random numbers: (target, feature, 3)
        self.uniforms = rng.random((TARGETS_PER_CLUSTER, max(self.F, 1), 3))
        self.random_order = np.array([rng.permutation(max(self.F, 1)) for _ in range(TARGETS_PER_CLUSTER)])
        self.value_freq = []
        for j in range(self.F):
            col = self.sc.codes[:, j]
            cnt = np.bincount(col[col >= 0], minlength=int(m[j])).astype(float)
            self.value_freq.append(cnt / cnt.sum())

    def answer(self, d: int, j: int):
        """User's answer to feature j in dialogue d: value index or None (don't know)."""
        u0, u1, u2 = self.uniforms[d, j]
        if u0 < self.delta[j]:
            return None
        target = self.targets[d]
        v = int(self.sc.codes[target, j])
        freq = self.value_freq[j]
        if v >= 0:
            if u1 >= self.eps[j]:
                return v
            other = freq.copy()
            other[v] = 0.0
            if other.sum() == 0:
                return v
            other /= other.sum()
            return int(min(np.searchsorted(np.cumsum(other), u2), len(other) - 1))
        return int(min(np.searchsorted(np.cumsum(freq), u2), len(freq) - 1))


def rank_of(b: np.ndarray, t: int) -> int:
    bt = b[t]
    return int((b > bt).sum() + ((b == bt) & (np.arange(b.size) < t)).sum())


def run_dialogue(c: Cluster, d: int, choose: Callable, k: int):
    """Returns (success, turns)."""
    t = c.targets[d]
    b = c.prior.copy()
    if rank_of(b, t) < k:
        return True, 0
    asked = np.zeros(max(c.F, 1), dtype=bool)
    for turn in range(1, MAX_TURNS + 1):
        if c.F == 0 or asked[:c.F].all():
            break
        j = choose(c, d, b, asked)
        asked[j] = True
        a = c.answer(d, j)
        if a is not None:
            b = c.sc.update(b, j, a, BELIEF_EPS)
        if rank_of(b, t) < k:
            return True, turn
    return False, FAIL_TURNS


def masked_argmax(scores: np.ndarray, asked: np.ndarray) -> int:
    s = np.where(asked[: scores.size], -np.inf, scores)
    return int(np.argmax(s))


def make_strategies(alpha_rpb: float, alpha_init: float) -> Dict[str, Callable]:
    static_cache: Dict[int, np.ndarray] = {}

    def random_(c, d, b, asked):
        for j in c.random_order[d]:
            if not asked[j]:
                return int(j)

    def initial(c, d, b, asked):
        key = id(c)
        if key not in static_cache:
            static_cache[key] = c.sc.initial(alpha_init)
        return masked_argmax(static_cache[key], asked)

    return {
        "Random": random_,
        "Initial D(P)": initial,
        "Entropy, default group": lambda c, d, b, a: masked_argmax(c.sc.entropy_default(b), a),
        "Entropy, real values": lambda c, d, b, a: masked_argmax(c.sc.entropy_real(b), a),
        "Corrected RPB": lambda c, d, b, a: masked_argmax(c.sc.rpb(b, c.delta_hat, alpha_rpb), a),
        "Corrected RPB, no coverage": lambda c, d, b, a: masked_argmax(
            c.sc.rpb(b, c.delta_hat, alpha_rpb, use_coverage=False), a),
        "Corrected RPB, no answerability": lambda c, d, b, a: masked_argmax(
            c.sc.rpb(b, c.delta_hat, alpha_rpb, use_answerability=False), a),
        "Corrected RPB, alpha=0": lambda c, d, b, a: masked_argmax(c.sc.rpb(b, c.delta_hat, 0.0), a),
    }


def evaluate(clusters: List[Cluster], choose: Callable, k: int):
    succ, turns = [], []
    for c in clusters:
        for d in range(TARGETS_PER_CLUSTER):
            s, t = run_dialogue(c, d, choose, k)
            succ.append(s)
            turns.append(t)
    return np.array(succ, dtype=float), np.array(turns, dtype=float)


def bootstrap_diff(x: np.ndarray, y: np.ndarray, reps: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    diff = x - y
    idx = rng.integers(0, diff.size, size=(reps, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/simulation")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    products = load_catalog()
    tables = [build_cluster_table(c, ps) for c, ps in sorted(by_cluster(products).items())]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(tables))
    n_tune = int(round(TUNE_SHARE * len(tables)))
    tune_ids, eval_ids = set(order[:n_tune].tolist()), set(order[n_tune:].tolist())
    print(f"{len(tables)} clusters: {len(tune_ids)} for tuning, {len(eval_ids)} for evaluation")

    report = {"clusters_tune": len(tune_ids), "clusters_eval": len(eval_ids),
              "targets_per_cluster": TARGETS_PER_CLUSTER, "assumptions": {}}
    for assumption in (1, 2):
        clusters = [Cluster(tb, assumption, args.seed * 1000 + i) for i, tb in enumerate(tables)]
        tune = [clusters[i] for i in sorted(tune_ids)]
        evalc = [clusters[i] for i in sorted(eval_ids)]

        # tune alpha on the tuning clusters (k=1, average turns)
        def tuned(name: str, grid: List[float]) -> float:
            best = None
            for a in grid:
                strat = make_strategies(a, a)[name]
                _, turns = evaluate(tune, strat, 1)
                if best is None or turns.mean() < best[1] - 1e-12:
                    best = (a, turns.mean())
                print(f"  A{assumption} tune {name} alpha={a}: {turns.mean():.3f}")
            return best[0]

        a_rpb = tuned("Corrected RPB", ALPHAS)
        a_init = tuned("Initial D(P)", INITIAL_ALPHAS)
        strategies = make_strategies(a_rpb, a_init)
        res = {"alpha_rpb": a_rpb, "alpha_initial": a_init, "k": {}}
        for k in (1, 3):
            per = {}
            raw = {}
            for name, strat in strategies.items():
                s, t = evaluate(evalc, strat, k)
                raw[name] = (s, t)
                per[name] = {"success": 100 * s.mean(), "turns": t.mean(), "n": int(s.size)}
                print(f"A{assumption} k={k} {name:34s} success={100 * s.mean():5.1f}% turns={t.mean():.2f}")
            diffs = {}
            s_r, t_r = raw["Corrected RPB"]
            for other in ("Entropy, default group", "Entropy, real values", "Initial D(P)"):
                s_o, t_o = raw[other]
                diffs[other] = {
                    "turns": bootstrap_diff(t_r, t_o),
                    "success_pp": tuple(100 * x for x in bootstrap_diff(s_r, s_o)),
                }
            res["k"][str(k)] = {"strategies": per, "paired_differences": diffs}
        report["assumptions"][str(assumption)] = res

    stats = {"features": 0, "single_valued": 0, "excluded_multi_valued": 0}
    cov = []
    for tb in tables:
        for key in stats:
            stats[key] += tb.stats.get(key, 0)
        cov += [float((f.codes >= 0).mean()) for f in tb.features]
    stats["features_per_cluster"] = stats["features"] / len(tables)
    stats["median_coverage"] = float(np.median(cov))
    stats["share_coverage_below_half"] = float(np.mean(np.array(cov) < 0.5))
    report["catalog"] = stats
    report["runtime_seconds"] = time.time() - t0
    (out / "simulation_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved {out / 'simulation_results.json'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
