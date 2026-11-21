#!/usr/bin/env python3
"""LaTeX tables and paired differences for the generation-method comparison.

Reads results/method_comparison/judge.jsonl (written by
compare_generation_methods.py) and writes
results/method_comparison/tables.tex and paired_differences.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.catalog import ROOT  # noqa: E402

DIR = ROOT / "results" / "method_comparison"
METRICS = ["fluency", "accuracy", "realism", "coverage", "helpfulness", "overall"]
SETTINGS = ["SPS + RPB + FA", "SPS + Random", "RTS + RPB + FA", "RTS + Random", "DPS + RPB + FA"]
SHOWN = ["overall", "accuracy", "coverage"]


def bootstrap_ci(d: np.ndarray, reps: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(reps, d.size))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    rows = [json.loads(line) for line in (DIR / "judge.jsonl").open(encoding="utf-8")]
    models = sorted({r["model"] for r in rows}, key=lambda m: (m.startswith("gemma"), m))
    mean = lambda sel, k: float(np.mean([r[k] for r in sel]))  # noqa: E731

    # table: averages per model over all settings
    lines = [r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule",
             "Criterion & " + " & ".join(rf"\texttt{{{m}}}" for m in models) + r" \\", r"\midrule"]
    for k in METRICS:
        vals = [mean([r for r in rows if r["model"] == m], k) for m in models]
        best = max(vals)
        cells = [rf"\textbf{{{v:.2f}}}" if abs(v - best) < 1e-9 else f"{v:.2f}" for v in vals]
        lines.append(f"{k.capitalize()} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    by_model = "\n".join(lines)

    # table: settings x models for the shown criteria
    head = " & ".join(rf"\multicolumn{{{len(models)}}}{{c}}{{{k.capitalize()}}}" for k in SHOWN)
    rules = " ".join(rf"\cmidrule(lr){{{2 + i * len(models)}-{1 + (i + 1) * len(models)}}}" for i in range(len(SHOWN)))
    short = {m: m.replace("gemini-", "").replace("-it", "") for m in models}
    lines = [r"\begin{tabular}{l" + "c" * len(models) * len(SHOWN) + "}", r"\toprule",
             f"Setting & {head} \\\\", rules,
             " & " + " & ".join(short[m] for m in models) * 1 + (" & " + " & ".join(short[m] for m in models)) * (len(SHOWN) - 1) + r" \\",
             r"\midrule"]
    table = {(s, m, k): mean([r for r in rows if r["setting"] == s and r["model"] == m], k)
             for s in SETTINGS for m in models for k in SHOWN
             if any(r["setting"] == s and r["model"] == m for r in rows)}
    for s in SETTINGS:
        cells = []
        for k in SHOWN:
            for m in models:
                v = table.get((s, m, k))
                col = [table[(x, m, k)] for x in SETTINGS if (x, m, k) in table]
                cells.append("--" if v is None else (rf"\textbf{{{v:.2f}}}" if abs(v - max(col)) < 1e-9 else f"{v:.2f}"))
        lines.append(f"{s} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    by_setting = "\n".join(lines)

    # paired differences RPB+FA minus Random, pooled over models and products
    diffs = {}
    for strat in ("SPS", "RTS"):
        for k in METRICS:
            d = []
            for m in models:
                a = {r["product_id"]: r[k] for r in rows if r["model"] == m and r["setting"] == f"{strat} + RPB + FA"}
                b = {r["product_id"]: r[k] for r in rows if r["model"] == m and r["setting"] == f"{strat} + Random"}
                d += [a[p] - b[p] for p in a if p in b]
            diffs.setdefault(strat, {})[k] = {"n": len(d), "mean_ci": bootstrap_ci(np.array(d, float))}
    counts = {m: sum(r["model"] == m for r in rows) for m in models}
    (DIR / "paired_differences.json").write_text(json.dumps({"counts": counts, "rpb_minus_random": diffs},
                                                            indent=2), encoding="utf-8")
    (DIR / "tables.tex").write_text(f"% by model\n{by_model}\n\n% by setting\n{by_setting}\n", encoding="utf-8")
    print(by_model, "\n", by_setting, "\n", json.dumps(diffs, indent=1), counts)


if __name__ == "__main__":
    main()
