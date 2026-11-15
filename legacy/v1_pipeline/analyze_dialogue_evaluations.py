#!/usr/bin/env python3
"""
Analyze LLM Judge Dialogue Evaluations
======================================

This script:
  - Loads per-product evaluation files from `dialogue_evaluations_llm_judge/`.
  - Aggregates scores per `kind` across all products.
  - Computes average scores for each metric (fluency, accuracy, realism, coverage,
    helpfulness, overall).
  - Saves a summary JSON and CSV.
  - Generates comparison plots to help decide which dialogue kind is best overall.
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt

EVAL_DIR = "dialogue_evaluations_llm_judge"
OUTPUT_SUMMARY_JSON = os.path.join(EVAL_DIR, "kinds_scores_summary.json")
OUTPUT_SUMMARY_CSV = os.path.join(EVAL_DIR, "kinds_scores_summary.csv")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")

METRICS = ["fluency", "accuracy", "realism", "coverage", "helpfulness", "overall"]


def load_all_evaluations() -> List[Dict[str, Any]]:
    """Load all per-product evaluation_* JSON files."""
    evaluations: List[Dict[str, Any]] = []
    for fname in os.listdir(EVAL_DIR):
        if not fname.startswith("evaluation_") or not fname.endswith(".json"):
            continue
        path = os.path.join(EVAL_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        evaluations.append(data)
    return evaluations


def aggregate_by_kind(evaluations: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Aggregate metric scores per kind across all products.

    Returns:
      { kind: {
          "kind_label": str,
          "count": int,
          "metrics": { metric: {"sum": float, "n": int} },
        }
      }
    """
    agg: Dict[int, Dict[str, Any]] = {}

    for ev in evaluations:
        parsed = ev.get("judge_output_parsed") or {}
        per_dialogue = parsed.get("per_dialogue") or []

        for entry in per_dialogue:
            kind = entry.get("kind")
            label = entry.get("kind_label") or ""
            scores = entry.get("scores") or {}
            if kind is None:
                continue

            if kind not in agg:
                agg[kind] = {
                    "kind_label": label,
                    "count": 0,
                    "metrics": {m: {"sum": 0.0, "n": 0} for m in METRICS},
                }
            agg[kind]["count"] += 1

            for m in METRICS:
                val = scores.get(m)
                if isinstance(val, (int, float)):
                    agg[kind]["metrics"][m]["sum"] += float(val)
                    agg[kind]["metrics"][m]["n"] += 1

    # Compute averages
    for kind, info in agg.items():
        averages: Dict[str, float] = {}
        for m in METRICS:
            s = info["metrics"][m]["sum"]
            n = info["metrics"][m]["n"]
            averages[m] = s / n if n > 0 else 0.0
        info["averages"] = averages

    return agg


def save_summary(agg: Dict[int, Dict[str, Any]]) -> None:
    """Save summary to JSON and CSV."""
    # JSON
    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    # CSV (one row per kind)
    lines: List[str] = []
    header = ["kind", "kind_label", "count"] + [f"avg_{m}" for m in METRICS]
    lines.append(",".join(header))

    for kind in sorted(agg.keys()):
        info = agg[kind]
        row = [
            str(kind),
            f"\"{info.get('kind_label', '')}\"",
            str(info.get("count", 0)),
        ]
        averages = info.get("averages", {})
        for m in METRICS:
            row.append(f"{averages.get(m, 0.0):.3f}")
        lines.append(",".join(row))

    with open(OUTPUT_SUMMARY_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_metric_bars(agg: Dict[int, Dict[str, Any]]) -> None:
    """Create bar plots per metric comparing kinds."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    kinds = sorted(agg.keys())
    labels = [f"{k} - {agg[k]['kind_label']}" for k in kinds]

    for metric in METRICS:
        scores = [agg[k]["averages"][metric] for k in kinds]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(kinds)), scores, color="skyblue")
        plt.xticks(range(len(kinds)), labels, rotation=45, ha="right")
        plt.ylabel("امتیاز (1 تا 10)")
        plt.title(f"مقایسه انواع گفتگو بر اساس معیار: {metric}")
        plt.ylim(0, 10)

        # Annotate bars
        for bar, score in zip(bars, scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        out_path = os.path.join(PLOTS_DIR, f"{metric}_by_kind.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def main() -> None:
    print("Analyzing LLM judge evaluations...")

    evaluations = load_all_evaluations()
    print(f"Loaded {len(evaluations)} per-product evaluation files.")

    agg = aggregate_by_kind(evaluations)

    print("\nAverage scores per kind:")
    for kind in sorted(agg.keys()):
        info = agg[kind]
        label = info["kind_label"]
        averages = info["averages"]
        print(f"Kind {kind} ({label}) - n={info['count']}:")
        for m in METRICS:
            print(f"  {m}: {averages[m]:.2f}")

    save_summary(agg)
    print(f"\nSaved summary JSON to {OUTPUT_SUMMARY_JSON}")
    print(f"Saved summary CSV to  {OUTPUT_SUMMARY_CSV}")

    plot_metric_bars(agg)
    print(f"Saved bar plots per metric to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()


