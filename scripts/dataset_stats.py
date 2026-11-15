#!/usr/bin/env python3
"""Statistics and automatic consistency checks of the generated dataset.

* size, coverage of clusters/categories, length of dialogues;
* plan fidelity: share of planned (non-binned) values that appear in the dialogue;
* feature tags: share of planned features tagged on an assistant turn;
* requirement-driven opening: the first user message does not name the target
  (fewer than half of the distinctive title tokens appear in it);
* question efficiency of the RPB plan (rank of the target after the answers).
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.catalog import ROOT, normalize_text  # noqa: E402

TOKEN = re.compile(r"[\w‌]+")


def tokens(text: str):
    return [t for t in TOKEN.findall(normalize_text(text).lower()) if len(t) >= 3]


def main() -> None:
    path = ROOT / "dataset" / "dialogues.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    n = len(rows)
    msgs = [len(r["dialogue"]) for r in rows]
    words = [len(tokens(t["text"])) for r in rows for t in r["dialogue"]]
    fidelity, tagged, opening = [], [], []
    for r in rows:
        text = normalize_text(" ".join(t["text"] for t in r["dialogue"]))
        tags = {normalize_text(str(t.get("feature") or "")) for t in r["dialogue"] if t.get("role") == "assistant"}
        for q in r["question_plan"]:
            short = normalize_text(q["feature"].split(" / ", 1)[-1])
            tagged.append(any(short in tg or tg in short for tg in tags if tg))
            if q["value"] is not None and not q["numeric_bin"]:
                fidelity.append(normalize_text(q["value"]) in text)
        cat_tokens = set(tokens(r["category"]))
        title = [t for t in tokens(r["title"]) if t not in cat_tokens]
        first_user = next((t["text"] for t in r["dialogue"] if t["role"] == "user"), "")
        fu = set(tokens(first_user))
        opening.append(not title or sum(t in fu for t in title) / len(title) < 0.5)
    stats = {
        "dialogues": n,
        "clusters": len({r["cluster"] for r in rows}),
        "categories": dict(collections.Counter(r["category"] for r in rows).most_common()),
        "models": dict(collections.Counter(r.get("model") for r in rows)),
        "messages_mean": float(np.mean(msgs)),
        "messages_total": int(sum(msgs)),
        "tokens_per_message_mean": float(np.mean(words)),
        "questions_mean": float(np.mean([len(r["question_plan"]) for r in rows])),
        "dont_know_share": float(np.mean([q["value"] is None for r in rows for q in r["question_plan"]])),
        "distinct_features_asked": len({q["feature"] for r in rows for q in r["question_plan"]}),
        "plan_value_fidelity": float(np.mean(fidelity)) if fidelity else None,
        "plan_feature_tagged": float(np.mean(tagged)) if tagged else None,
        "requirement_driven_opening": float(np.mean(opening)),
        "target_top1": float(np.mean([r["target_rank_after_plan"] == 0 for r in rows])),
    }
    out = ROOT / "results" / "dataset_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
