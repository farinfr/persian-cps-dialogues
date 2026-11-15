#!/usr/bin/env python3
"""LLM-as-judge evaluation of generated dialogues (Table 7 rubric).

Scores a random sample of the new dataset and, for comparison, the dialogues
of the first dataset version (GPT-4o, initial RPB features) on fluency,
accuracy, realism, coverage, helpfulness and overall quality (1-10).

    python scripts/judge_dialogues.py --sample 100 --legacy ../seeder-app/full_dialogues_dataset/llm_generated_dialogues_highest_metric.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.catalog import ROOT, load_catalog  # noqa: E402
from cpsdata.llm import GeminiClient, QuotaExhausted, parse_json  # noqa: E402

METRICS = ["fluency", "accuracy", "realism", "coverage", "helpfulness", "overall"]
SYSTEM = "You are a strict expert evaluator of Persian e-commerce dialogues. Answer only with JSON."
RUBRIC = """Evaluate the Persian customer-assistant dialogue below against the product information.
Give an integer score from 1 to 10 for each criterion:
- fluency: naturalness, smoothness and grammatical correctness of the Persian text.
- accuracy: consistency with the product information; penalise wrong or invented specs or prices.
- realism: believability as a real shopping conversation (realistic questions, no robotic tone).
- coverage: whether the important product aspects are discussed.
- helpfulness: how well the assistant helps the customer reach a purchase decision.
- overall: global quality combining all criteria.

PRODUCT INFORMATION
{product}

DIALOGUE
{dialogue}

Return JSON: {{"fluency": n, "accuracy": n, "realism": n, "coverage": n, "helpfulness": n, "overall": n, "comment": "one sentence"}}"""


def product_text(p) -> str:
    lines = [f"title: {p['title']}", f"brand: {p.get('brand')}", f"category: {p['category']}",
             f"price: {p.get('price'):,.0f} rial" if p.get("price") else "price: unknown"]
    for k, v in list((p.get("specs") or {}).items())[:25]:
        lines.append(f"{k}: {'، '.join(v)[:150]}")
    return "\n".join(lines)


def dialogue_text(turns) -> str:
    return "\n".join(f"{t.get('role')}: {t.get('text')}" for t in turns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "dataset" / "dialogues.jsonl"))
    ap.add_argument("--legacy", default=None, help="first-version dialogues (JSON with generated_dialogues)")
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "judge_scores.jsonl"))
    args = ap.parse_args()

    catalog = {p["id"]: p for p in load_catalog()}
    items = []
    new = [json.loads(line) for line in open(args.dataset, encoding="utf-8")]
    random.Random(0).shuffle(new)
    items += [("v2 (Gemini, corrected RPB)", r["product_id"], r["dialogue"]) for r in new[: args.sample]]
    if args.legacy:
        old = json.load(open(args.legacy, encoding="utf-8"))["generated_dialogues"]
        items += [("v1 (GPT-4o, initial RPB)", int(r["product_id"]), r["dialogue"]) for r in old]

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["set"], r["product_id"]) for r in map(json.loads, out.open(encoding="utf-8"))}
    client = GeminiClient(models=args.models, temperature=0.0, max_tokens=512)
    with out.open("a", encoding="utf-8") as fh:
        for name, pid, turns in items:
            if (name, pid) in done or pid not in catalog:
                continue
            prompt = RUBRIC.format(product=product_text(catalog[pid]), dialogue=dialogue_text(turns))
            try:
                obj = parse_json(client.chat(SYSTEM, prompt))
            except QuotaExhausted as e:
                print(f"stopping: {e}")
                break
            if not obj or not all(isinstance(obj.get(m), (int, float)) for m in METRICS):
                continue
            rec = {"set": name, "product_id": pid, "judge": client.model, **{m: obj[m] for m in METRICS},
                   "comment": obj.get("comment", "")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(name, pid, [obj[m] for m in METRICS])

    rows = [json.loads(line) for line in out.open(encoding="utf-8")]
    summary = {}
    for name in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == name]
        summary[name] = {"n": len(rs), **{m: round(sum(r[m] for r in rs) / len(rs), 2) for m in METRICS}}
    (out.parent / "judge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
