#!/usr/bin/env python3
"""Compare dialogue-generation strategies and LLMs (Section 4.4 of the paper).

Ten target products (at least one per category, not in the released dataset)
are verbalised with five settings and every generator model:

* SPS + RPB + FA  : single prompt, RPB question plan, contextual augmentation
* SPS + Random    : single prompt, five random features of the target, no augmentation
* RTS + RPB + FA  : template dialogue built from the RPB plan, rewritten by the LLM
* RTS + Random    : template dialogue from five random features, rewritten by the LLM
* DPS + RPB + FA  : two LLM agents (customer and seller) converse turn by turn

Every dialogue is then scored by one fixed LLM judge with the rubric of
``judge_dialogues.py``.  The script is resumable and stops cleanly when the
free daily quota of a model is exhausted.

    python scripts/compare_generation_methods.py --dry-run
    python scripts/compare_generation_methods.py                # generate + judge
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpsdata.augment import augment_cluster, load_embeddings  # noqa: E402
from cpsdata.catalog import ROOT, by_cluster, load_catalog  # noqa: E402
from cpsdata.features import PRICE, build_cluster_table  # noqa: E402
from cpsdata.llm import GeminiClient, QuotaExhausted, parse_json  # noqa: E402
from cpsdata.scoring import ClusterScorer  # noqa: E402
from generate_dialogues import (MAX_QUESTIONS, SYSTEM, build_prompt, coerce, fmt_price,  # noqa: E402
                                plan_questions, product_block, product_type, short_name, validate)
from judge_dialogues import METRICS, RUBRIC, dialogue_text, product_text  # noqa: E402
from judge_dialogues import SYSTEM as JUDGE_SYSTEM  # noqa: E402

OUT_DIR = ROOT / "results" / "method_comparison"
SETTINGS = ["SPS + RPB + FA", "SPS + Random", "RTS + RPB + FA", "RTS + Random", "DPS + RPB + FA"]
DEFAULT_GENERATORS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemma-4-26b-a4b-it"]
DEFAULT_JUDGE = "gemini-3.1-flash-lite-preview"
DPS_MAX_MESSAGES = 14
NO_AUG = {"summary": {}, "labels": []}


# ---------------------------------------------------------------- products and plans

def select_products(products, tables, n: int, seed: int) -> List[tuple]:
    """n targets with at least one per category, at least five features, not in the dataset."""
    used = set()
    ds = ROOT / "dataset" / "dialogues.jsonl"
    if ds.exists():
        used = {json.loads(line)["product_id"] for line in ds.open(encoding="utf-8")}
    rng = np.random.default_rng(seed)
    pool = defaultdict(list)
    for c, t in tables.items():
        if len(t.features) < 5:
            continue
        for row, p in enumerate(t.products):
            has = sum(f.codes[row] >= 0 for f in t.features)
            if p["id"] not in used and has >= 5 and p.get("rating_count"):
                pool[p["category"]].append((c, row))
    cats = sorted(pool)
    chosen = [pool[cat][rng.integers(len(pool[cat]))] for cat in cats]
    rest = [x for cat in cats for x in pool[cat] if x not in chosen]
    for i in rng.permutation(len(rest))[: max(0, n - len(chosen))]:
        chosen.append(rest[int(i)])
    return chosen[:n]


def random_plan(table, row: int, seed: int) -> Dict[str, Any]:
    """Five random features that the target has (the 'Random' baseline)."""
    rng = np.random.default_rng(seed)
    has = [j for j, f in enumerate(table.features) if f.codes[row] >= 0 and not f.free_text]
    pick = [has[int(i)] for i in rng.permutation(len(has))[:MAX_QUESTIONS]]
    plan = [{"feature": table.features[j].name, "value": table.features[j].value_of(row),
             "numeric_bin": table.features[j].numeric} for j in pick]
    return {"plan": plan}


def feature_label(q) -> str:
    return "قیمت (بازه)" if q["feature"] == PRICE else short_name(q["feature"])


def answer_text(q) -> str:
    return q["value"] if q["value"] is not None else "نمی‌دانم / برایم مهم نیست"


# ---------------------------------------------------------------- the three strategies

def rts_template(target, plan, alts) -> List[Dict[str, Any]]:
    """Rule-based dialogue following the feature-based filtering flow."""
    turns = [{"role": "user", "text": f"سلام، دنبال {product_type(target)} هستم.", "feature": None}]
    for q in plan["plan"]:
        name = feature_label(q)
        turns.append({"role": "assistant", "text": f"{name} مورد نظرتان چیست؟", "feature": q["feature"]})
        turns.append({"role": "user", "text": f"{answer_text(q)}.", "feature": None})
    names = [target] + [a for a, _ in alts]
    offer = "؛ ".join(f"{p['title']} با قیمت {fmt_price(p.get('price'))}" for p in names)
    turns.append({"role": "assistant", "text": f"این محصولات با نیاز شما سازگارند: {offer}.", "feature": None})
    turns.append({"role": "user", "text": f"{target['title']} را انتخاب می‌کنم.", "feature": None})
    return turns


def rts_prompt(target, aug, plan, alts) -> str:
    tmpl = "\n".join(f"{t['role']}: {t['text']}" for t in rts_template(target, plan, alts))
    plan_features = [q["feature"] for q in plan["plan"]]
    alt_blocks = "\n".join(f"--- گزینه {i + 1} ---\n{product_block(a, a_aug, [])[:900]}"
                           for i, (a, a_aug) in enumerate(alts))
    return f"""گفتگوی قالبیِ زیر بین «user» (مشتری) و «assistant» (دستیار فروشگاه اینترنتی) را به یک گفتگوی فارسیِ طبیعی، روان و محاوره‌ای مؤدبانه بازنویسی کنید.
قواعد: ترتیب پرسش‌ها، ویژگی‌ها و پاسخ‌های کاربر را حفظ کنید؛ می‌توانید جمله‌های طبیعی، توضیح کوتاه ویژگی‌ها و مقایسه‌ی محصولات را بر اساس اطلاعات زیر اضافه کنید؛ هیچ مشخصه، قیمت یا ادعایی خارج از اطلاعات داده‌شده ننویسید.

گفتگوی قالبی:
{tmpl}

محصول هدف:
{product_block(target, aug, plan_features)}

گزینه‌های جایگزین:
{alt_blocks or '(ندارد)'}

خروجی را فقط به صورت JSON برگردانید:
{{"dialogue": [{{"role": "user", "text": "...", "feature": null}}, {{"role": "assistant", "text": "...", "feature": "نام ویژگی یا null"}}], "summary": "خلاصه‌ی یک‌جمله‌ای"}}"""


def dps_seller_system(target, aug, plan, alts) -> str:
    plan_lines = "\n".join(f"{i}. {feature_label(q)}" for i, q in enumerate(plan["plan"], 1))
    alt_blocks = "\n".join(f"--- گزینه {i + 1} ---\n{product_block(a, a_aug, [])[:900]}"
                           for i, (a, a_aug) in enumerate(alts))
    return f"""شما دستیار فروش یک فروشگاه اینترنتی ایرانی هستید و به فارسی محاوره‌ای مؤدبانه با مشتری حرف می‌زنید.
هدف: با پرسیدن ویژگی‌های زیر، به ترتیب و در هر نوبت فقط یک پرسش، نیاز مشتری را بفهمید؛ سپس ۲ تا ۳ محصول از فهرست زیر را با قیمت دقیق پیشنهاد و مقایسه کنید و به پرسش‌های مشتری فقط بر اساس همین اطلاعات پاسخ دهید.
ویژگی‌هایی که باید بپرسید:
{plan_lines}

محصولات در دسترس:
{product_block(target, aug, [q['feature'] for q in plan['plan']])}
{alt_blocks}

هر بار فقط متن پیام بعدی خودتان را بنویسید (یک تا سه جمله)، بدون برچسب نقش."""


def dps_customer_system(target, plan) -> str:
    need = "\n".join(f"- {feature_label(q)}: {answer_text(q)}" for q in plan["plan"])
    return f"""شما مشتری یک فروشگاه اینترنتی ایرانی هستید و به فارسی محاوره‌ای حرف می‌زنید.
دنبال «{product_type(target)}» در دسته‌ی «{target['category']}» هستید، اما برند، مدل یا نام دقیق محصول را نمی‌دانید و نمی‌گویید.
نیازهای شما (فقط وقتی پرسیده شد بگویید؛ اعداد را تقریبی بیان کنید):
{need}
وقتی فروشنده محصولاتی پیشنهاد داد، یک سؤال مقایسه‌ای یا درباره‌ی قیمت بپرسید و سپس محصولی را انتخاب کنید که با نیازهایتان سازگارتر است، و با جمله‌ای که شامل کلمه‌ی «انتخاب» است خرید را تمام کنید.
هر بار فقط متن پیام بعدی خودتان را بنویسید (یک تا دو جمله)، بدون برچسب نقش."""


def run_dps(client: GeminiClient, target, aug, plan, alts) -> Dict[str, Any]:
    seller_sys = dps_seller_system(target, aug, plan, alts)
    cust_sys = dps_customer_system(target, plan)
    turns: List[Dict[str, Any]] = []
    role = "user"
    while len(turns) < DPS_MAX_MESSAGES:
        history = "\n".join(f"{'مشتری' if t['role'] == 'user' else 'فروشنده'}: {t['text']}" for t in turns)
        ask = (f"گفتگو تا اینجا:\n{history or '(هنوز شروع نشده؛ شما گفتگو را شروع می‌کنید)'}\n\n"
               f"پیام بعدی شما:")
        text = client.chat(cust_sys if role == "user" else seller_sys, ask, json_mode=False).strip()
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        text = text.removeprefix("مشتری:").removeprefix("فروشنده:").strip()
        turns.append({"role": role, "text": text, "feature": None})
        if role == "user" and len(turns) > 4 and "انتخاب" in text:
            break
        role = "assistant" if role == "user" else "user"
    return {"dialogue": turns, "summary": ""}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--generators", nargs="*", default=DEFAULT_GENERATORS)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--interval", type=float, default=6.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    products = load_catalog()
    clusters = by_cluster(products)
    tables = {c: build_cluster_table(c, ps) for c, ps in clusters.items()}
    targets = select_products(products, tables, args.n, args.seed)
    emb = load_embeddings()
    print(f"{len(targets)} products:", ", ".join(f"{tables[c].products[r]['id']} ({tables[c].products[r]['category']})"
                                               for c, r in targets))

    jobs = []
    for c, row in targets:
        table = tables[c]
        sc = ClusterScorer(table)
        aug = augment_cluster(clusters[c], emb)
        target = table.products[row]
        rpb = plan_questions(table, sc, row)
        rnd = random_plan(table, row, args.seed + target["id"])
        alts_rpb = [(table.products[i], aug[table.products[i]["id"]]) for i in rpb["candidates"][1:3]]
        alts_rnd = [(p, NO_AUG) for p, _ in alts_rpb]  # same alternatives, without augmentation
        jobs.append({"target": target, "aug": aug[target["id"]], "rpb": rpb, "rnd": rnd,
                     "alts_rpb": alts_rpb, "alts_rnd": alts_rnd})

    if args.dry_run:
        j = jobs[0]
        print(rts_prompt(j["target"], NO_AUG, j["rnd"], j["alts_rnd"]))
        print("=" * 60)
        print(dps_customer_system(j["target"], j["rpb"]))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gen_path, judge_path = OUT_DIR / "dialogues.jsonl", OUT_DIR / "judge.jsonl"
    done = set()
    if gen_path.exists():
        done = {(r["model"], r["setting"], r["product_id"])
                for r in map(json.loads, gen_path.open(encoding="utf-8"))}

    with gen_path.open("a", encoding="utf-8") as fh:
        for model in args.generators:
            client = GeminiClient(models=[model], min_interval=args.interval)
            try:
                for j in jobs:
                    t = j["target"]
                    for setting in SETTINGS:
                        key = (model, setting, t["id"])
                        if key in done:
                            continue
                        rpb_set = "RPB" in setting
                        plan = j["rpb"] if rpb_set else j["rnd"]
                        aug = j["aug"] if rpb_set else NO_AUG
                        alts = j["alts_rpb"] if rpb_set else j["alts_rnd"]
                        obj = None
                        try:
                            if setting.startswith("DPS"):
                                obj = run_dps(client, t, aug, plan, alts)
                            else:
                                prompt = (build_prompt(t, aug, plan, alts) if setting.startswith("SPS")
                                          else rts_prompt(t, aug, plan, alts))
                                for _ in range(2):
                                    obj = coerce(parse_json(client.chat(SYSTEM, prompt)))
                                    if validate(obj):
                                        break
                                    obj = None
                        except QuotaExhausted:
                            raise
                        except Exception as e:  # noqa: BLE001  server errors: skip, a later run retries
                            print(f"  {model} {setting} {t['id']}: {str(e)[:100]}; skipped")
                            obj = None
                        if obj is None:
                            print(f"  {model} {setting} {t['id']}: invalid output, skipped")
                            continue
                        rec = {"model": model, "setting": setting, "product_id": t["id"],
                               "plan": [q["feature"] for q in plan["plan"]], "dialogue": obj["dialogue"]}
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                        done.add(key)
                        print(f"  {model:24s} {setting:15s} {t['id']}: {len(obj['dialogue'])} messages")
            except QuotaExhausted as e:
                print(f"{model}: {e}")

    if args.skip_judge:
        return
    catalog = {p["id"]: p for p in products}
    judged = set()
    if judge_path.exists():
        judged = {(r["model"], r["setting"], r["product_id"])
                  for r in map(json.loads, judge_path.open(encoding="utf-8"))}
    judge = GeminiClient(models=[args.judge], min_interval=args.interval, temperature=0.0)
    with judge_path.open("a", encoding="utf-8") as fh:
        try:
            for r in map(json.loads, gen_path.open(encoding="utf-8")):
                key = (r["model"], r["setting"], r["product_id"])
                if key in judged:
                    continue
                prompt = RUBRIC.format(product=product_text(catalog[r["product_id"]]),
                                       dialogue=dialogue_text(r["dialogue"]))
                try:
                    s = parse_json(judge.chat(JUDGE_SYSTEM, prompt))
                except QuotaExhausted:
                    raise
                except Exception as e:  # noqa: BLE001
                    print(f"  judge error for {key}: {str(e)[:100]}")
                    continue
                if not isinstance(s, dict) or not all(isinstance(s.get(m), (int, float)) for m in METRICS):
                    print(f"  judge output unreadable for {key}")
                    continue
                fh.write(json.dumps({"model": r["model"], "setting": r["setting"], "product_id": r["product_id"],
                                     "judge": args.judge, **{m: s[m] for m in METRICS}},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                judged.add(key)
        except QuotaExhausted as e:
            print(f"judge: {e}")

    rows = [json.loads(line) for line in judge_path.open(encoding="utf-8")]
    summary: Dict[str, Any] = {"judge": args.judge, "n_products": len(jobs), "by_model": {}, "by_setting": {}}
    for m in args.generators:
        sel = [r for r in rows if r["model"] == m]
        if sel:
            summary["by_model"][m] = {"n": len(sel), **{k: round(float(np.mean([r[k] for r in sel])), 2)
                                                         for k in METRICS}}
        for s in SETTINGS:
            ss = [r for r in sel if r["setting"] == s]
            if ss:
                summary["by_setting"].setdefault(s, {})[m] = {
                    "n": len(ss), **{k: round(float(np.mean([r[k] for r in ss])), 2) for k in METRICS}}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
