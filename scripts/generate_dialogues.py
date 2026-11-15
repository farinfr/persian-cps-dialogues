#!/usr/bin/env python3
"""Generate the Persian conversational product-search dataset with a free LLM.

For each target product the RPB score (Eq. 2) chooses, turn by
turn, which feature the assistant asks about next (Algorithm 1, popularity
prior, soft belief update).  The resulting question plan, the contextual
augmentation and two alternative products are given to Gemini with the
Single Prompt Strategy (SPS+FA), which verbalises the whole dialogue.

    python scripts/generate_dialogues.py --n 1000          # resumable
    python scripts/generate_dialogues.py --dry-run         # show one prompt, no API call
"""

from __future__ import annotations

import argparse
import re
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.augment import LABEL_TEXT, PRICE_POSITION_TEXT, augment_cluster, load_embeddings  # noqa: E402
from cpsdata.catalog import ROOT, by_cluster, load_catalog  # noqa: E402
from cpsdata.features import PRICE, build_cluster_table  # noqa: E402
from cpsdata.scoring import ClusterScorer  # noqa: E402

PROMPT_VERSION = "sps-fa-rpb-v3"
ALPHA = 0.1                  # selected on the tuning clusters (Section 4.3, assumption 1)
MAX_QUESTIONS = 5
MIN_QUESTIONS = 3
BELIEF_EPS = 0.1
OUT = ROOT / "dataset" / "dialogues.jsonl"

SYSTEM = ("شما یک متخصص تولید گفتگوهای طبیعی و واقعی بین مشتری و دستیار یک فروشگاه اینترنتی ایرانی هستید. "
          "فقط از اطلاعات داده‌شده استفاده می‌کنید و هیچ مشخصه‌ای از خودتان نمی‌سازید. خروجی فقط JSON است.")


def delta_by_type(table) -> np.ndarray:
    """'Don't know' rate by feature type when no dialogue data is available:
    general features (at most 12 values) 0.05, technical features (more values) 0.5,
    free-text descriptions 0.95."""
    m = np.array([f.m for f in table.features])
    free = np.array([f.free_text for f in table.features])
    return np.where(free, 0.95, np.where(m <= 12, 0.05, 0.5))


def rank_of(b: np.ndarray, t: int) -> int:
    return int((b > b[t]).sum() + ((b == b[t]) & (np.arange(b.size) < t)).sum())


def plan_questions(table, sc: ClusterScorer, row: int) -> Dict[str, Any]:
    """Dynamic question selection with the RPB score for target `row`."""
    b = table.popularity / table.popularity.sum()
    dh = delta_by_type(table)
    asked = np.zeros(sc.F, dtype=bool)
    plan = []
    for _ in range(min(MAX_QUESTIONS, sc.F)):
        scores = sc.rpb(b, dh, ALPHA)
        scores[asked] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]) or scores[j] <= 0:
            break
        asked[j] = True
        f = table.features[j]
        value = f.value_of(row)
        plan.append({"feature": f.name, "value": value, "rpb": round(float(scores[j]), 4),
                     "numeric_bin": f.numeric})
        if value is not None:
            b = sc.update(b, j, int(f.codes[row]), BELIEF_EPS)
        if len(plan) >= MIN_QUESTIONS and rank_of(b, row) == 0:
            break
    order = [int(i) for i in np.argsort(-b, kind="stable")]
    ptype = product_type(table.products[row])
    same = [i for i in order if i != row and product_type(table.products[i]) == ptype]
    others = [i for i in order if i != row and i not in same]
    candidates = [row] + (same + others)[:2]
    return {"plan": plan, "target_rank": rank_of(b, row), "candidates": candidates,
            "remaining_effective": float(1.0 / (b ** 2).sum())}


_TYPE_STOP = re.compile(r"(\s+مدل\b|\s+طرح\b|\s+کد\b|\s+سری\b|\s+[A-Za-z0-9])")


def product_type(p: Dict[str, Any]) -> str:
    """Generic product type from the title, e.g. 'جامدادی' from 'جامدادی کلیپس مدل 1015'.
    Takes the title up to 'مدل'/'طرح'/'کد' or the first Latin/digit token, without the brand."""
    title = p["title"]
    m = _TYPE_STOP.search(title)
    head = title[: m.start()] if m else title
    brand = (p.get("brand") or "").strip()
    if brand and brand in head:
        head = head[: head.index(brand)]
    if "کتاب" in p["category"]:
        return "کتاب"  # a book's title words would reveal the target
    words = head.split()
    return " ".join(words[:3]) if words else p["category"]


def short_name(feature: str) -> str:
    return feature.split(" / ", 1)[-1]


def fmt_price(p) -> str:
    return f"{p:,.0f} ریال" if p else "نامشخص"


def product_block(p: Dict[str, Any], aug: Dict[str, Any], plan_features: List[str]) -> str:
    lines = [f"عنوان: {p['title']}", f"برند: {p.get('brand') or 'نامشخص'}", f"دسته‌بندی: {p['category']}",
             f"قیمت: {fmt_price(p.get('price'))}"]
    if p.get("rating_count"):
        lines.append(f"امتیاز کاربران: {p['rating_average']:.0f} از ۱۰۰ ({p['rating_count']} نظر)")
    specs = p.get("specs") or {}
    shown = [k for k in plan_features if k in specs] + [k for k in specs if k not in plan_features]
    spec_lines = []
    for k in shown[:12]:
        v = "، ".join(specs[k])
        spec_lines.append(f"  - {short_name(k)}: {v[:160]}")
    if spec_lines:
        lines.append("مشخصات:")
        lines += spec_lines
    if p.get("description"):
        lines.append(f"توضیحات: {p['description'][:400]}")
    s = aug.get("summary", {})
    if "price_position" in s:
        lines.append(f"جایگاه قیمت بین ۱۰ محصول مشابه: {PRICE_POSITION_TEXT[s['price_position']]} "
                     f"({s['price_diff_percent']:+.0f}% نسبت به میانگین مشابه‌ها)")
    lines.append("برچسب‌های مقایسه‌ای: " + "؛ ".join(LABEL_TEXT[x] for x in aug.get("labels", [])))
    return "\n".join(lines)


def build_prompt(target, aug, plan, alternatives) -> str:
    q_lines = []
    for i, q in enumerate(plan["plan"], 1):
        name = "قیمت (بازه)" if q["feature"] == PRICE else short_name(q["feature"])
        ans = q["value"] if q["value"] is not None else "کاربر نمی‌داند یا برایش مهم نیست"
        q_lines.append(f"{i}. ویژگی: «{name}» ← پاسخ کاربر: {ans}")
    alt_blocks = []
    for a, a_aug in alternatives:
        alt_blocks.append(product_block(a, a_aug, [])[:900])
    plan_features = [q["feature"] for q in plan["plan"]]
    return f"""یک گفتگوی خرید «نیازمحور» به زبان فارسی بین «user» (مشتری) و «assistant» (دستیار فروشگاه) بنویسید.

قواعد:
1. کاربر گفتگو را با بیان نیاز کلی خود شروع می‌کند: نوع کلی محصول («{product_type(target)}» در دسته‌ی «{target['category']}») را می‌گوید، اما برند، مدل یا عنوان دقیق محصول را نمی‌داند و نمی‌گوید (مگر برند در برنامه‌ی پرسش آمده باشد).
2. دستیار دقیقاً به ترتیب «برنامه‌ی پرسش» زیر، در هر نوبت فقط درباره‌ی یک ویژگی سؤال می‌پرسد و در صورت نیاز ویژگی‌های فنی را کوتاه توضیح می‌دهد.
3. پاسخ‌های کاربر باید دقیقاً با برنامه‌ی پرسش سازگار باشد؛ اگر نوشته شده «نمی‌داند یا برایش مهم نیست»، کاربر همین را به زبان طبیعی می‌گوید. کاربر اطلاعات دیگری از خودش اضافه نمی‌کند.
4. پس از پرسش‌ها، دستیار ۲ تا ۳ محصول (محصول هدف و گزینه‌های جایگزین زیر) را با قیمت دقیق پیشنهاد و با استفاده از برچسب‌های مقایسه‌ای مقایسه می‌کند.
5. کاربر یک سؤال مقایسه‌ای یا درباره‌ی قیمت/کیفیت می‌پرسد و دستیار فقط بر اساس داده‌های داده‌شده پاسخ می‌دهد.
6. در پایان کاربر «محصول هدف» را انتخاب می‌کند.
7. هیچ مشخصه، قیمت یا ادعایی خارج از اطلاعات داده‌شده ننویسید.
8. گفتگو بین ۸ تا ۱۶ پیام، با لحن دوستانه و محاوره‌ای مؤدبانه باشد.
9. اگر پاسخ کاربر یک بازه‌ی عددی است (مثلاً قیمت یا وزن)، کاربر آن را تقریبی و طبیعی بیان می‌کند (مثلاً با اعداد گرد یا به تومان)، نه با اعداد دقیق مرز بازه.
10. برای هر پیامِ دستیار که درباره‌ی یکی از ویژگی‌های برنامه سؤال می‌کند، نام همان ویژگی را در فیلد feature بنویسید؛ برای بقیه‌ی پیام‌ها null.

برنامه‌ی پرسش (انتخاب‌شده با امتیاز RPB):
{chr(10).join(q_lines) if q_lines else "(بدون پرسش؛ دستیار مستقیماً پیشنهاد می‌دهد)"}

محصول هدف:
{product_block(target, aug, plan_features)}

گزینه‌های جایگزین:
{chr(10).join(f'--- گزینه {i + 1} ---{chr(10)}{b}' for i, b in enumerate(alt_blocks)) or '(ندارد)'}

خروجی را فقط به صورت JSON با این ساختار برگردانید:
{{"dialogue": [{{"role": "user", "text": "...", "feature": null}}, {{"role": "assistant", "text": "...", "feature": "نام ویژگی یا null"}}], "summary": "خلاصه‌ی یک‌جمله‌ای"}}"""


ROLE_MAP = {"user": "user", "customer": "user", "کاربر": "user", "مشتری": "user",
            "assistant": "assistant", "seller": "assistant", "دستیار": "assistant", "فروشنده": "assistant"}


def coerce(obj):
    """Accept small format variations: a top-level list, other key names, Persian role names."""
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict) and \
            any(k in obj[0] for k in ("dialogue", "conversation", "messages", "turns")):
        obj = obj[0]  # the object wrapped in a one-element list
    if isinstance(obj, list):
        obj = {"dialogue": obj}
    if not isinstance(obj, dict):
        return None
    turns = next((obj[k] for k in ("dialogue", "conversation", "messages", "turns") if isinstance(obj.get(k), list)), None)
    if turns is None:
        return None
    clean = []
    for t in turns:
        if not isinstance(t, dict):
            return None
        role = ROLE_MAP.get(str(t.get("role") or t.get("speaker") or "").strip().lower())
        text = str(t.get("text") or t.get("content") or t.get("message") or "").strip()
        if role is None or not text:
            return None
        feat = t.get("feature")
        clean.append({"role": role, "text": text, "feature": feat if feat not in ("", "null", "None") else None})
    return {"dialogue": clean, "summary": obj.get("summary", "")}


def validate(obj) -> bool:
    return bool(obj) and len(obj["dialogue"]) >= 6 and obj["dialogue"][0]["role"] == "user"


def select_targets(clusters: Dict[int, List[Dict[str, Any]]], tables, n_total: int, seed: int) -> List[tuple]:
    """Stratified sample: each cluster gets a quota proportional to its size (at least one),
    interleaved round-robin so that a partial run stays balanced across clusters."""
    rng = np.random.default_rng(seed)
    total = sum(len(v) for v in clusters.values())
    queues = {}
    for c, ps in clusters.items():
        if not tables[c].features:
            continue
        quota = max(1, int(round(n_total * len(ps) / total)))
        rows = rng.permutation(len(ps))[:quota]
        queues[c] = [(c, int(r)) for r in rows]
    order = []
    while any(queues.values()):
        for c in sorted(queues):
            if queues[c]:
                order.append(queues[c].pop(0))
    return order[:n_total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--interval", type=float, default=6.5, help="seconds between requests (free tier ~10/min)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plans-only", action="store_true", help="write question plans without calling the LLM")
    ap.add_argument("--recover-failed", action="store_true",
                    help="reuse valid raw outputs saved in results/failed before calling the API")
    args = ap.parse_args()

    products = load_catalog()
    clusters = by_cluster(products)
    tables = {c: build_cluster_table(c, ps) for c, ps in clusters.items()}
    targets = select_targets(clusters, tables, args.n, args.seed)
    print(f"{len(targets)} targets from {len({c for c, _ in targets})} clusters")
    emb = load_embeddings()
    aug_cache: Dict[int, Dict[int, Any]] = {}
    scorers: Dict[int, ClusterScorer] = {}

    def context(c: int, row: int):
        if c not in scorers:
            scorers[c] = ClusterScorer(tables[c])
            aug_cache[c] = augment_cluster(clusters[c], emb)
        table, sc = tables[c], scorers[c]
        plan = plan_questions(table, sc, row)
        target = table.products[row]
        alts = [(table.products[i], aug_cache[c][table.products[i]["id"]]) for i in plan["candidates"] if i != row][:2]
        return table, target, plan, alts

    if args.dry_run:
        c, row = targets[0]
        _, target, plan, alts = context(c, row)
        print(build_prompt(target, aug_cache[c][target["id"]], plan, alts))
        print(json.dumps({"plan": plan}, ensure_ascii=False, indent=1))
        return

    out = OUT.with_name("question_plans.jsonl") if args.plans_only else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            done.add(json.loads(line)["product_id"])
    print(f"{len(done)} already done; resuming")

    client = None
    if not args.plans_only:
        from cpsdata.llm import GeminiClient, QuotaExhausted, parse_json
        client = GeminiClient(models=args.models, min_interval=args.interval)

    written = 0
    with out.open("a", encoding="utf-8") as fh:
        for c, row in targets:
            target = tables[c].products[row]
            if target["id"] in done:
                continue
            table, target, plan, alts = context(c, row)
            rec = {
                "id": f"pcps-{target['id']}",
                "product_id": target["id"],
                "digikala_id": target["digikala_id"],
                "cluster": c,
                "category": target["category"],
                "title": target["title"],
                "price": target.get("price"),
                "question_plan": plan["plan"],
                "target_rank_after_plan": plan["target_rank"],
                "effective_candidates_after_plan": round(plan["remaining_effective"], 3),
                "candidate_product_ids": [table.products[i]["id"] for i in plan["candidates"]],
                "augmentation_labels": aug_cache[c][target["id"]]["labels"],
                "product_type": product_type(target),
            }
            if client is not None:
                prompt = build_prompt(target, aug_cache[c][target["id"]], plan, alts)
                obj = None
                if args.recover_failed:
                    for f in sorted((ROOT / "results" / "failed").glob(f"{target['id']}_*.txt")):
                        cand = coerce(parse_json(f.read_text(encoding="utf-8")))
                        if validate(cand):
                            obj = cand
                            rec.update(model=f.stem.split("_", 1)[1], prompt_version=PROMPT_VERSION,
                                       created_at=datetime.fromtimestamp(f.stat().st_mtime, timezone.utc)
                                       .isoformat(timespec="seconds"),
                                       dialogue=obj["dialogue"], summary=obj.get("summary", ""))
                            break
                try:
                    for _ in range(0 if obj is not None else 2):
                        raw = client.chat(SYSTEM, prompt)
                        obj = coerce(parse_json(raw))
                        if validate(obj):
                            break
                        fail_dir = ROOT / "results" / "failed"
                        fail_dir.mkdir(parents=True, exist_ok=True)
                        (fail_dir / f"{target['id']}_{client.model}.txt").write_text(raw, encoding="utf-8")
                        obj = None
                except QuotaExhausted as e:
                    print(f"stopping: {e}")
                    break
                if obj is None:
                    print(f"  skipped {target['id']} (invalid output)")
                    continue
                if "dialogue" not in rec:
                    rec.update(model=client.model, prompt_version=PROMPT_VERSION,
                               created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               dialogue=obj["dialogue"], summary=obj.get("summary", ""))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if written % 10 == 0 or client is None and written % 200 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {len(done) + written} / {len(targets)}")
    print(f"wrote {written} records to {out}")


if __name__ == "__main__":
    main()
