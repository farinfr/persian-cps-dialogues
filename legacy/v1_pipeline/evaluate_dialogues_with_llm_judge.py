#!/usr/bin/env python3
"""
Evaluate Dialogues with LLM Judge
=================================

We currently have multiple *kinds* of generated dialogues for the same 10 products.
Goal of this script:

  - For each product_id, collect all available dialogues across different kinds
    (1..6) from the existing JSON files.
  - In a single LLM call per product, send:
      * A short description of the product
      * All candidate dialogues (each tagged with its kind)
    and ask the LLM to *judge and score* each dialogue across several metrics:
      - fluency / naturalness
      - factual accuracy relative to product info
      - realism / believability
      - coverage of key features
      - helpfulness for product search/recommendation
      - overall quality
  - Store the results in a JSON file per product and an aggregate JSON.

We use the same LLM provider (Avala via OpenAI client) as the other scripts.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


# --------------------------- Configuration ------------------------------------

ENHANCED_PRODUCTS_FILE = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"

KIND_SOURCES = [
    # kind 1: dual method - highest metric rewritten
    {"kind": 1, "label": "dual_highest_metric", "dir": "dual_method_dialogues", "file_pattern_prefix": "highest_metric_rewritten_", "expected_type": 1},
    # kind 2: dual method - random rewritten
    {"kind": 2, "label": "dual_random", "dir": "dual_method_dialogues", "file_pattern_prefix": "random_rewritten_", "expected_type": 2},
    # kind 3: LLM generated from scratch (random features)
    {"kind": 3, "label": "llm_generated_random", "dir": "llm_generated_dialogues", "file_pattern_prefix": "dialogue_", "expected_type": 3},
    # kind 4: LLM generated from scratch (highest metric features)
    {"kind": 4, "label": "llm_generated_highest_metric", "dir": "llm_generated_dialogues_highest_metric", "file_pattern_prefix": "dialogue_", "expected_type": 4},
    # kind 5: two-agent highest metric (original dir)
    {"kind": 5, "label": "two_agent_highest_metric", "dir": "llm_generated_dialogues_two_agent_highest_metric", "file_pattern_prefix": "type5_dialogue_", "expected_type": 5},
    # kind 6: two-agent highest metric (simulate version)
    {"kind": 6, "label": "two_agent_highest_metric_simulate", "dir": "llm_generated_dialogues_two_agent_highest_metric_kind6", "file_pattern_prefix": "type6_dialogue_", "expected_type": 6},
]

TARGET_PRODUCT_IDS = [4477, 11767, 4863, 10101, 7643, 1454, 6669, 8311, 3408, 5890]

OUTPUT_DIR = "dialogue_evaluations_llm_judge"

# LLM config (reuse Avala endpoint)
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = "meta.llama3-1-405b-instruct-v1:0"  # Using Meta LLaMA 3.1 405B Instruct on Avala AI
MAX_RETRIES = 3
DELAY_BETWEEN_REQUESTS = 5.0


# --------------------------- Data Structures ----------------------------------

@dataclass
class DialogueVariant:
    kind: int
    kind_label: str
    product_id: int
    source_file: str
    dialogue: List[Dict[str, str]]
    dialogue_summary: str


class LLMEvaluator:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = MODEL_NAME
        self.max_retries = MAX_RETRIES
        self.delay_between_requests = DELAY_BETWEEN_REQUESTS

    def call_judge(self, prompt: str) -> str:
        """Call LLM judge with retries."""
        system_prompt = (
            "شما یک داور متخصص برای ارزیابی کیفیت گفتگوهای بین کاربر و دستیار فروش "
            "در فروشگاه آنلاین هستید. شما گفتگوها را از نظر روانی، دقت، واقع‌گرایی و "
            "میزان کمک به انتخاب محصول بررسی می‌کنید و خروجی را فقط به صورت JSON برمی‌گردانید."
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM judge error (attempt {attempt + 1}/{self.max_retries}): {e}")
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))
        if last_error:
            raise last_error
        raise RuntimeError("Unknown LLM judge error")

    @staticmethod
    def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM, stripping markdown fences if present."""
        raw = text.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Error parsing judge JSON: {e}")
            print(f"Response (head): {raw[:400]}")
            return None


# --------------------------- Helpers ------------------------------------------

def load_enhanced_products(path: str) -> Dict[int, Dict[str, Any]]:
    """Load enhanced product dataset and index by product id."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    products = data.get("products", data)
    by_id: Dict[int, Dict[str, Any]] = {}
    for p in products:
        pid = p.get("id")
        if isinstance(pid, int):
            by_id[pid] = p
    return by_id


def format_product_brief(product: Dict[str, Any]) -> str:
    """Short human-readable summary to let LLM judge check factuality."""
    title = product.get("title", "نامشخص")
    brand = product.get("brand", "نامشخص")
    category = product.get("category_name", "نامشخص")
    price = product.get("price") or product.get("selling_price")
    rating = product.get("rating_average")
    rating_count = product.get("rating_count")

    lines = [
        f"عنوان: {title}",
        f"برند: {brand}",
        f"دسته‌بندی: {category}",
    ]
    if price:
        try:
            lines.append(f"قیمت تقریبی: {price:,.0f} ریال")
        except (TypeError, ValueError):
            lines.append(f"قیمت تقریبی: {price} ریال")
    if rating is not None and rating_count is not None:
        lines.append(f"میانگین امتیاز کاربران: {rating:.1f} از 100 بر اساس حدود {rating_count:.0f} نظر")

    labels = product.get("comparison_labels", [])
    if labels:
        label_texts = [lbl.get("label") for lbl in labels if isinstance(lbl, dict)]
        if label_texts:
            lines.append(f"برچسب‌های مقایسه‌ای: {', '.join(label_texts)}")

    return "\n".join(lines)


def collect_dialogue_variants() -> Dict[int, List[DialogueVariant]]:
    """Scan all sources and build mapping: product_id -> list of DialogueVariant."""
    product_to_variants: Dict[int, List[DialogueVariant]] = {}

    for src in KIND_SOURCES:
        kind = src["kind"]
        label = src["label"]
        directory = src["dir"]
        prefix = src["file_pattern_prefix"]
        expected_type = src.get("expected_type")

        if not os.path.isdir(directory):
            continue

        for fname in os.listdir(directory):
            if not fname.endswith(".json"):
                continue
            if not fname.startswith(prefix):
                continue
            # Skip the main aggregate JSON files
            if fname.endswith("_dialogues.json") or fname == "llm_generated_dialogues.json" or fname == "llm_rewritten_dialogues.json":
                continue
            path = os.path.join(directory, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue

            # Filter by type field if expected_type is specified
            # Also check kind field as fallback if type is None
            if expected_type is not None:
                file_type = data.get("type")
                file_kind = data.get("kind")
                # Match if type matches, OR if type is None but kind matches expected_type
                if file_type != expected_type and not (file_type is None and file_kind == expected_type):
                    continue

            pid = data.get("product_id")
            if not isinstance(pid, int):
                continue

            dialogue = data.get("dialogue") or data.get("dialogues") or []
            if not isinstance(dialogue, list):
                continue

            summary = data.get("dialogue_summary", "")

            variant = DialogueVariant(
                kind=kind,
                kind_label=label,
                product_id=pid,
                source_file=path,
                dialogue=dialogue,
                dialogue_summary=summary,
            )
            product_to_variants.setdefault(pid, []).append(variant)

    return product_to_variants


def format_dialogues_for_judge(variants: List[DialogueVariant]) -> str:
    """Render all candidate dialogues for a product in a compact, labeled form."""
    blocks: List[str] = []
    for v in sorted(variants, key=lambda x: x.kind):
        lines: List[str] = []
        lines.append(f"=== گفتگوی نوع {v.kind} ({v.kind_label}) - فایل: {os.path.basename(v.source_file)} ===")
        if v.dialogue_summary:
            lines.append(f"خلاصه (طبق تولید اولیه): {v.dialogue_summary}")
        lines.append("متن گفتگو (کاربر/دستیار):")
        for turn in v.dialogue:
            role = turn.get("role", "")
            text = turn.get("text", "")
            if role == "user":
                lines.append(f"کاربر: {text}")
            elif role == "assistant":
                lines.append(f"دستیار: {text}")
            else:
                lines.append(f"{role}: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_judge_prompt(product_brief: str, variants: List[DialogueVariant]) -> str:
    """Construct the full evaluation prompt for one product."""
    dialogues_text = format_dialogues_for_judge(variants)

    prompt = f"""ما چند گفتگوی مختلف درباره یک محصول واحد داریم که با روش‌های مختلف تولید شده‌اند.
وظیفه شما این است که این گفتگوها را از نظر کیفیت بررسی و مقایسه کنید.

اطلاعات خلاصه محصول:
--------------------
{product_brief}

گفتگوهای کاندید:
----------------
{dialogues_text}

لطفاً برای *هر* گفتگوی نوع N (kind) معیارهای زیر را بین 1 تا 10 نمره‌دهی کنید (10 بهترین است):
- fluency: روانی و طبیعی بودن زبان
- accuracy: میزان تطابق اطلاعات با مشخصات محصول (تا حدی که از توضیحات می‌توان استنباط کرد)
- realism: واقع‌گرایی و قابل‌باور بودن مکالمه
- coverage: میزان پوشش ویژگی‌های مهم و نیازهای کاربر
- helpfulness: میزان کمک به کاربر برای انتخاب محصول مناسب
- overall: ارزیابی کلی کیفیت گفتگو (نه لزوماً میانگین ساده اعداد بالا)

سپس:
1. یک *رتبه‌بندی* از بهترین تا ضعیف‌ترین گفتگو بر اساس معیار overall ارائه کنید.
2. برای هر گفتگو یک توضیح کوتاه (1-2 جمله) بنویسید که چرا این نمرات را داده‌اید.

خروجی را دقیقاً به صورت JSON با ساختار زیر برگردانید:
{{
  "per_dialogue": [
    {{
      "kind": 1,
      "kind_label": "dual_highest_metric",
      "scores": {{
        "fluency": 0-10,
        "accuracy": 0-10,
        "realism": 0-10,
        "coverage": 0-10,
        "helpfulness": 0-10,
        "overall": 0-10
      }},
      "comment": "توضیح کوتاه"
    }},
    ...
  ],
  "ranking": [1, 4, 3, 2, 6, 5]  // فهرست kind ها از بهترین تا ضعیف‌ترین
}}

فقط JSON معتبر را برگردانید، بدون هیچ متن اضافی."""
    return prompt


# --------------------------- Main Evaluation Logic ----------------------------

def main() -> None:
    print("LLM Judge Evaluation for Generated Dialogues")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading enhanced products...")
    products_by_id = load_enhanced_products(ENHANCED_PRODUCTS_FILE)
    print(f"Loaded {len(products_by_id)} products from enhanced dataset.")

    print("Collecting dialogue variants across kinds...")
    product_to_variants = collect_dialogue_variants()
    print(f"Found dialogues for {len(product_to_variants)} distinct products.")

    evaluator = LLMEvaluator()

    aggregate_results: Dict[int, Dict[str, Any]] = {}

    for pid in TARGET_PRODUCT_IDS:
        variants = product_to_variants.get(pid, [])
        if not variants:
            print(f"\n[Product {pid}] No dialogues found across kinds, skipping.")
            continue

        product = products_by_id.get(pid)
        if not product:
            print(f"\n[Product {pid}] Product not found in enhanced dataset, skipping.")
            continue

        print(f"\n[Product {pid}] Evaluating {len(variants)} dialogue variants...")

        product_brief = format_product_brief(product)
        prompt = build_judge_prompt(product_brief, variants)

        try:
            response_text = evaluator.call_judge(prompt)
            parsed = evaluator.parse_json_response(response_text)
            if not parsed:
                print(f"[Product {pid}] Failed to parse judge response; skipping.")
                continue
        except Exception as e:
            print(f"[Product {pid}] LLM judge error: {e}")
            continue

        # Save per-product result
        result = {
            "product_id": pid,
            "product_title": product.get("title"),
            "evaluated_kinds": [v.kind for v in variants],
            "judge_output_raw": response_text,
            "judge_output_parsed": parsed,
        }
        aggregate_results[pid] = result

        out_path = os.path.join(OUTPUT_DIR, f"evaluation_{pid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[Product {pid}] ✅ Saved evaluation to {out_path}")

        # Delay between judge calls to respect rate limits
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Save aggregate file
    aggregate_path = os.path.join(OUTPUT_DIR, "all_evaluations.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_results, f, ensure_ascii=False, indent=2)

    print("\n✅ LLM judge evaluation completed.")
    print(f"Aggregate results saved to {aggregate_path}")


if __name__ == "__main__":
    main()


