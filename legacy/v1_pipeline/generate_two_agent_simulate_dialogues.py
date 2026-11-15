#!/usr/bin/env python3
"""
Two-Agent Dialogue Generator (Preference Agent + Seller Agent)
=============================================================

This script generates dialogues where:
- Agent 1 (User Preference Agent) has information about what the user wants
- Agent 2 (Seller Agent) has information about the actual product to recommend
  (including specifications, highest-metric features and comparison labels)

The LLM is instructed to simulate a natural conversation between a real user
and an assistant, using those two internal perspectives, and return a single
user/assistant dialogue.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from generate_dialogues_with_highest_metric_features import (
    HighestMetricDialogueGenerator,
)

# Import shared product IDs configuration
try:
    from dialogue_generation_config import TARGET_PRODUCT_IDS
except ImportError:
    # Fallback if config file doesn't exist
    TARGET_PRODUCT_IDS = [4477, 11767, 4863, 10101, 7643, 1454, 6669, 8311, 3408, 5890]


class TwoAgentDialogueGenerator:
    def __init__(
        self,
        products_file: str,
        sampling_probabilities_file: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.products_file = products_file
        self.sampling_probabilities_file = sampling_probabilities_file

        # Reuse highest-metric generator for loading data and feature selection/helpers
        self.hm_generator = HighestMetricDialogueGenerator(
            products_file=products_file,
            sampling_probabilities_file=sampling_probabilities_file,
            api_key=api_key,
            base_url=base_url,
        )
        self.products_data: List[Dict[str, Any]] = []

        # LLM client (reuse same config as hm_generator)
        self.api_key = self.hm_generator.api_key
        self.base_url = self.hm_generator.base_url
        self.client = self.hm_generator.client
        self.model = self.hm_generator.model
        self.max_retries = self.hm_generator.max_retries
        self.delay_between_requests = self.hm_generator.delay_between_requests

        self.generated_dialogues: List[Dict[str, Any]] = []

    def load_data(self) -> List[Dict[str, Any]]:
        """Load products and probabilities through the highest-metric generator."""
        products_data, _ = self.hm_generator.load_data()
        self.products_data = products_data
        return products_data

    def get_products_by_ids(self, product_ids: List[int]) -> List[Dict[str, Any]]:
        """Proxy to underlying highest-metric generator."""
        return self.hm_generator.get_products_by_ids(product_ids)

    def build_user_agent_profile(
        self,
        product: Dict[str, Any],
        selected_features: List[Tuple[str, str, float]],
    ) -> str:
        """
        Build an abstract 'what the customer wants' profile based on the target product
        and its key features/comparison labels, without mentioning the exact model name.
        """
        lines: List[str] = []
        category = product.get("category_name", "نامشخص")
        brand = product.get("brand", "")

        lines.append(f"کاربر به دنبال خرید محصولی در دسته‌بندی «{category}» است.")
        if brand:
            lines.append("برند برای او مهم است اما الزاما همان برند فعلی نباید در ابتدا ذکر شود.")

        # Use comparison labels to describe what the user cares about
        labels = product.get("comparison_labels", [])
        cares_about_price = False
        cares_about_quality = False
        cares_about_popularity = False

        for item in labels:
            if not isinstance(item, dict):
                continue
            key = item.get("label")
            if key in ("good_price", "premium_price", "high_value_pick"):
                cares_about_price = True
            if key in ("high_rated", "needs_rating_improvement", "high_value_pick"):
                cares_about_quality = True
            if key in ("popular_choice", "niche_pick"):
                cares_about_popularity = True

        prefs: List[str] = []
        if cares_about_price:
            prefs.append("می‌خواهد تناسب خوبی بین قیمت و امکانات داشته باشد.")
        if cares_about_quality:
            prefs.append("حساسیت زیادی روی کیفیت و امتیاز کاربران دارد.")
        if cares_about_popularity:
            prefs.append("برای او مهم است که محصول، انتخاب پرطرفدار یا مطمئنی باشد.")

        if prefs:
            lines.append("ترجیحات کلی کاربر:")
            for p in prefs:
                lines.append(f"- {p}")

        # Include key technical/functional needs derived from selected features
        if selected_features:
            lines.append("\nنیازهای فنی/عملکردی که کاربر در ذهن دارد (بدون شناخت مدل دقیق):")
            for feat_name, feat_value, _ in selected_features:
                display_name = feat_name.split("_", 1)[-1] if "_" in feat_name else feat_name
                value_text = feat_value
                if len(value_text) > 120:
                    value_text = value_text[:120] + "..."
                lines.append(f"- به دنبال محصولی است که در زمینه «{display_name}» چیزی شبیه به: «{value_text}» داشته باشد.")

        lines.append(
            "\nکاربر نام دقیق مدل و جزئیات فنی کامل را نمی‌داند؛ فقط نیازها و اولویت‌های خود را بیان می‌کند."
        )

        return "\n".join(lines)

    def build_seller_agent_knowledge(
        self,
        product: Dict[str, Any],
        selected_features: List[Tuple[str, str, float]],
    ) -> str:
        """
        Build the seller-side knowledge about the exact product, using the
        enriched formatter from HighestMetricDialogueGenerator.
        """
        details = self.hm_generator.format_product_details_with_features(
            product, selected_features
        )

        lines: List[str] = []
        lines.append("این اطلاعاتی است که فروشنده دقیقاً درباره محصول پیشنهادی می‌داند:")
        lines.append(details)
        lines.append(
            "\nفروشنده می‌داند این محصول دقیقاً چه جایگاهی نسبت به محصولات مشابه دارد "
            "و می‌تواند در صورت نیاز آن را با گزینه‌های دیگر مقایسه کند."
        )

        return "\n".join(lines)

    def create_two_agent_prompt(
        self,
        product: Dict[str, Any],
        selected_features: List[Tuple[str, str, float]],
    ) -> str:
        """Create a prompt that describes the two internal agents and desired final dialogue."""
        user_profile = self.build_user_agent_profile(product, selected_features)
        seller_knowledge = self.build_seller_agent_knowledge(product, selected_features)

        feature_names = [
            feat[0].split("_", 1)[-1] if "_" in feat[0] else feat[0]
            for feat in selected_features
        ]

        prompt = f"""شما باید یک گفتگوی طبیعی بین مشتری (کاربر) و دستیار فروش (فروشنده) ایجاد کنید.
برای ساخت این گفتگو، دو عامل داخلی در ذهن خود در نظر بگیرید:

1) «عامل ترجیحات کاربر» که فقط اطلاعات زیر را می‌داند و بر اساس آن رفتار می‌کند:

{user_profile}

2) «عامل فروشنده» که اطلاعات کامل زیر را درباره محصول پیشنهادی و جایگاه آن در بازار می‌داند:

{seller_knowledge}

نحوه تولید گفتگو:
- ابتدا در ذهن خود، این دو عامل را شبیه‌سازی کنید که با هم تعامل دارند؛
  عامل ترجیحات کاربر توضیح می‌دهد چه می‌خواهد و عامل فروشنده تصمیم می‌گیرد
  چه سوال‌هایی بپرسد و در نهایت کدام محصول را پیشنهاد دهد.
- خروجی نهایی که به من نشان می‌دهید، فقط باید یک گفتگوی واقعی بین «کاربر» و «دستیار» باشد
  (بدون اشاره به عامل‌های داخلی).

الزامات گفتگو:
1. گفتگو باید طبیعی، روان و واقعی باشد.
2. کاربر در ابتدا به صورت کلی می‌گوید چه نوع محصولی می‌خواهد و چه دغدغه‌هایی دارد.
3. دستیار سوالات حرفه‌ای می‌پرسد تا نیاز کاربر را دقیق‌تر بفهمد (بر اساس ویژگی‌ها و برچسب‌های مقایسه‌ای).
4. گفتگو باید به طور خاص روی این ویژگی‌ها و نیازها بحث کند: {", ".join(feature_names)}.
5. دستیار باید از اطلاعات مقایسه‌ای (قیمت، ارزش خرید، محبوبیت و ...) برای توجیه پیشنهاد خود استفاده کند.
6. اگر محصول قیمت پریمیوم یا اقتصادی دارد، دستیار باید این موضوع را توضیح دهد و توجیه کند.
7. در طول گفتگو، کاربر می‌تواند درباره گزینه‌های ارزان‌تر/گران‌تر یا با کیفیت متفاوت سوال کند
   و دستیار با تکیه بر جایگاه محصول، پاسخ دهد.
8. تعداد دورهای گفتگو باید بین 12 تا 22 نوبت (پیام) باشد.
9. گفتگو باید کاملاً به زبان فارسی و با لحن دوستانه و حرفه‌ای باشد.
10. در پایان، دستیار باید این محصول را به عنوان گزینه اصلی پیشنهاد دهد و خلاصه‌ای از مزایا ارائه کند.

خروجی را فقط به صورت JSON با ساختار زیر برگردانید:
{{
  "dialogue": [
    {{"role": "user", "text": "متن کاربر"}},
    {{"role": "assistant", "text": "متن دستیار"}},
    ...
  ],
  "features_discussed": ["نام ویژگی 1", "نام ویژگی 2", ...],
  "dialogue_summary": "خلاصه کوتاه از گفتگو (چرا این محصول مناسب کاربر است)"
}}

فقط JSON را برگردانید، بدون هیچ توضیح اضافی."""
        return prompt

    def call_llm(self, prompt: str) -> str:
        """Call LLM with retries."""
        system_prompt = (
            "شما متخصص ساخت گفتگوهای طبیعی بین مشتری و فروشنده در فروشگاه آنلاین هستید. "
            "بر اساس اطلاعات داده شده، گفتگو را طوری طراحی می‌کنید که هم نیازهای کاربر روشن شود "
            "و هم محصول مناسب با استدلال حرفه‌ای پیشنهاد شود."
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2500,
                    temperature=0.8,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                else:
                    raise

    @staticmethod
    def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response (strip code fences if present)."""
        text = response.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Response head: {text[:400]}")
            return None

    def generate_dialogue_for_product(self, product: Dict[str, Any], num_features: int = 5) -> Optional[Dict[str, Any]]:
        """Generate a two-agent style dialogue for a single product."""
        product_id = product.get("id")
        title = product.get("title", "Unknown")
        cluster_id = self.hm_generator.get_product_cluster(product)

        print(f"\nGenerating two-agent dialogue for product: {title}")
        print(f"  Product ID: {product_id} | Cluster: {cluster_id}")

        # Extract and select highest-metric features
        product_features = self.hm_generator.extract_product_features(product)
        if not product_features:
            print("  Warning: No product features found; skipping.")
            return None

        selected_features = self.hm_generator.select_highest_metric_features(
            cluster_id=str(cluster_id) if cluster_id is not None else "",
            product_features=product_features,
            num_features=num_features,
        )

        if not selected_features:
            print("  Warning: No features selected; skipping.")
            return None

        print(f"  Selected {len(selected_features)} features for two-agent dialogue.")

        prompt = self.create_two_agent_prompt(product, selected_features)

        try:
            response = self.call_llm(prompt)
            parsed = self.parse_llm_response(response)
            if not parsed:
                return None

            dialogue_obj: Dict[str, Any] = {
                "product_id": product_id,
                "product_title": title,
                "brand": product.get("brand"),
                "category": product.get("category_name"),
                "price": product.get("price") or product.get("selling_price"),
                "cluster_id": cluster_id,
                "dialogue_rounds": len(parsed.get("dialogue", [])),
                "selected_features": [
                    {
                        "feature_name": feat[0],
                        "feature_value": feat[1],
                        "custom_metric": feat[2],
                    }
                    for feat in selected_features
                ],
                "features_discussed": parsed.get("features_discussed", []),
                "dialogue_summary": parsed.get("dialogue_summary", ""),
                "dialogue": parsed.get("dialogue", []),
                "llm_response_raw": response,
                "type": 6,  # Type 6 for two-agent simulate dialogues
                "kind": 6,  # Kind 6 for two-agent highest-metric dialogues (simulate version)
                "product_details": {
                    "title": title,
                    "brand": product.get("brand"),
                    "category": product.get("category_name"),
                    "price": product.get("price") or product.get("selling_price"),
                },
            }
            return dialogue_obj
        except Exception as e:
            print(f"Error generating two-agent dialogue: {e}")
            return None

    def generate_batch(
        self,
        product_ids: List[int],
        num_features: int = 5,
        output_dir: str = "llm_generated_dialogues_two_agent_highest_metric_kind6",
    ) -> str:
        """Generate two-agent dialogues for a fixed list of product IDs."""
        os.makedirs(output_dir, exist_ok=True)

        products = self.get_products_by_ids(product_ids)
        print(f"\nStarting two-agent generation for {len(products)} products...")

        for idx, product in enumerate(products, start=1):
            print(f"\n[{idx}/{len(products)}] Processing product ID {product.get('id')}")
            dialogue = self.generate_dialogue_for_product(product, num_features=num_features)
            if dialogue:
                self.generated_dialogues.append(dialogue)
                print(f"✅ Two-agent dialogue generated ({dialogue['dialogue_rounds']} rounds)")
            else:
                print("❌ Failed to generate two-agent dialogue")

            if idx < len(products):
                time.sleep(self.delay_between_requests)

        output_data = {
            "metadata": {
                "total_generated": len(self.generated_dialogues),
                "generation_date": datetime.now().isoformat(),
                "model": self.model,
                "source_file": self.products_file,
                "num_features_per_product": num_features,
                "selection_method": "highest_custom_metric_two_agent_simulate",
                "type": 6,
                "kind": 6,
            },
            "generated_dialogues": self.generated_dialogues,
        }

        output_path = os.path.join(output_dir, "llm_generated_dialogues_two_agent_highest_metric_kind6.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved {len(self.generated_dialogues)} two-agent dialogues (kind 6) to {output_path}")

        # Save individuals with type6_dialogue_ prefix for evaluation script compatibility
        for idx, dialogue in enumerate(self.generated_dialogues, start=1):
            individual_path = os.path.join(
                output_dir, f"type6_dialogue_{idx}_{dialogue['product_id']}.json"
            )
            with open(individual_path, "w", encoding="utf-8") as f:
                json.dump(dialogue, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved individual two-agent dialogue files to {output_dir}/")

        return output_path


def main() -> None:
    print("Two-Agent Dialogue Generator (Highest Metric Features)")
    print("=" * 70)

    products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    sampling_probabilities_file = "cluster_sampling_probability_results_custome/cluster_sampling_probabilities.json"

    # Use shared product IDs from config (same as other dialogue generators)
    product_ids = TARGET_PRODUCT_IDS

    generator = TwoAgentDialogueGenerator(
        products_file=products_file,
        sampling_probabilities_file=sampling_probabilities_file,
    )

    generator.load_data()

    output_path = generator.generate_batch(
        product_ids=product_ids,
        num_features=5,
        output_dir="llm_generated_dialogues_two_agent_highest_metric_kind6",
    )

    print(f"\n🎉 Two-agent generation completed. Results saved to: {output_path}")


if __name__ == "__main__":
    main()


