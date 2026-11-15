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
        # Inherit API config from hm_generator (already updated to OpenAI)
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

    def call_llm_for_agent(self, system_prompt: str, user_content: str) -> str:
        """Generic LLM call for a specific agent with its own system prompt."""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=600,
                    temperature=0.8,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                else:
                    raise

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
        """(Deprecated) kept for backward compatibility, not used in step-wise mode."""
        return ""

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

    @staticmethod
    def format_history_for_agent(history: List[Dict[str, str]]) -> str:
        """Render current dialogue history as simple labeled lines."""
        lines: List[str] = []
        for turn in history:
            role = turn.get("role")
            text = turn.get("text", "")
            if role == "user":
                lines.append(f"کاربر: {text}")
            elif role == "assistant":
                lines.append(f"دستیار: {text}")
        return "\n".join(lines)

    def simulate_two_agent_conversation(
        self,
        product: Dict[str, Any],
        selected_features: List[Tuple[str, str, float]],
        num_turn_pairs: int = 8,
    ) -> List[Dict[str, str]]:
        """
        Simulate a conversation turn-by-turn using two explicit agents.

        Each turn:
        - User agent LLM is called with history to produce next user message.
        - Seller agent LLM is called with updated history to produce next assistant message.
        """
        history: List[Dict[str, str]] = []

        user_profile = self.build_user_agent_profile(product, selected_features)
        seller_knowledge = self.build_seller_agent_knowledge(product, selected_features)

        # Pre-build system prompts for agents
        user_system = (
            "شما «عامل ترجیحات کاربر» هستید.\n"
            "فقط از دید یک مشتری واقعی صحبت می‌کنید و به اطلاعات زیر درباره نیازها و ترجیحات او دسترسی دارید.\n"
            "نام مدل دقیق یا مشخصات فنی کامل محصول پیشنهادی را نمی‌دانید.\n\n"
            f"{user_profile}\n\n"
            "در هر نوبت، فقط یک پیام کاربر را تولید کنید."
        )

        seller_system = (
            "شما «عامل فروشنده» هستید.\n"
            "شما همه اطلاعات فنی و مقایسه‌ای محصول پیشنهادی زیر را می‌دانید و باید مانند یک فروشنده حرفه‌ای پاسخ دهید.\n\n"
            f"{seller_knowledge}\n\n"
            "در هر نوبت، فقط پیام دستیار (فروشنده) را تولید کنید."
        )

        for pair_idx in range(num_turn_pairs):
            # ---- User turn ----
            history_text = self.format_history_for_agent(history)
            user_prompt = (
                "تاریخچه مکالمه تا این لحظه:\n"
                f"{history_text if history_text else '(خالی)'}\n\n"
                "حالا فقط «پیام بعدی کاربر» را بنویس.\n"
                "- از دید یک مشتری واقعی صحبت کن.\n"
                "- می‌توانی سوال بپرسی، نگرانی‌ها را مطرح کنی یا اولویت‌ها را شفاف‌تر کنی.\n"
                "- جواب را کوتاه و طبیعی بنویس (۱ تا ۳ جمله).\n"
                "- از نوشتن پاسخ دستیار یا تحلیل متا خودداری کن."
            )
            user_text = self.call_llm_for_agent(user_system, user_prompt)
            history.append({"role": "user", "text": user_text})

            # ---- Seller turn ----
            history_text = self.format_history_for_agent(history)
            seller_prompt = (
                "تاریخچه مکالمه تا این لحظه بین کاربر و دستیار:\n"
                f"{history_text}\n\n"
                "حالا فقط «پاسخ بعدی دستیار (فروشنده)» را بنویس.\n"
                "- به صورت حرفه‌ای و صمیمی پاسخ بده.\n"
                "- از اطلاعات مقایسه‌ای (قیمت، ارزش خرید، امتیاز کاربران و ...) استفاده کن.\n"
                "- به تدریج به سمت یک پیشنهاد مشخص حرکت کن.\n"
                "- در نوبت‌های پایانی (به خصوص در نوبت‌های آخر)، محصول را جمع‌بندی و معرفی نهایی را انجام بده.\n"
                "- جواب را کوتاه و طبیعی بنویس (۱ تا ۴ جمله).\n"
                "- از نوشتن متن کاربر یا توضیحات متا خودداری کن."
            )
            seller_text = self.call_llm_for_agent(seller_system, seller_prompt)
            history.append({"role": "assistant", "text": seller_text})

        return history

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

        # Simulate explicit two-agent conversation with step-wise LLM calls
        try:
            history = self.simulate_two_agent_conversation(
                product, selected_features, num_turn_pairs=8
            )

            dialogue_obj: Dict[str, Any] = {
                "product_id": product_id,
                "product_title": title,
                "brand": product.get("brand"),
                "category": product.get("category_name"),
                "price": product.get("price") or product.get("selling_price"),
                "cluster_id": cluster_id,
                "type": 5,  # Two-agent dialogue with highest metric features
                "dialogue_rounds": len(history),
                "selected_features": [
                    {
                        "feature_name": feat[0],
                        "feature_value": feat[1],
                        "custom_metric": feat[2],
                    }
                    for feat in selected_features
                ],
                # In step-wise mode we don't ask LLM for this list; use selected feature names.
                "features_discussed": [
                    feat[0].split("_", 1)[-1] if "_" in feat[0] else feat[0]
                    for feat in selected_features
                ],
                "dialogue_summary": f"گفتگوی چندمرحله‌ای بین کاربر و دستیار درباره «{title}» "
                "بر اساس نیازهای کاربر و مقایسه با محصولات مشابه.",
                "dialogue": history,
                "llm_response_raw": None,
                "kind": 5,  # new kind for two-agent highest-metric dialogues
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
        output_dir: str = "llm_generated_dialogues_two_agent_highest_metric",
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
                "selection_method": "highest_custom_metric_two_agent",
                "kind": 5,
            },
            "generated_dialogues": self.generated_dialogues,
        }

        output_path = os.path.join(output_dir, "llm_generated_dialogues_two_agent_highest_metric.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Saved {len(self.generated_dialogues)} two-agent dialogues to {output_path}")

        # Save individuals with type5_dialogue_ prefix for evaluation script
        for idx, dialogue in enumerate(self.generated_dialogues, start=1):
            individual_path = os.path.join(
                output_dir, f"type5_dialogue_{idx}_{dialogue['product_id']}.json"
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

    # Same 10 products used in other dialogue generators
    product_ids = [4477, 11767, 4863, 10101, 7643, 1454, 6669, 8311, 3408, 5890]

    generator = TwoAgentDialogueGenerator(
        products_file=products_file,
        sampling_probabilities_file=sampling_probabilities_file,
    )

    generator.load_data()

    output_path = generator.generate_batch(
        product_ids=product_ids,
        num_features=5,
        output_dir="llm_generated_dialogues_two_agent_highest_metric",
    )

    print(f"\n🎉 Two-agent generation completed. Results saved to: {output_path}")


if __name__ == "__main__":
    main()


