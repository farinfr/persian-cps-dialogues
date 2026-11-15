#!/usr/bin/env python3
"""
LLM Dialogue Generator with Highest Custom Metric Features
==========================================================

This script:
1. Selects 10 products
2. For each product, selects features with highest custom_metric values from cluster
3. Sends product details + only those highest custom_metric features to LLM
4. Generates dialogues
"""

import json
import os
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from datetime import datetime

class HighestMetricDialogueGenerator:
    def __init__(self, 
                 products_file: str,
                 sampling_probabilities_file: str,
                 api_key: str = None,
                 base_url: str = None):
        self.products_file = products_file
        self.sampling_probabilities_file = sampling_probabilities_file
        self.products_data = None
        self.sampling_probs = {}
        self.generated_dialogues = []
        
        # Initialize LLM client - Using Avala AI API
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # LLM configuration
        self.model = "gpt-4o"  # Using GPT-4o
        self.max_retries = 3
        self.delay_between_requests = 3.0
        
    def load_data(self):
        """Load products and sampling probabilities"""
        print("Loading products...")
        with open(self.products_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'products' in data:
            self.products_data = data['products']
        else:
            self.products_data = data
        
        print(f"Loaded {len(self.products_data)} products")
        
        print("Loading cluster sampling probabilities...")
        with open(self.sampling_probabilities_file, 'r', encoding='utf-8') as f:
            self.sampling_probs = json.load(f)
        
        print(f"Loaded probabilities for {len(self.sampling_probs)} clusters")
        return self.products_data, self.sampling_probs
    
    def get_products_by_ids(self, product_ids: List[int]) -> List[Dict]:
        """Get products by their IDs"""
        # Create a lookup dictionary for faster access
        products_lookup = {p.get('id'): p for p in self.products_data if p.get('id') is not None}
        
        sampled = []
        not_found = []
        for product_id in product_ids:
            if product_id in products_lookup:
                sampled.append(products_lookup[product_id])
            else:
                not_found.append(product_id)
        
        if not_found:
            print(f"Warning: {len(not_found)} product IDs not found: {not_found[:5]}...")
        
        print(f"Found {len(sampled)} products from {len(product_ids)} requested IDs")
        return sampled
    
    def sample_products(self, num_products: int = 10, product_ids: Optional[List[int]] = None) -> List[Dict]:
        """Sample products - either by predefined IDs or randomly"""
        if product_ids is not None:
            # Use predefined list of product IDs
            return self.get_products_by_ids(product_ids)
        else:
            # Random sampling (backward compatibility)
            if num_products > len(self.products_data):
                num_products = len(self.products_data)
            
            sampled = random.sample(self.products_data, num_products)
            print(f"Sampled {len(sampled)} products randomly")
            return sampled
    
    def get_product_cluster(self, product: Dict) -> Optional[str]:
        """Get the cluster ID for a product"""
        cluster_id = product.get('cluster_id')
        if cluster_id is None:
            cluster_id = product.get('cluster')
        return str(cluster_id) if cluster_id is not None else None
    
    def extract_product_features(self, product: Dict) -> Dict[str, Any]:
        """Extract features from product's specifications"""
        specs = product.get('specifications', {})
        if not specs or specs is None:
            return {}
        
        product_features = {}
        
        for category, category_specs in specs.items():
            if category_specs:
                for key, value in category_specs.items():
                    if value:
                        # Create full feature name: category_key
                        full_key = f"{category}_{key}"
                        # Handle list values
                        if isinstance(value, list):
                            if len(value) > 1:
                                value_str = ", ".join(str(v).strip() for v in value)
                            else:
                                value_str = str(value[0]).strip() if value else ""
                        else:
                            value_str = str(value).strip()
                        
                        if value_str:
                            product_features[full_key] = value_str
        
        return product_features
    
    def select_highest_metric_features(self,
                                      cluster_id: str,
                                      product_features: Dict[str, Any],
                                      num_features: int = 5) -> List[Tuple[str, str, float]]:
        """
        Select features with highest custom_metric values that exist in product
        
        Args:
            cluster_id: Cluster ID
            product_features: Features available in the product
            num_features: Number of features to select
            
        Returns:
            List of (feature_name, feature_value, custom_metric) tuples, sorted by custom_metric
        """
        if cluster_id not in self.sampling_probs:
            print(f"Warning: Cluster {cluster_id} not found in sampling probabilities")
            # Fallback: return any product features
            available = list(product_features.items())
            selected = available[:num_features] if len(available) >= num_features else available
            return [(name, value, 0.0) for name, value in selected]
        
        cluster_features = self.sampling_probs[cluster_id].get('features', {})
        
        # Get features that exist in both cluster and product
        eligible_features = []
        for feature_name, feature_data in cluster_features.items():
            if feature_name in product_features:
                custom_metric = feature_data.get('custom_metric', 0.0)
                feature_value = product_features[feature_name]
                eligible_features.append((feature_name, feature_value, custom_metric))
        
        if not eligible_features:
            # Fallback: use any product features
            available = list(product_features.items())
            selected = available[:num_features] if len(available) >= num_features else available
            return [(name, value, 0.0) for name, value in selected]
        
        # Sort by custom_metric (descending)
        eligible_features.sort(key=lambda x: x[2], reverse=True)
        
        # Select top features
        selected = eligible_features[:num_features]
        
        return selected
    
    def translate_price_position(self, position: Optional[str]) -> str:
        """Convert price position label to a friendly Persian phrase."""
        mapping = {
            "budget_range": "قیمت اقتصادی در بین مشابه‌ها",
            "mid_range": "قیمت متوسط در بین مشابه‌ها",
            "premium_range": "قیمت پریمیوم در بین مشابه‌ها",
            "unknown": "نامشخص",
        }
        return mapping.get(position, "نامشخص")

    def map_label_to_text(self, label: str) -> str:
        """Provide human-friendly Persian description for comparison labels."""
        mapping = {
            "high_rated": "امتیاز کاربران نسبت به رقبا بسیار بالاست",
            "popular_choice": "تعداد نظرات کاربران نشان می‌دهد محصول پرطرفدار است",
            "good_price": "قیمت نسبت به محصولات مشابه مقرون‌به‌صرفه است",
            "premium_price": "قیمت نسبت به محصولات مشابه بالاتر است (محصول پریمیوم)",
            "high_value_pick": "ترکیب قیمت و کیفیت باعث شده ارزش خرید بالایی داشته باشد",
            "niche_pick": "محصول خاص و با مخاطبان محدود است",
            "needs_rating_improvement": "امتیاز کاربران پایین‌تر از محصولات مشابه است",
            "well_aligned": "از نظر قیمت و عملکرد با محصولات مشابه هم‌سطح است",
        }
        return mapping.get(label, label)

    def format_comparison_labels(self, product: Dict[str, Any]) -> List[str]:
        """Format comparison labels into readable Persian bullet points."""
        labels = product.get("comparison_labels", [])
        formatted = []

        for item in labels:
            if not isinstance(item, dict):
                continue

            label_key = item.get("label", "")
            sentiment = item.get("sentiment", "")
            reason = item.get("reason", "")
            price_position = item.get("price_position")
            label_text = self.map_label_to_text(label_key)

            sentiment_map = {
                "positive": "مثبت",
                "neutral": "خنثی",
                "attention": "نیازمند توجه",
            }
            sentiment_text = sentiment_map.get(sentiment, "نامشخص")

            price_position_text = (
                f" | جایگاه قیمتی: {self.translate_price_position(price_position)}"
                if price_position
                else ""
            )

            formatted.append(
                f"- {label_text} ({sentiment_text}){price_position_text}\n  علت: {reason}"
            )

        return formatted

    def format_similarity_summary(self, product: Dict[str, Any]) -> List[str]:
        """Summarize price and rating comparisons."""
        summary = product.get("similarity_summary", {})
        formatted = []

        price_info = summary.get("price", {})
        if isinstance(price_info, dict):
            price_position_label = price_info.get("price_position_label") or product.get("price_position_label")
            percentile = price_info.get("price_percentile")
            diff_percent = price_info.get("price_diff_percent_vs_similar_avg")
            if price_position_label or percentile is not None or diff_percent is not None:
                parts = []
                if price_position_label:
                    parts.append(self.translate_price_position(price_position_label))
                if percentile is not None:
                    parts.append(f"جایگاه قیمت در صدک: {percentile:.2f}")
                if diff_percent is not None:
                    parts.append(f"اختلاف قیمت با میانگین مشابه‌ها: {diff_percent:.1f}%")

                if parts:
                    formatted.append("• تحلیل قیمت: " + " | ".join(parts))

        rating_info = summary.get("rating", {})
        if isinstance(rating_info, dict):
            product_rating = rating_info.get("product_rating")
            similar_avg = rating_info.get("similar_avg")
            if product_rating is not None and similar_avg is not None:
                formatted.append(
                    f"• میانگین امتیاز کاربران: {product_rating:.1f} (میانگین مشابه‌ها: {similar_avg:.1f})"
                )

        rating_count_info = summary.get("rating_count", {})
        if isinstance(rating_count_info, dict):
            product_count = rating_count_info.get("product_rating_count")
            similar_avg_count = rating_count_info.get("similar_avg")
            if product_count is not None and similar_avg_count is not None:
                formatted.append(
                    f"• تعداد نظرات کاربران: {product_count:.0f} (میانگین مشابه‌ها: {similar_avg_count:.0f})"
                )

        return formatted

    def format_similar_products(self, product: Dict[str, Any]) -> List[str]:
        """Format top similar products for context."""
        similar_products = product.get("similar_products", []) or []
        formatted = []

        for item in similar_products[:3]:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "نامشخص")
            price_relation = item.get("price_relation", "similar")
            price_diff = item.get("price_diff_percent")
            similarity_score = item.get("similarity_score")

            price_relation_map = {
                "higher": "گران‌تر از محصول اصلی",
                "lower": "ارزان‌تر از محصول اصلی",
                "similar": "قیمت مشابه",
            }
            relation_text = price_relation_map.get(price_relation, "قیمت مشابه")

            parts = [f"- {title} ({relation_text})"]
            if price_diff is not None:
                parts.append(f" اختلاف قیمت: {price_diff:.1f}%")
            if similarity_score is not None:
                parts.append(f" امتیاز شباهت: {similarity_score:.2f}")

            formatted.append("".join(parts))

        return formatted

    def format_product_details_with_features(self, product: Dict, selected_features: List[Tuple[str, str, float]]) -> str:
        """Format product details with selected features and comparison insights for LLM prompt"""
        details = []
        details.append(f"عنوان محصول: {product.get('title', 'N/A')}")
        details.append(f"برند: {product.get('brand', 'N/A')}")
        details.append(f"دسته‌بندی: {product.get('category_name', 'N/A')}")
        
        if product.get('description'):
            details.append(f"\nتوضیحات:\n{product['description'][:500]}...")
        
        # Add selected features (highest custom_metric)
        if selected_features:
            details.append("\nویژگی‌های مهم محصول (با بالاترین اهمیت):")
            for feat_name, feat_value, custom_metric in selected_features:
                # Clean feature name (remove category prefix if too long)
                display_name = feat_name.split('_', 1)[-1] if '_' in feat_name else feat_name
                # Limit value length
                if len(feat_value) > 150:
                    feat_value = feat_value[:150] + "..."
                details.append(f"  - {display_name}: {feat_value}")
                details.append(f"    (اهمیت: {custom_metric:.6f})")
        
        price = product.get('price') or product.get('selling_price')
        if price:
            try:
                details.append(f"\nقیمت: {price:,.0f} ریال")
            except (TypeError, ValueError):
                details.append(f"\nقیمت: {price} ریال")

        # Add price position label if available
        price_position_label = product.get("price_position_label")
        if price_position_label:
            details.append(f"جایگاه قیمتی در بازار: {self.translate_price_position(price_position_label)}")

        # Add similarity summary insights
        summary_lines = self.format_similarity_summary(product)
        if summary_lines:
            details.append("\nتحلیل رقابت با محصولات مشابه:")
            details.extend(summary_lines)

        # Add comparison labels
        comparison_lines = self.format_comparison_labels(product)
        if comparison_lines:
            details.append("\nبرچسب‌های مقایسه‌ای محصول (برای لحن و توصیه):")
            details.extend(comparison_lines)

        # Add top similar products for context
        similar_lines = self.format_similar_products(product)
        if similar_lines:
            details.append("\nمحصولات مشابه شاخص (برای زمینه‌سازی گفتگو):")
            details.extend(similar_lines)
        
        return "\n".join(details)
    
    def create_dialogue_prompt(self, product: Dict, selected_features: List[Tuple[str, str, float]]) -> str:
        """Create prompt for LLM to generate dialogue"""
        product_details = self.format_product_details_with_features(product, selected_features)
        
        # Get feature names for prompt
        feature_names = [feat[0].split('_', 1)[-1] if '_' in feat[0] else feat[0] for feat in selected_features]
        
        prompt = f"""شما یک متخصص تولید گفتگوهای طبیعی بین مشتری و دستیار فروشگاه آنلاین هستید. بر اساس اطلاعات محصول زیر، یک گفتگوی عمیق، حرفه‌ای و واقعی بین کاربر (مشتری) و دستیار (فروشنده) ایجاد کنید.

الزامات گفتگو:
1. گفتگو باید طبیعی، روان و واقعی باشد
2. کاربر در ابتدا به دنبال محصولی در دسته‌بندی محصول می‌گردد
3. دستیار باید سوالات مفید و حرفه‌ای بپرسد
4. کاربر باید بر اساس ویژگی‌های انتخاب شده محصول (که در زیر آمده) پاسخ دهد
5. گفتگو باید شامل بحث درباره تمام ویژگی‌های انتخاب شده باشد: {', '.join(feature_names)}
6. در پایان، دستیار محصول را با ذکر قیمت پیشنهاد می‌دهد
7. دستیار باید از برچسب‌های مقایسه‌ای محصول (مانند ارزش خرید، قیمت، محبوبیت) برای جهت‌دهی مکالمه استفاده کند
8. دستیار باید جایگاه قیمتی محصول نسبت به رقبای مشابه را در گفتگو شرح دهد و اگر قیمت اقتصادی یا پریمیوم است، آن را توجیه کند
9. اگر محصول پرطرفدار، خاص یا دارای امتیاز بالا است، دستیار باید این موضوع را برجسته کند و دلایل آن را توضیح دهد
10. کاربر در طول مکالمه می‌تواند درباره گزینه‌های جایگزین یا نگرانی‌های قیمت/کیفیت سوال کند و دستیار با تکیه بر مقایسه‌ها پاسخ دهد
11. تعداد دورهای گفتگو باید بین 10 تا 20 دور باشد
12. گفتگو باید به فارسی و با لحن دوستانه و حرفه‌ای باشد
13. تمرکز اصلی روی ویژگی‌های انتخاب شده محصول باشد
14. مشخصاتی که در عنوان محصول ذکر شده نیز ویژگی‌های مهم است و درون گفتگو استفاده شود
15. دستیار سؤالات درباره‌ی ویژگی‌های محصول را بپرسد و در انتهای مکالمه محصول مورد نظر را با ویژگی‌های آن معرفی کند
16. در صورت اشاره به تفاوت قیمت، حتماً از داده‌های مقایسه‌ای ارائه شده استفاده شود و دلیل ارائه شود (مثلاً ارزش خرید بالا یا پریمیوم بودن)


اطلاعات محصول:
{product_details}

لطفاً یک گفتگوی کامل و طبیعی ایجاد کنید که بر روی ویژگی‌های انتخاب شده محصول تمرکز کند. خروجی را به صورت JSON با ساختار زیر برگردانید:
{{
  "dialogue": [
    {{"role": "user", "text": "متن کاربر"}},
    {{"role": "assistant", "text": "متن دستیار"}},
    ...
  ],
  "features_discussed": ["نام ویژگی 1", "نام ویژگی 2", ...],
  "dialogue_summary": "خلاصه کوتاه از گفتگو"
}}

فقط JSON را برگردانید، بدون توضیح اضافی."""
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        """Call LLM to generate dialogue"""
        system_prompt = """شما یک متخصص تولید گفتگوهای طبیعی و واقعی بین مشتری و فروشنده در فروشگاه آنلاین هستید. گفتگوها را بر اساس اطلاعات محصول و ویژگی‌های مهم آن ایجاد می‌کنید."""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2500,
                    temperature=0.8
                )
                
                result = response.choices[0].message.content.strip()
                return result
                
            except Exception as e:
                print(f"LLM error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                else:
                    raise
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON"""
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print(f"Response: {response[:500]}")
            return None
    
    def generate_dialogue(self, product: Dict, num_features: int = 5) -> Optional[Dict]:
        """Generate a dialogue for a product using highest custom_metric features"""
        product_id = product.get('id')
        product_title = product.get('title', 'Unknown')
        cluster_id = self.get_product_cluster(product)
        
        print(f"\nGenerating dialogue for product: {product_title}")
        print(f"  Cluster: {cluster_id}")
        
        # Extract product features
        product_features = self.extract_product_features(product)
        
        if not product_features:
            print(f"  Warning: No features found for product {product_id}")
            return None
        
        # Select highest custom_metric features
        selected_features = self.select_highest_metric_features(
            cluster_id, product_features, num_features
        )
        
        if not selected_features:
            print(f"  Warning: No features selected for product {product_id}")
            return None
        
        print(f"  Selected {len(selected_features)} features with highest custom_metric:")
        for feat_name, feat_value, metric in selected_features:
            display_name = feat_name.split('_', 1)[-1] if '_' in feat_name else feat_name
            print(f"    - {display_name}: metric={metric:.6f}")
        
        # Create prompt
        prompt = self.create_dialogue_prompt(product, selected_features)
        
        # Call LLM
        try:
            response = self.call_llm(prompt)
            parsed_response = self.parse_llm_response(response)
            
            if parsed_response:
                # Get similar products from product data
                similar_products = product.get('similar_products', []) or []
                
                generated_dialogue = {
                    'product_id': product_id,
                    'product_title': product_title,
                    'brand': product.get('brand'),
                    'category': product.get('category_name'),
                    'price': product.get('price') or product.get('selling_price'),
                    'cluster_id': cluster_id,
                    'type': 4,  # LLM-generated with highest metric features
                    'dialogue_rounds': len(parsed_response.get('dialogue', [])),
                    'selected_features': [
                        {
                            'feature_name': feat[0],
                            'feature_value': feat[1],
                            'custom_metric': feat[2]
                        }
                        for feat in selected_features
                    ],
                    'features_discussed': parsed_response.get('features_discussed', []),
                    'dialogue_summary': parsed_response.get('dialogue_summary', ''),
                    'dialogue': parsed_response.get('dialogue', []),
                    'llm_response_raw': response,
                    'similar_products': similar_products,  # Include similar products
                    'product_details': {
                        'title': product_title,
                        'brand': product.get('brand'),
                        'category': product.get('category_name'),
                        'price': product.get('price') or product.get('selling_price'),
                    }
                }
                return generated_dialogue
            else:
                return None
                
        except Exception as e:
            print(f"Error generating dialogue: {e}")
            return None
    
    def save_dialogue_immediately(self, dialogue: Dict, output_dir: str, dialogue_count: int, num_features: int = 5):
        """Save a single dialogue immediately after generation"""
        # Save individual file
        individual_path = os.path.join(output_dir, f'dialogue_{dialogue_count}_{dialogue["product_id"]}.json')
        with open(individual_path, 'w', encoding='utf-8') as f:
            json.dump(dialogue, f, ensure_ascii=False, indent=2)
        
        # Update main collection file
        output_path = os.path.join(output_dir, 'llm_generated_dialogues_highest_metric.json')
        
        # Load existing data if file exists
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
        else:
            output_data = {
                'metadata': {
                    'model': self.model,
                    'source_file': self.products_file,
                    'selection_method': 'highest_custom_metric',
                    'start_date': datetime.now().isoformat(),
                    'num_features_per_product': num_features
                },
                'generated_dialogues': []
            }
        
        # Add new dialogue
        output_data['generated_dialogues'].append(dialogue)
        output_data['metadata']['total_generated'] = len(output_data['generated_dialogues'])
        output_data['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save updated data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return individual_path, output_path
    
    def generate_batch_continuous(self, num_features: int = 5, 
                                  output_dir: str = "llm_generated_dialogues_highest_metric"):
        """
        Continuously generate dialogues by randomly selecting products until API fails.
        Saves each dialogue immediately after generation.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nStarting continuous dialogue generation...")
        print(f"Using top {num_features} features with highest custom_metric for each product")
        print(f"Will continue until API key fails or is exhausted")
        print(f"Model: {self.model}")
        print("-" * 60)
        
        dialogue_count = 0
        api_failed = False
        
        while not api_failed:
            # Randomly select a product
            product = random.choice(self.products_data)
            dialogue_count += 1
            
            print(f"\n[Round {dialogue_count}] Randomly selected product: {product.get('title', 'Unknown')} (ID: {product.get('id')})")
            
            try:
                dialogue = self.generate_dialogue(product, num_features=num_features)
                
                if dialogue:
                    self.generated_dialogues.append(dialogue)
                    # Save immediately
                    individual_path, main_path = self.save_dialogue_immediately(dialogue, output_dir, dialogue_count, num_features)
                    print(f"✅ Dialogue generated and saved ({dialogue['dialogue_rounds']} rounds)")
                    print(f"   Individual file: {individual_path}")
                    print(f"   Total dialogues so far: {len(self.generated_dialogues)}")
                else:
                    print(f"⚠️  Failed to generate dialogue (product may have no features)")
                    # Don't treat this as API failure, continue with next product
                
                # Delay between requests to avoid rate limits
                time.sleep(self.delay_between_requests)
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check if it's an API key/authentication error
                if any(keyword in error_msg for keyword in ['api', 'key', 'authentication', 'unauthorized', 'forbidden', 'quota', 'limit', 'insufficient']):
                    print(f"\n❌ API Error detected: {e}")
                    print(f"Stopping generation. API key may be invalid or exhausted.")
                    api_failed = True
                else:
                    # Other errors - log but continue
                    print(f"⚠️  Error generating dialogue: {e}")
                    print(f"Continuing with next product...")
                    time.sleep(self.delay_between_requests)
        
        # Final save and statistics
        print(f"\n" + "=" * 60)
        print(f"Generation stopped after {dialogue_count} rounds")
        print(f"Successfully generated {len(self.generated_dialogues)} dialogues")
        
        # Print statistics
        if self.generated_dialogues:
            avg_rounds = sum(d['dialogue_rounds'] for d in self.generated_dialogues) / len(self.generated_dialogues)
            avg_features = sum(len(d.get('features_discussed', [])) for d in self.generated_dialogues) / len(self.generated_dialogues)
            avg_metric = sum(
                sum(f.get('custom_metric', 0) for f in d.get('selected_features', []))
                for d in self.generated_dialogues
            ) / len(self.generated_dialogues)
            
            print(f"\n📊 Final Statistics:")
            print(f"  Total dialogues generated: {len(self.generated_dialogues)}")
            print(f"  Average dialogue rounds: {avg_rounds:.1f}")
            print(f"  Average features discussed: {avg_features:.1f}")
            print(f"  Average total custom_metric per product: {avg_metric:.6f}")
        
        output_path = os.path.join(output_dir, 'llm_generated_dialogues_highest_metric.json')
        return output_path


def main():
    """Main function"""
    print("LLM Dialogue Generator with Highest Custom Metric Features")
    print("=" * 60)
    
    # Configuration
    products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    sampling_probabilities_file = "cluster_sampling_probability_results_custome/cluster_sampling_probabilities.json"
    num_features = 5  # Number of highest custom_metric features to use
    output_dir = "full_dialogues_dataset"
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set.")
        print("   Please set it using: export OPENAI_API_KEY='your-key-here'")
        print("   Or the script will fail when making API calls.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Initialize generator
    generator = HighestMetricDialogueGenerator(
        products_file=products_file,
        sampling_probabilities_file=sampling_probabilities_file,
        api_key=api_key
    )
    
    # Load data
    generator.load_data()
    
    # Generate dialogues continuously until API fails
    output_path = generator.generate_batch_continuous(
        num_features=num_features,
        output_dir=output_dir
    )
    
    print(f"\n🎉 Process completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()

