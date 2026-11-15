#!/usr/bin/env python3
"""
Dual Method Dialogue Generator and Refiner
==========================================

This script:
1. Samples 10 products (once, same products for both methods)
2. Creates template dialogues using 2 methods:
   - Method 1: Random feature sampling
   - Method 2: Highest custom_metric feature selection
3. Rewrites both sets with LLM
4. Refines them
"""

import json
import os
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from datetime import datetime

# Import shared product IDs configuration
try:
    from dialogue_generation_config import TARGET_PRODUCT_IDS
except ImportError:
    # Fallback if config file doesn't exist
    TARGET_PRODUCT_IDS = [4477, 11767, 4863, 10101, 7643, 1454, 6669, 8311, 3408, 5890]

class DualMethodDialogueGenerator:
    def __init__(self,
                 products_file: str,
                 sampling_probabilities_file: str,
                 api_key: str = None,
                 base_url: str = None):
        self.products_file = products_file
        self.sampling_probabilities_file = sampling_probabilities_file
        self.products_data = None
        self.sampling_probs = {}
        self.sampled_products = []
        self.random_dialogues = []
        self.highest_metric_dialogues = []
        self.llm_rewritten_random = []
        self.llm_rewritten_highest_metric = []
        
        # Initialize LLM client - Using Avala AI API
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.model = "meta.llama3-1-405b-instruct-v1:0"  # Using Meta LLaMA 3.1 405B Instruct on Avala AI
        self.max_retries = 3
        self.delay_between_requests = 3.0
        
        # Template dialogue components
        self._init_templates()
    
    def _init_templates(self):
        """Initialize dialogue templates"""
        self.user_search_templates = [
            "می‌خواهم محصولی در دسته {category} پیدا کنم.",
            "به دنبال محصولی در گروه {category} هستم.",
            "آیا محصولی در دسته‌بندی {category} دارید؟",
        ]
        
        self.assistant_search_response_templates = [
            "بله، در دسته {category} محصولات خوبی داریم. برای کمک بهتر، لطفا {feature} مورد نظر خود را بگویید.",
            "حتما. در گروه {category} محصولات متنوعی موجود است. می‌توانید {feature} مورد نظر خود را مشخص کنید؟",
        ]
        
        self.user_answer_templates = [
            "می‌خواهم {feature} {value} باشد.",
            "من به دنبال محصولی با {feature} {value} می‌گردم.",
            "ترجیح می‌دهم {feature} {value} باشد.",
        ]
        
        self.assistant_follow_up_templates = [
            "بسیار خوب، پس {feature} {value} می‌خواهید. در مورد {next_feature} چه می‌خواهید؟",
            "عالی، پس {feature} {value} است. برای پیدا کردن محصول بهتر، لطفا {next_feature} را هم بگویید.",
        ]
        
        self.closing_templates = [
            "عالی بود. بر اساس مشخصاتی که گفتید، {title} از برند {brand} با قیمت {price} تومان می‌تواند مناسب شما باشد.",
            "بسیار خوشحالم. بر اساس نیازهایی که گفتید، {title} {brand} با قیمت {price} تومان را پیشنهاد می‌کنم.",
        ]
    
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
    
    def sample_products(self, num_products: int = 10, product_ids: Optional[List[int]] = None) -> List[Dict]:
        """Sample products - either by predefined IDs or randomly"""
        if product_ids is not None:
            products_lookup = {p.get('id'): p for p in self.products_data if p.get('id') is not None}
            sampled = [products_lookup[pid] for pid in product_ids if pid in products_lookup]
            print(f"Found {len(sampled)} products from {len(product_ids)} requested IDs")
            return sampled
        else:
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
                        full_key = f"{category}_{key}"
                        if isinstance(value, list):
                            value_str = ", ".join(str(v).strip() for v in value) if len(value) > 1 else str(value[0]).strip() if value else ""
                        else:
                            value_str = str(value).strip()
                        if value_str:
                            product_features[full_key] = value_str
        return product_features
    
    def sample_features_randomly(self, product_features: Dict[str, Any], num_features: int = 5) -> List[Tuple[str, str]]:
        """Sample features randomly from product"""
        available = list(product_features.items())
        if len(available) <= num_features:
            selected = available
        else:
            selected = random.sample(available, num_features)
        return [(name, value) for name, value in selected]
    
    def select_highest_metric_features(self, cluster_id: str, product_features: Dict[str, Any], 
                                       num_features: int = 5) -> List[Tuple[str, str, float]]:
        """Select features with highest custom_metric values"""
        if cluster_id not in self.sampling_probs:
            available = list(product_features.items())
            selected = available[:num_features] if len(available) >= num_features else available
            return [(name, value, 0.0) for name, value in selected]
        
        cluster_features = self.sampling_probs[cluster_id].get('features', {})
        eligible_features = []
        for feature_name, feature_data in cluster_features.items():
            if feature_name in product_features:
                custom_metric = feature_data.get('custom_metric', 0.0)
                feature_value = product_features[feature_name]
                eligible_features.append((feature_name, feature_value, custom_metric))
        
        if not eligible_features:
            available = list(product_features.items())
            selected = available[:num_features] if len(available) >= num_features else available
            return [(name, value, 0.0) for name, value in selected]
        
        eligible_features.sort(key=lambda x: x[2], reverse=True)
        return eligible_features[:num_features]
    
    def format_price(self, price: Optional[float]) -> str:
        """Format price for display"""
        if price is None or price == 0:
            return "نامشخص"
        try:
            return f"{price:,.0f}"
        except (TypeError, ValueError):
            return "نامشخص"
    
    def clean_feature_name(self, name: str) -> str:
        """Clean feature name for display"""
        return name.split('_', 1)[-1] if '_' in name else name
    
    def generate_template_dialogue(self, product: Dict, selected_features: List[Tuple], 
                               method: str = "random") -> Dict[str, Any]:
        """Generate template dialogue for a product"""
        title = product.get('title', 'محصول')
        brand = product.get('brand', 'نامشخص')
        category = product.get('category_name', 'نامشخص')
        price = product.get('price') or product.get('selling_price') or None
        
        dialogue = []
        
        # Round 1: User searches by category
        user_search = random.choice(self.user_search_templates).format(category=category)
        dialogue.append({"role": "user", "text": user_search})
        
        # Round 2: Assistant asks about first feature
        if selected_features:
            first_feature_name = self.clean_feature_name(selected_features[0][0])
            assistant_response = random.choice(self.assistant_search_response_templates).format(
                category=category,
                feature=first_feature_name
            )
            dialogue.append({"role": "assistant", "text": assistant_response})
            
            # Rounds 3+: User answers and assistant asks about next feature
            for idx, feature_data in enumerate(selected_features):
                feature_name = feature_data[0]
                feature_value = feature_data[1] if len(feature_data) > 1 else ""
                cleaned_name = self.clean_feature_name(feature_name)
                
                # User answers
                user_answer = random.choice(self.user_answer_templates).format(
                    feature=cleaned_name,
                    value=feature_value
                )
                dialogue.append({"role": "user", "text": user_answer})
                
                # Assistant responds
                if idx < len(selected_features) - 1:
                    next_feature_name = self.clean_feature_name(selected_features[idx + 1][0])
                    assistant_follow_up = random.choice(self.assistant_follow_up_templates).format(
                        feature=cleaned_name,
                        value=feature_value,
                        next_feature=next_feature_name
                    )
                    dialogue.append({"role": "assistant", "text": assistant_follow_up})
        
        # Closing
        closing = random.choice(self.closing_templates).format(
            title=title,
            brand=brand,
            price=self.format_price(price)
        )
        dialogue.append({"role": "assistant", "text": closing})
        
        return {
            "product_id": product.get('id'),
            "product_title": title,
            "brand": brand,
            "category": category,
            "price": price,
            "cluster_id": self.get_product_cluster(product),
            "method": method,
            "type": 1 if method == "highest_metric" else 2,  # 1 for highest metric, 2 for random
            "dialogue_rounds": len(dialogue),
            "num_features": len(selected_features),
            "features": [
                {
                    "feature_name": self.clean_feature_name(f[0]),
                    "feature_value": f[1] if len(f) > 1 else "",
                    "custom_metric": f[2] if len(f) > 2 else 0.0
                }
                for f in selected_features
            ],
            "dialogue": dialogue
        }
    
    def call_llm(self, prompt: str) -> str:
        """Call LLM to rewrite dialogue"""
        system_prompt = """شما یک متخصص بازنویسی و بهبود گفتگوهای طبیعی بین مشتری و فروشنده هستید. گفتگوها را طبیعی‌تر، روان‌تر و واقعی‌تر کنید."""
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.8
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_between_requests * (attempt + 1))
                else:
                    raise
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response to extract JSON"""
        response = response.strip()
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
            return None
    
    def rewrite_dialogue_with_llm(self, dialogue: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Rewrite dialogue using LLM"""
        dialogue_text = "\n".join([
            f"{'کاربر' if turn['role'] == 'user' else 'دستیار'}: {turn['text']}"
            for turn in dialogue['dialogue']
        ])
        
        features_text = ", ".join([f['feature_name'] for f in dialogue['features']])
        
        prompt = f"""لطفا گفتگوی زیر را بازنویسی کنید تا طبیعی‌تر، روان‌تر و واقعی‌تر شود. گفتگو باید شامل بحث درباره این ویژگی‌ها باشد: {features_text}

گفتگوی فعلی:
{dialogue_text}

لطفاً گفتگوی بازنویسی شده را به صورت JSON با ساختار زیر برگردانید:
{{
  "dialogue": [
    {{"role": "user", "text": "متن کاربر"}},
    {{"role": "assistant", "text": "متن دستیار"}},
    ...
  ],
  "improvements": "توضیح کوتاه از بهبودهای اعمال شده"
}}

فقط JSON را برگردانید."""
        
        try:
            response = self.call_llm(prompt)
            parsed = self.parse_llm_response(response)
            
            if parsed:
                rewritten = dialogue.copy()
                rewritten['dialogue'] = parsed.get('dialogue', dialogue['dialogue'])
                rewritten['llm_improvements'] = parsed.get('improvements', '')
                rewritten['llm_rewritten'] = True
                rewritten['original_dialogue_rounds'] = dialogue['dialogue_rounds']
                rewritten['rewritten_dialogue_rounds'] = len(parsed.get('dialogue', []))
                return rewritten
        except Exception as e:
            print(f"Error rewriting dialogue: {e}")
        
        return None
    
    def process_products(self, num_products: int = 10, product_ids: Optional[List[int]] = None, 
                         num_features: int = 5):
        """Process products and generate dialogues with both methods"""
        # Sample products once
        self.sampled_products = self.sample_products(num_products, product_ids)
        
        print(f"\nProcessing {len(self.sampled_products)} products with both methods...")
        
        for idx, product in enumerate(self.sampled_products, 1):
            product_id = product.get('id')
            product_title = product.get('title', 'Unknown')
            cluster_id = self.get_product_cluster(product)
            
            print(f"\n[{idx}/{len(self.sampled_products)}] Product {product_id}: {product_title[:50]}...")
            
            # Extract product features
            product_features = self.extract_product_features(product)
            
            if not product_features:
                print(f"  ⚠️  No features found, skipping...")
                continue
            
            # Method 1: Random features
            print(f"  Method 1: Random feature sampling...")
            random_features = self.sample_features_randomly(product_features, num_features)
            random_dialogue = self.generate_template_dialogue(product, random_features, method="random")
            self.random_dialogues.append(random_dialogue)
            print(f"    ✅ Generated template dialogue ({random_dialogue['dialogue_rounds']} rounds)")
            
            # Method 2: Highest custom_metric features
            print(f"  Method 2: Highest custom_metric features...")
            highest_metric_features = self.select_highest_metric_features(
                cluster_id, product_features, num_features
            )
            highest_metric_dialogue = self.generate_template_dialogue(
                product, highest_metric_features, method="highest_metric"
            )
            self.highest_metric_dialogues.append(highest_metric_dialogue)
            print(f"    ✅ Generated template dialogue ({highest_metric_dialogue['dialogue_rounds']} rounds)")
    
    def rewrite_all_dialogues(self):
        """Rewrite all dialogues with LLM"""
        print(f"\n{'='*60}")
        print("Rewriting dialogues with LLM...")
        print(f"{'='*60}")
        
        # Rewrite random method dialogues
        print(f"\nRewriting {len(self.random_dialogues)} random method dialogues...")
        for idx, dialogue in enumerate(self.random_dialogues, 1):
            print(f"[{idx}/{len(self.random_dialogues)}] Rewriting dialogue for product {dialogue['product_id']}...")
            rewritten = self.rewrite_dialogue_with_llm(dialogue)
            if rewritten:
                self.llm_rewritten_random.append(rewritten)
                print(f"  ✅ Rewritten ({rewritten['rewritten_dialogue_rounds']} rounds)")
            else:
                print(f"  ❌ Failed to rewrite")
            time.sleep(self.delay_between_requests)
        
        # Rewrite highest metric method dialogues
        print(f"\nRewriting {len(self.highest_metric_dialogues)} highest metric method dialogues...")
        for idx, dialogue in enumerate(self.highest_metric_dialogues, 1):
            print(f"[{idx}/{len(self.highest_metric_dialogues)}] Rewriting dialogue for product {dialogue['product_id']}...")
            rewritten = self.rewrite_dialogue_with_llm(dialogue)
            if rewritten:
                self.llm_rewritten_highest_metric.append(rewritten)
                print(f"  ✅ Rewritten ({rewritten['rewritten_dialogue_rounds']} rounds)")
            else:
                print(f"  ❌ Failed to rewrite")
            time.sleep(self.delay_between_requests)
    
    def save_results(self, output_dir: str = "dual_method_dialogues"):
        """Save all results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save template dialogues
        template_data = {
            'metadata': {
                'total_products': len(self.sampled_products),
                'generation_date': datetime.now().isoformat(),
                'methods': ['random', 'highest_metric']
            },
            'random_method': {
                'dialogues': self.random_dialogues
            },
            'highest_metric_method': {
                'dialogues': self.highest_metric_dialogues
            }
        }
        
        template_path = os.path.join(output_dir, 'template_dialogues.json')
        with open(template_path, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved template dialogues to {template_path}")
        
        # Save LLM rewritten dialogues
        rewritten_data = {
            'metadata': {
                'total_products': len(self.sampled_products),
                'rewrite_date': datetime.now().isoformat(),
                'model': self.model
            },
            'random_method_rewritten': {
                'dialogues': self.llm_rewritten_random
            },
            'highest_metric_method_rewritten': {
                'dialogues': self.llm_rewritten_highest_metric
            }
        }
        
        rewritten_path = os.path.join(output_dir, 'llm_rewritten_dialogues.json')
        with open(rewritten_path, 'w', encoding='utf-8') as f:
            json.dump(rewritten_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved LLM rewritten dialogues to {rewritten_path}")
        
        # Save individual files
        for dialogue in self.llm_rewritten_random:
            path = os.path.join(output_dir, f"random_rewritten_{dialogue['product_id']}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dialogue, f, ensure_ascii=False, indent=2)
        
        for dialogue in self.llm_rewritten_highest_metric:
            path = os.path.join(output_dir, f"highest_metric_rewritten_{dialogue['product_id']}.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dialogue, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved individual dialogue files to {output_dir}/")
        
        # Print statistics
        print(f"\n📊 Statistics:")
        print(f"  Total products: {len(self.sampled_products)}")
        print(f"  Random method - Template: {len(self.random_dialogues)}, LLM Rewritten: {len(self.llm_rewritten_random)}")
        print(f"  Highest metric method - Template: {len(self.highest_metric_dialogues)}, LLM Rewritten: {len(self.llm_rewritten_highest_metric)}")
        
        return template_path, rewritten_path


def main():
    """Main function"""
    print("Dual Method Dialogue Generator and Refiner")
    print("=" * 60)
    
    # Configuration
    products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    sampling_probabilities_file = "cluster_sampling_probability_results_custome/cluster_sampling_probabilities.json"
    num_products = 10
    num_features = 5
    output_dir = "dual_method_dialogues"
    
    # Use shared product IDs from config (same as other dialogue generators)
    # This ensures we don't randomly choose products
    product_ids = TARGET_PRODUCT_IDS
    
    # Initialize generator
    generator = DualMethodDialogueGenerator(
        products_file=products_file,
        sampling_probabilities_file=sampling_probabilities_file
    )
    
    # Load data
    generator.load_data()
    
    # Process products and generate template dialogues
    generator.process_products(
        num_products=num_products,
        product_ids=product_ids,
        num_features=num_features
    )
    
    # Rewrite with LLM
    generator.rewrite_all_dialogues()
    
    # Save results
    template_path, rewritten_path = generator.save_results(output_dir=output_dir)
    
    print(f"\n🎉 Process completed!")
    print(f"  Template dialogues: {template_path}")
    print(f"  LLM rewritten dialogues: {rewritten_path}")


if __name__ == "__main__":
    main()

