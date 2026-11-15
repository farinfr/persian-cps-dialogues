#!/usr/bin/env python3
"""
LLM Dialogue Generator from Product Details
==========================================

This script samples products and asks the LLM to generate dialogues
from scratch based only on product details (brand, category, price, specifications).
"""

import json
import os
import time
import random
from typing import Dict, List, Any, Optional
from openai import OpenAI
from datetime import datetime

class LLMDialogueGenerator:
    def __init__(self, 
                 products_file: str,
                 api_key: str = None,
                 base_url: str = None):
        self.products_file = products_file
        self.products_data = None
        self.generated_dialogues = []
        
        # Initialize LLM client - Using Avala AI API
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # LLM configuration
        self.model = "meta.llama3-1-405b-instruct-v1:0"  # Using Meta LLaMA 3.1 405B Instruct on Avala AI
        self.max_retries = 3
        self.delay_between_requests = 3.0  # Longer delay to avoid rate limits
        
    def load_data(self):
        """Load products data"""
        print("Loading products...")
        with open(self.products_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'products' in data:
            self.products_data = data['products']
        else:
            self.products_data = data
        
        print(f"Loaded {len(self.products_data)} products")
        return self.products_data
    
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
    
    def sample_features_randomly(self, product_features: Dict[str, Any], num_features: int = 5) -> Dict[str, Any]:
        """Randomly sample features from product specifications"""
        available = list(product_features.items())
        if len(available) <= num_features:
            selected = available
        else:
            selected = random.sample(available, num_features)
        return dict(selected)
    
    def format_product_details(self, product: Dict, sampled_features: Dict[str, Any]) -> str:
        """Format product details with sampled features for LLM prompt"""
        details = []
        details.append(f"عنوان محصول: {product.get('title', 'N/A')}")
        details.append(f"برند: {product.get('brand', 'N/A')}")
        details.append(f"دسته‌بندی: {product.get('category_name', 'N/A')}")
        
        if product.get('description'):
            details.append(f"\nتوضیحات:\n{product['description'][:500]}...")
        
        # Add only sampled features
        if sampled_features:
            details.append("\nویژگی‌های انتخاب شده محصول:")
            for feature_name, feature_value in sampled_features.items():
                # Clean feature name (remove category prefix if too long)
                display_name = feature_name.split('_', 1)[-1] if '_' in feature_name else feature_name
                # Limit value length
                if len(feature_value) > 150:
                    feature_value = feature_value[:150] + "..."
                details.append(f"  - {display_name}: {feature_value}")
        
        price = product.get('price') or product.get('selling_price')
        if price:
            try:
                details.append(f"\nقیمت: {price:,.0f} ریال")
            except (TypeError, ValueError):
                details.append(f"\nقیمت: {price} ریال")
        
        return "\n".join(details)
    
    def create_dialogue_prompt(self, product: Dict, sampled_features: Dict[str, Any]) -> str:
        """Create prompt for LLM to generate dialogue"""
        product_details = self.format_product_details(product, sampled_features)
        
        # Get feature names for prompt
        feature_names = [name.split('_', 1)[-1] if '_' in name else name for name in sampled_features.keys()]
        
        prompt = f"""شما یک متخصص تولید گفتگوهای طبیعی بین مشتری و دستیار فروشگاه آنلاین هستید. بر اساس اطلاعات محصول زیر، یک گفتگوی طبیعی و واقعی بین کاربر (مشتری) و دستیار (فروشنده) ایجاد کنید.

الزامات گفتگو:
1. گفتگو باید طبیعی، روان و واقعی باشد
2. کاربر در ابتدا به دنبال محصولی در دسته‌بندی محصول می‌گردد
3. دستیار باید سوالات مفید و حرفه‌ای بپرسد
4. کاربر باید بر اساس ویژگی‌های انتخاب شده محصول (که در زیر آمده) پاسخ دهد
5. گفتگو باید شامل بحث درباره تمام ویژگی‌های انتخاب شده باشد: {', '.join(feature_names)}
6. در پایان، دستیار محصول را با ذکر قیمت پیشنهاد می‌دهد
7. تعداد دورهای گفتگو باید بین 10 تا 20 دور باشد
8. گفتگو باید به فارسی و با لحن دوستانه و حرفه‌ای باشد
9. تمرکز اصلی روی ویژگی‌های انتخاب شده محصول باشد

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
        system_prompt = """شما یک متخصص تولید گفتگوهای طبیعی و واقعی بین مشتری و فروشنده در فروشگاه آنلاین هستید. گفتگوها را بر اساس اطلاعات محصول ایجاد می‌کنید."""
        
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
        # Try to extract JSON from response
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
        """Generate a dialogue for a product using LLM with randomly sampled features"""
        product_id = product.get('id')
        product_title = product.get('title', 'Unknown')
        
        print(f"\nGenerating dialogue for product: {product_title}")
        
        # Extract product features
        product_features = self.extract_product_features(product)
        
        if not product_features:
            print(f"  Warning: No features found for product {product_id}")
            return None
        
        # Randomly sample features
        sampled_features = self.sample_features_randomly(product_features, num_features)
        
        print(f"  Sampled {len(sampled_features)} features randomly:")
        for feat_name, feat_value in sampled_features.items():
            display_name = feat_name.split('_', 1)[-1] if '_' in feat_name else feat_name
            print(f"    - {display_name}")
        
        # Create prompt with sampled features
        prompt = self.create_dialogue_prompt(product, sampled_features)
        
        # Call LLM
        try:
            response = self.call_llm(prompt)
            parsed_response = self.parse_llm_response(response)
            
            if parsed_response:
                generated_dialogue = {
                    'product_id': product_id,
                    'product_title': product_title,
                    'brand': product.get('brand'),
                    'category': product.get('category_name'),
                    'price': product.get('price') or product.get('selling_price'),
                    'cluster_id': product.get('cluster_id'),
                    'type': 3,  # LLM-generated with random features
                    'dialogue_rounds': len(parsed_response.get('dialogue', [])),
                    'sampled_features': [
                        {
                            'feature_name': name.split('_', 1)[-1] if '_' in name else name,
                            'feature_value': value
                        }
                        for name, value in sampled_features.items()
                    ],
                    'features_discussed': parsed_response.get('features_discussed', []),
                    'dialogue_summary': parsed_response.get('dialogue_summary', ''),
                    'dialogue': parsed_response.get('dialogue', []),
                    'llm_response_raw': response,
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
    
    def generate_batch(self, num_products: int = 10, output_dir: str = "llm_generated_dialogues",
                      product_ids: Optional[List[int]] = None, num_features: int = 5):
        """Generate dialogues for multiple products"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample products (either by IDs or randomly)
        sampled_products = self.sample_products(num_products, product_ids=product_ids)
        
        print(f"\nStarting to generate {len(sampled_products)} dialogues...")
        print(f"Using random feature sampling ({num_features} features per product)")
        
        for idx, product in enumerate(sampled_products, 1):
            print(f"\n[{idx}/{len(sampled_products)}] Processing product...")
            
            dialogue = self.generate_dialogue(product, num_features=num_features)
            
            if dialogue:
                self.generated_dialogues.append(dialogue)
                print(f"✅ Dialogue generated successfully ({dialogue['dialogue_rounds']} rounds)")
            else:
                print(f"❌ Failed to generate dialogue")
            
            # Delay between requests to avoid rate limits
            if idx < len(sampled_products):
                time.sleep(self.delay_between_requests)
        
        # Save results
        output_data = {
            'metadata': {
                'total_generated': len(self.generated_dialogues),
                'generation_date': datetime.now().isoformat(),
                'model': self.model,
                'source_file': self.products_file
            },
            'generated_dialogues': self.generated_dialogues
        }
        
        output_path = os.path.join(output_dir, 'llm_generated_dialogues.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Saved {len(self.generated_dialogues)} generated dialogues to {output_path}")
        
        # Also save individual files for easier review
        for idx, dialogue in enumerate(self.generated_dialogues):
            individual_path = os.path.join(output_dir, f'dialogue_{idx+1}_{dialogue["product_id"]}.json')
            with open(individual_path, 'w', encoding='utf-8') as f:
                json.dump(dialogue, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved individual dialogue files to {output_dir}/")
        
        # Print statistics
        if self.generated_dialogues:
            avg_rounds = sum(d['dialogue_rounds'] for d in self.generated_dialogues) / len(self.generated_dialogues)
            avg_features = sum(len(d.get('features_discussed', [])) for d in self.generated_dialogues) / len(self.generated_dialogues)
            
            print(f"\n📊 Statistics:")
            print(f"  Average dialogue rounds: {avg_rounds:.1f}")
            print(f"  Average features discussed: {avg_features:.1f}")
        
        return output_path


def main():
    """Main function"""
    print("LLM Dialogue Generator from Product Details")
    print("=" * 60)
    
    # Configuration
    products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    num_products = 10
    num_features = 5  # Number of features to randomly sample
    output_dir = "llm_generated_dialogues"
    
    # Product IDs from dual_method_dialogues (same products used in dual method generation)
    product_ids = [4477, 11767, 4863, 10101, 7643, 1454, 6669, 8311, 3408, 5890]
    
    # Initialize generator
    generator = LLMDialogueGenerator(products_file=products_file)
    
    # Load data
    generator.load_data()
    
    # Generate dialogues
    output_path = generator.generate_batch(
        num_products=num_products, 
        output_dir=output_dir,
        product_ids=product_ids,
        num_features=num_features
    )
    
    print(f"\n🎉 Process completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()

