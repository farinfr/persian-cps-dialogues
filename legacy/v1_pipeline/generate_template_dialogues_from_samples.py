#!/usr/bin/env python3
"""
Template Dialogue Generator from Sampled Products
=================================================

This script generates template-based dialogues using:
- Brand
- Category
- Price
- Selected specifications (from sampled products)

It uses the sampled products from batch_500_products_5_features.json
"""

import json
import random
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

class TemplateDialogueGenerator:
    def __init__(self,
                 sampled_products_file: str,
                 full_products_file: str):
        """
        Initialize the dialogue generator
        
        Args:
            sampled_products_file: Path to sampled products JSON file
            full_products_file: Path to full products JSON file (for price and other details)
        """
        self.sampled_products_file = sampled_products_file
        self.full_products_file = full_products_file
        self.samples = []
        self.products_lookup = {}
        self.dialogues = []
        
        # Dialogue templates
        self.user_search_templates = [
            "می‌خواهم محصولی در دسته {category} پیدا کنم.",
            "به دنبال محصولی در گروه {category} هستم.",
            "آیا محصولی در دسته‌بندی {category} دارید؟",
            "می‌خواهم محصولی در {category} جستجو کنم.",
            "لطفا محصولی در دسته {category} به من نشان دهید.",
            "من به دنبال محصولی در دسته‌بندی {category} می‌گردم.",
        ]
        
        self.user_brand_templates = [
            "آیا محصولی از برند {brand} دارید؟",
            "من به دنبال محصولی از {brand} می‌گردم.",
            "می‌خواهم محصولی از برند {brand} ببینم.",
            "آیا {brand} محصولی در این دسته دارد؟",
            "ترجیح می‌دهم محصول از برند {brand} باشد.",
        ]
        
        self.user_price_templates = [
            "بودجه من حدود {price} ریال است.",
            "می‌خواهم محصولی با قیمت حدود {price} ریال.",
            "آیا محصولی در محدوده قیمت {price} ریال دارید؟",
            "من به دنبال محصولی با قیمت {price} ریال هستم.",
            "حدود {price} ریال می‌خواهم هزینه کنم.",
        ]
        
        self.assistant_search_response_templates = [
            "بله، در دسته {category} محصولات خوبی داریم. برای کمک بهتر، لطفا {feature} مورد نظر خود را بگویید.",
            "حتما. در گروه {category} محصولات متنوعی موجود است. می‌توانید {feature} مورد نظر خود را مشخص کنید؟",
            "خوشحالم که می‌توانم کمک کنم. برای یافتن محصول مناسب در {category}، لطفا {feature} را مشخص کنید.",
            "بله، در خدمت شما هستم. در دسته {category} محصولات مختلفی داریم. در مورد {feature} چه می‌خواهید؟",
            "بسیار خوشحالم. برای پیدا کردن محصول مناسب در {category}، لطفا {feature} مورد نظر را بگویید.",
        ]
        
        self.assistant_feature_questions = [
            "در مورد {feature} چه می‌خواهید؟",
            "آیا {feature} خاصی مد نظر دارید؟",
            "لطفا {feature} مورد نظر خود را بگویید.",
            "برای یافتن محصول مناسب، {feature} را مشخص کنید.",
            "در مورد {feature} چه ترجیحی دارید؟",
        ]
        
        self.user_answer_templates = [
            "می‌خواهم {feature} {value} باشد.",
            "من به دنبال محصولی با {feature} {value} می‌گردم.",
            "ترجیح می‌دهم {feature} {value} باشد.",
            "آیا محصولی با {feature} {value} دارید؟",
            "من می‌خواهم {feature} {value} باشد.",
            "محصولی با {feature} {value} می‌خواهم.",
        ]
        
        self.assistant_follow_up_templates = [
            "بسیار خوب، پس {feature} {value} می‌خواهید. در مورد {next_feature} چه می‌خواهید؟",
            "عالی، پس {feature} {value} است. برای پیدا کردن محصول بهتر، لطفا {next_feature} را هم بگویید.",
            "درست است. {feature} {value} را ثبت کردم. در مورد {next_feature} چه می‌خواهید؟",
            "بله، {feature} {value} را ثبت کردم. لطفا {next_feature} مورد نظر را هم بگویید.",
            "خوب، پس {feature} {value} می‌خواهید. در مورد {next_feature} چه می‌خواهید؟",
        ]
        
        self.assistant_confirmation_templates = [
            "بله، {feature} {value} را ثبت کردم.",
            "عالی، {feature} {value} را یادداشت کردم.",
            "بسیار خوب، پس {feature} {value} است.",
            "درست است، {feature} {value} را ثبت کردم.",
            "خوب، {feature} {value} را یادداشت کردم.",
        ]
        
        self.assistant_brand_response_templates = [
            "بله، محصولات خوبی از برند {brand} داریم.",
            "حتما، {brand} محصولات با کیفیتی در این دسته دارد.",
            "بله، {brand} یکی از برندهای معتبر در این زمینه است.",
            "حتما، محصولات {brand} در دسترس هستند.",
        ]
        
        self.assistant_price_response_templates = [
            "بله، محصولاتی در این محدوده قیمت داریم.",
            "حتما، در این محدوده قیمت گزینه‌های خوبی موجود است.",
            "بله، می‌توانم محصولاتی با این قیمت به شما پیشنهاد دهم.",
            "حتما، در این محدوده قیمت محصولات متنوعی داریم.",
        ]
        
        self.closing_templates = [
            "عالی بود. بر اساس مشخصاتی که گفتید، {title} از برند {brand} با قیمت {price} ریال می‌تواند مناسب شما باشد. اگر سوال دیگری دارید، خوشحال می‌شوم کمک کنم.",
            "بسیار خوشحالم. بر اساس نیازهایی که گفتید، {title} {brand} با قیمت {price} ریال را پیشنهاد می‌کنم. اگر اطلاعات بیشتری نیاز دارید، در خدمت هستم.",
            "خوب، بر اساس ویژگی‌هایی که مشخص کردید، {title} از {brand} با قیمت {price} ریال می‌تواند گزینه مناسبی باشد. سوال دیگری دارید؟",
            "بر اساس ترجیحاتی که گفتید، {title} {brand} با قیمت {price} ریال را پیشنهاد می‌دهم. امیدوارم مفید بوده باشد.",
            "عالی، با توجه به مشخصاتی که گفتید، {title} از برند {brand} با قیمت {price} ریال می‌تواند برای شما مناسب باشد. اگر نیاز به اطلاعات بیشتری دارید، بفرمایید.",
        ]
    
    def load_data(self):
        """Load sampled products and full products for price lookup"""
        print("Loading sampled products...")
        with open(self.sampled_products_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'samples' in data:
            self.samples = data['samples']
        else:
            self.samples = data
        
        print(f"Loaded {len(self.samples)} sampled products")
        
        print("Loading full products for price lookup...")
        with open(self.full_products_file, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        if 'products' in products_data:
            products = products_data['products']
        else:
            products = products_data
        
        # Create lookup by product ID
        for product in products:
            self.products_lookup[product['id']] = product
        
        print(f"Loaded {len(self.products_lookup)} products for price lookup")
        return self.samples, self.products_lookup
    
    def format_price(self, price: Optional[float]) -> str:
        """Format price for display"""
        if price is None or price == 0:
            return "نامشخص"
        try:
            return f"{price:,.0f}"
        except (TypeError, ValueError):
            return "نامشخص"
    
    def clean_feature_value(self, value: str) -> str:
        """Clean feature value for display in dialogue"""
        if not value:
            return ""
        # Remove excessive newlines and whitespace
        value = value.replace('\r\n', ' ').replace('\n', ' ')
        # Collapse multiple spaces
        while '  ' in value:
            value = value.replace('  ', ' ')
        return value.strip()
    
    def generate_dialogue(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a dialogue for a sampled product
        
        Args:
            sample: Sampled product dictionary
            
        Returns:
            Dialogue dictionary or None if failed
        """
        # Get product info from sample
        product_info = sample.get('product', {})
        product_id = product_info.get('id')
        
        if not product_id:
            print(f"Warning: No product ID found in sample")
            return None
        
        # Get full product details for price
        full_product = self.products_lookup.get(product_id)
        if not full_product:
            print(f"Warning: Product {product_id} not found in full products")
            # Use sample data as fallback
            title = product_info.get('title', 'محصول')
            brand = product_info.get('brand', 'نامشخص')
            category = product_info.get('category_name', 'نامشخص')
            price = None
        else:
            title = full_product.get('title', product_info.get('title', 'محصول'))
            brand = full_product.get('brand', product_info.get('brand', 'نامشخص'))
            category = full_product.get('category_name', product_info.get('category_name', 'نامشخص'))
            price = full_product.get('price') or full_product.get('selling_price') or None
        
        # Get sampled features
        sampled_features = sample.get('sampled_features', {})
        feature_order = sample.get('feature_sampling_order', list(sampled_features.keys()))
        
        if not sampled_features:
            print(f"Warning: No sampled features for product {product_id}")
            return None
        
        # Build dialogue
        dialogue = []
        
        # Round 1: User searches by category
        user_search = random.choice(self.user_search_templates).format(category=category)
        dialogue.append({"role": "user", "text": user_search})
        
        # Round 2: Assistant asks about first feature
        if feature_order:
            first_feature_name = feature_order[0]
            first_feature_value = sampled_features.get(first_feature_name, "")
            assistant_response = random.choice(self.assistant_search_response_templates).format(
                category=category,
                feature=first_feature_name
            )
            dialogue.append({"role": "assistant", "text": assistant_response})
        else:
            # Fallback if no order specified
            first_feature_name = list(sampled_features.keys())[0]
            first_feature_value = sampled_features[first_feature_name]
            assistant_response = random.choice(self.assistant_search_response_templates).format(
                category=category,
                feature=first_feature_name
            )
            dialogue.append({"role": "assistant", "text": assistant_response})
        
        # Rounds 3+: User answers and assistant asks about next feature
        for idx, feature_name in enumerate(feature_order):
            feature_value = sampled_features.get(feature_name, "")
            cleaned_value = self.clean_feature_value(feature_value)
            
            if not cleaned_value:
                continue
            
            # User answers about current feature
            user_answer = random.choice(self.user_answer_templates).format(
                feature=feature_name,
                value=cleaned_value
            )
            dialogue.append({"role": "user", "text": user_answer})
            
            # Assistant responds
            if idx < len(feature_order) - 1:
                # More features to discuss
                next_feature_name = feature_order[idx + 1]
                assistant_follow_up = random.choice(self.assistant_follow_up_templates).format(
                    feature=feature_name,
                    value=cleaned_value,
                    next_feature=next_feature_name
                )
                dialogue.append({"role": "assistant", "text": assistant_follow_up})
            else:
                # Last feature - confirmation
                assistant_confirmation = random.choice(self.assistant_confirmation_templates).format(
                    feature=feature_name,
                    value=cleaned_value
                )
                dialogue.append({"role": "assistant", "text": assistant_confirmation})
        
        # Optional: User mentions brand (30% chance)
        if random.random() < 0.3:
            user_brand = random.choice(self.user_brand_templates).format(brand=brand)
            dialogue.append({"role": "user", "text": user_brand})
            assistant_brand_response = random.choice(self.assistant_brand_response_templates).format(brand=brand)
            dialogue.append({"role": "assistant", "text": assistant_brand_response})
        
        # Optional: User mentions price (30% chance)
        if price and random.random() < 0.3:
            user_price = random.choice(self.user_price_templates).format(price=self.format_price(price))
            dialogue.append({"role": "user", "text": user_price})
            assistant_price_response = random.choice(self.assistant_price_response_templates)
            dialogue.append({"role": "assistant", "text": assistant_price_response})
        
        # Closing: Assistant recommends product
        closing = random.choice(self.closing_templates).format(
            title=title,
            brand=brand,
            price=self.format_price(price)
        )
        dialogue.append({"role": "assistant", "text": closing})
        
        return {
            "product_id": product_id,
            "product_title": title,
            "brand": brand,
            "category": category,
            "price": price,
            "cluster_id": sample.get('cluster_id'),
            "dialogue_rounds": len(dialogue),
            "num_features_discussed": len(sampled_features),
            "features_discussed": [
                {
                    "feature_name": name,
                    "feature_value": self.clean_feature_value(value)
                }
                for name, value in sampled_features.items()
            ],
            "feature_sampling_order": feature_order,
            "dialogue": dialogue
        }
    
    def generate_batch(self, num_dialogues: int = None, output_dir: str = "template_dialogues_from_samples"):
        """Generate dialogues for sampled products"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Use all samples if num_dialogues not specified
        if num_dialogues is None:
            num_dialogues = len(self.samples)
        
        # Limit to available samples
        num_dialogues = min(num_dialogues, len(self.samples))
        
        print(f"\nGenerating {num_dialogues} dialogues from sampled products...")
        
        for idx, sample in enumerate(self.samples[:num_dialogues], 1):
            print(f"[{idx}/{num_dialogues}] Processing product {sample.get('product', {}).get('id')}...")
            
            dialogue = self.generate_dialogue(sample)
            
            if dialogue:
                self.dialogues.append(dialogue)
                print(f"  ✅ Generated dialogue with {dialogue['num_features_discussed']} features, {dialogue['dialogue_rounds']} rounds")
            else:
                print(f"  ❌ Failed to generate dialogue")
        
        # Save results
        output_data = {
            "metadata": {
                "total_dialogues": len(self.dialogues),
                "generation_date": datetime.now().isoformat(),
                "generation_method": "template_based_from_sampled_products",
                "source_file": self.sampled_products_file
            },
            "dialogues": self.dialogues
        }
        
        output_path = os.path.join(output_dir, "dialogues_template_from_samples.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Generated {len(self.dialogues)} dialogues")
        print(f"✅ Saved to {output_path}")
        
        # Print statistics
        if self.dialogues:
            avg_features = sum(d['num_features_discussed'] for d in self.dialogues) / len(self.dialogues)
            avg_rounds = sum(d['dialogue_rounds'] for d in self.dialogues) / len(self.dialogues)
            products_with_price = sum(1 for d in self.dialogues if d.get('price'))
            
            print(f"\n📊 Statistics:")
            print(f"  Average features per dialogue: {avg_features:.1f}")
            print(f"  Average dialogue rounds: {avg_rounds:.1f}")
            print(f"  Products with price: {products_with_price}/{len(self.dialogues)}")
        
        return output_path


def main():
    """Main function"""
    print("Template Dialogue Generator from Sampled Products")
    print("=" * 60)
    
    # Configuration
    sampled_products_file = "sampled_products/batch_500_products_5_features.json"
    full_products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    num_dialogues = 500  # Generate for all sampled products
    output_dir = "template_dialogues_from_samples"
    
    # Initialize generator
    generator = TemplateDialogueGenerator(
        sampled_products_file=sampled_products_file,
        full_products_file=full_products_file
    )
    
    # Load data
    generator.load_data()
    
    # Generate dialogues
    output_path = generator.generate_batch(num_dialogues=num_dialogues, output_dir=output_dir)
    
    print(f"\n🎉 Process completed! Results saved to: {output_path}")


if __name__ == "__main__":
    main()

