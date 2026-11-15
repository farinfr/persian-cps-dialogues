#!/usr/bin/env python3
"""
LLM-Enhanced Product Question Answering Dataset Generator
=========================================================

This script creates a Product Question Answering (PQA) dataset using a Large Language Model
to generate more natural, friendly, and dynamic questions. It uses the enhanced sampled
dataset with cluster analysis and generates questions that:
1. Identify specific products using natural language
2. Tag questions based on cluster performance status
3. Include cluster comparison information in answers
4. Use LLM to make questions more conversational and user-like

Tagging System:
- "good_quality_product": High rating, low price vs cluster
- "premium_product": High rating, high price vs cluster  
- "budget_product": Low rating, low price vs cluster
- "overpriced_product": Low rating, high price vs cluster
- "average_product": Average rating and price vs cluster
"""

import json
import random
import time
import os
from typing import Dict, List, Tuple, Any
from openai import OpenAI

class LLMEnhancedPQAGenerator:
    def __init__(self, enhanced_sample_path: str, api_key: str = None, base_url: str = None):
        self.enhanced_sample_path = enhanced_sample_path
        self.sample_data = None
        self.pqa_dataset = []
        
        # Initialize LLM client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # LLM configuration
        self.model = "qwen-turbo"
        self.max_retries = 3
        self.delay_between_requests = 2.0
        
    def load_enhanced_sample(self):
        """Load the enhanced sample dataset"""
        print("Loading enhanced sample dataset...")
        with open(self.enhanced_sample_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.sample_data = data['products']
        print(f"Loaded {len(self.sample_data)} enhanced products")
        return self.sample_data
    
    def determine_product_tag(self, product: Dict) -> str:
        """Determine the tag based on cluster performance"""
        rating_vs_cluster = product.get('rating_average_vs_cluster', 'unknown')
        price_vs_cluster = product.get('price_vs_cluster', 'unknown')
        
        # Define tagging logic based on cluster performance
        if rating_vs_cluster == 'high' and price_vs_cluster == 'low':
            return "good_quality_product"  # High rating, low price = excellent value
        elif rating_vs_cluster == 'high' and price_vs_cluster == 'high':
            return "premium_product"  # High rating, high price = premium
        elif rating_vs_cluster == 'high' and price_vs_cluster == 'average':
            return "good_quality_product"  # High rating, average price = good quality
        elif rating_vs_cluster == 'average' and price_vs_cluster == 'low':
            return "budget_product"  # Average rating, low price = budget
        elif rating_vs_cluster == 'low' and price_vs_cluster == 'low':
            return "budget_product"  # Low rating, low price = budget
        elif rating_vs_cluster == 'low' and price_vs_cluster == 'high':
            return "overpriced_product"  # Low rating, high price = overpriced
        elif rating_vs_cluster == 'average' and price_vs_cluster == 'average':
            return "average_product"  # Average rating, average price = average
        elif rating_vs_cluster == 'average' and price_vs_cluster == 'high':
            return "premium_product"  # Average rating, high price = premium
        else:
            return "average_product"  # Default for unknown combinations
    
    def get_tag_description(self, tag: str) -> str:
        """Get human-readable description for the tag"""
        descriptions = {
            "good_quality_product": "محصول با کیفیت خوب و قیمت مناسب نسبت به هم‌رده‌هایش",
            "premium_product": "محصول پریمیوم با قیمت بالا و کیفیت خوب",
            "budget_product": "محصول اقتصادی با قیمت پایین",
            "overpriced_product": "محصول گران‌قیمت نسبت به کیفیتش",
            "average_product": "محصول متوسط از نظر کیفیت و قیمت"
        }
        return descriptions.get(tag, "محصول با مشخصات متوسط")
    
    def generate_llm_question(self, product: Dict, question_type: str) -> str:
        """Generate a natural question using LLM"""
        
        # Extract product information
        title = product.get('title', '')
        category = product.get('category_name', '')
        brand = product.get('brand', '')
        price = product.get('selling_price', 0) or 0
        rating = product.get('rating_average', 0) or 0
        rating_count = product.get('rating_count', 0) or 0
        
        # Format price for display
        if price > 0:
            if price >= 1_000_000_000:
                price_text = f"{price/1_000_000_000:.1f} میلیارد ریال"
            elif price >= 1_000_000:
                price_text = f"{price/1_000_000:.0f} میلیون ریال"
            else:
                price_text = f"{price:,.0f} ریال"
        else:
            price_text = "قیمت نامشخص"
        
        # Create prompt based on question type
        prompts = {
            "general": f"""
            برای محصول زیر یک سوال طبیعی و دوستانه به زبان فارسی تولید کن که کاربران واقعی ممکن است بپرسند:
            
            محصول: {title}
            برند: {brand}
            دسته‌بندی: {category}
            قیمت: {price_text}
            امتیاز: {rating:.1f} از 100
            تعداد نظرات: {rating_count}
            
            سوال باید:
            - طبیعی و دوستانه باشد
            - مثل سوالات کاربران واقعی باشد
            - شامل نام محصول باشد
            - کوتاه و قابل فهم باشد
            - فقط سوال باشد، بدون توضیح اضافی
            
            مثال: "این گوشی سامسونگ چطوره؟" یا "آیا این لپ تاپ ایسوس ارزش خرید داره؟"
            """,
            
            "price_focused": f"""
            برای محصول زیر یک سوال طبیعی و دوستانه به زبان فارسی تولید کن که روی قیمت تمرکز دارد:
            
            محصول: {title}
            برند: {brand}
            دسته‌بندی: {category}
            قیمت: {price_text}
            امتیاز: {rating:.1f} از 100
            
            سوال باید:
            - روی قیمت یا ارزش محصول تمرکز کند
            - طبیعی و دوستانه باشد
            - شامل نام محصول باشد
            - کوتاه باشد
            
            مثال: "این قیمت برای این گوشی مناسبه؟" یا "آیا این لپ تاپ ارزش این قیمت رو داره؟"
            """,
            
            "quality_focused": f"""
            برای محصول زیر یک سوال طبیعی و دوستانه به زبان فارسی تولید کن که روی کیفیت تمرکز دارد:
            
            محصول: {title}
            برند: {brand}
            دسته‌بندی: {category}
            امتیاز: {rating:.1f} از 100
            تعداد نظرات: {rating_count}
            
            سوال باید:
            - روی کیفیت یا عملکرد محصول تمرکز کند
            - طبیعی و دوستانه باشد
            - شامل نام محصول باشد
            - کوتاه باشد
            
            مثال: "این گوشی کیفیتش چطوره؟" یا "آیا این لپ تاپ برای کارهای سنگین مناسبه؟"
            """,
            
            "comparison": f"""
            برای محصول زیر یک سوال طبیعی و دوستانه به زبان فارسی تولید کن که مقایسه‌ای باشد:
            
            محصول: {title}
            برند: {brand}
            دسته‌بندی: {category}
            قیمت: {price_text}
            امتیاز: {rating:.1f} از 100
            
            سوال باید:
            - مقایسه‌ای یا انتخابی باشد
            - طبیعی و دوستانه باشد
            - شامل نام محصول باشد
            - کوتاه باشد
            
            مثال: "این گوشی بهتره یا آیفون؟" یا "آیا این لپ تاپ از همون برند دیگه بهتره؟"
            """
        }
        
        prompt = prompts.get(question_type, prompts["general"])
        
        try:
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "شما یک متخصص تولید سوالات طبیعی و دوستانه برای محصولات هستید. فقط سوال تولید کنید، بدون توضیح اضافی."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=100,
                        temperature=0.8
                    )
                    
                    question = response.choices[0].message.content.strip()
                    
                    # Clean up the question
                    question = question.replace('"', '').replace("'", '').strip()
                    
                    # Ensure it's a question
                    if not question.endswith('؟') and not question.endswith('?'):
                        question += '؟'
                    
                    return question
                    
                except Exception as e:
                    print(f"LLM error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.delay_between_requests)
                    else:
                        # Fallback to template-based question
                        return self.generate_fallback_question(product, question_type)
        
        except Exception as e:
            print(f"LLM request failed: {e}")
            return self.generate_fallback_question(product, question_type)
    
    def generate_fallback_question(self, product: Dict, question_type: str) -> str:
        """Generate fallback question if LLM fails"""
        title = product.get('title', '')
        brand = product.get('brand', '')
        category = product.get('category_name', '')
        
        fallback_templates = {
            "general": [
                f"در مورد محصول {title} چه می‌دانید؟",
                f"محصول {title} چطوره؟",
                f"آیا محصول {title} خوبه؟"
            ],
            "price_focused": [
                f"قیمت محصول {title} مناسبه؟",
                f"آیا محصول {title} ارزش قیمتش رو داره؟"
            ],
            "quality_focused": [
                f"کیفیت محصول {title} چطوره؟",
                f"محصول {title} خوب کار می‌کنه؟"
            ],
            "comparison": [
                f"محصول {title} بهتره یا محصولات مشابه؟",
                f"آیا محصول {title} از برندهای دیگه بهتره؟"
            ]
        }
        
        templates = fallback_templates.get(question_type, fallback_templates["general"])
        return random.choice(templates)
    
    def create_answer_with_cluster_info(self, product: Dict, tag: str) -> str:
        """Create a comprehensive answer including cluster comparison information"""
        title = product.get('title', '')
        category = product.get('category_name', '')
        brand = product.get('brand', '')
        price = product.get('selling_price', 0) or 0
        rating = product.get('rating_average', 0) or 0
        rating_count = product.get('rating_count', 0) or 0
        
        # Format price
        if price > 0:
            if price >= 1_000_000_000:
                price_text = f"{price/1_000_000_000:.1f} میلیارد ریال"
            elif price >= 1_000_000:
                price_text = f"{price/1_000_000:.0f} میلیون ریال"
            else:
                price_text = f"{price:,.0f} ریال"
        else:
            price_text = "قیمت نامشخص"
        
        # Cluster comparison information
        cluster_id = product.get('cluster_id', 'نامشخص')
        cluster_avg_rating = product.get('cluster_avg_rating_average', 0) or 0
        cluster_avg_price = product.get('cluster_avg_price', 0) or 0
        rating_diff_percent = product.get('rating_average_diff_percent', 0) or 0
        price_diff_percent = product.get('price_diff_percent', 0) or 0
        
        # Format cluster price
        if cluster_avg_price > 0:
            if cluster_avg_price >= 1_000_000_000:
                cluster_price_text = f"{cluster_avg_price/1_000_000_000:.1f} میلیارد ریال"
            elif cluster_avg_price >= 1_000_000:
                cluster_price_text = f"{cluster_avg_price/1_000_000:.0f} میلیون ریال"
            else:
                cluster_price_text = f"{cluster_avg_price:,.0f} ریال"
        else:
            cluster_price_text = "قیمت نامشخص"
        
        # Build answer
        answer_parts = [
            f"محصول «{title}» از برند {brand} در دسته‌بندی {category} است.",
            f"قیمت این محصول {price_text} و امتیاز آن {rating:.1f} از 100 است.",
            f"این محصول {rating_count} نظر دریافت کرده است."
        ]
        
        # Add cluster comparison information
        if cluster_id != 'نامشخص':
            answer_parts.append(f"این محصول در خوشه {cluster_id} قرار دارد.")
            
            if rating_diff_percent != 0:
                if rating_diff_percent > 0:
                    answer_parts.append(f"امتیاز این محصول {rating_diff_percent:+.1f}% بالاتر از میانگین خوشه ({cluster_avg_rating:.1f}) است.")
                else:
                    answer_parts.append(f"امتیاز این محصول {rating_diff_percent:+.1f}% پایین‌تر از میانگین خوشه ({cluster_avg_rating:.1f}) است.")
            
            if price_diff_percent != 0:
                if price_diff_percent > 0:
                    answer_parts.append(f"قیمت این محصول {price_diff_percent:+.1f}% بالاتر از میانگین خوشه ({cluster_price_text}) است.")
                else:
                    answer_parts.append(f"قیمت این محصول {price_diff_percent:+.1f}% پایین‌تر از میانگین خوشه ({cluster_price_text}) است.")
        
        # Add tag-based assessment
        tag_description = self.get_tag_description(tag)
        answer_parts.append(f"این محصول {tag_description} محسوب می‌شود.")
        
        return " ".join(answer_parts)
    
    def generate_llm_pqa_dataset(self, num_questions_per_product: int = 3, sample_size: int = 100) -> List[Dict]:
        """Generate the complete PQA dataset using LLM"""
        print(f"Generating LLM PQA dataset with {num_questions_per_product} questions per product...")
        print(f"Processing {sample_size} products (to manage API costs)...")
        
        # Sample products to manage API costs
        sampled_products = random.sample(self.sample_data, min(sample_size, len(self.sample_data)))
        
        self.pqa_dataset = []
        question_types = ["general", "price_focused", "quality_focused", "comparison"]
        
        for i, product in enumerate(sampled_products):
            print(f"Processing product {i+1}/{len(sampled_products)}: {product.get('title', 'Unknown')[:50]}...")
            
            # Determine product tag
            tag = self.determine_product_tag(product)
            
            # Generate questions for this product
            questions = []
            for j in range(num_questions_per_product):
                question_type = question_types[j % len(question_types)]
                question = self.generate_llm_question(product, question_type)
                questions.append(question)
                
                # Add delay between requests
                if j < num_questions_per_product - 1:
                    time.sleep(self.delay_between_requests)
            
            # Create PQA entries
            for question in questions:
                answer = self.create_answer_with_cluster_info(product, tag)
                
                pqa_entry = {
                    "question": question,
                    "answer": answer,
                    "product_id": product.get('id'),
                    "product_title": product.get('title', ''),
                    "category": product.get('category_name', ''),
                    "brand": product.get('brand', ''),
                    "price": product.get('selling_price', 0),
                    "rating": product.get('rating_average', 0),
                    "rating_count": product.get('rating_count', 0),
                    "cluster_id": product.get('cluster_id'),
                    "cluster_avg_rating": product.get('cluster_avg_rating_average', 0),
                    "cluster_avg_price": product.get('cluster_avg_price', 0),
                    "rating_vs_cluster": product.get('rating_average_vs_cluster', 'unknown'),
                    "price_vs_cluster": product.get('price_vs_cluster', 'unknown'),
                    "rating_diff_percent": product.get('rating_average_diff_percent', 0),
                    "price_diff_percent": product.get('price_diff_percent', 0),
                    "tag": tag,
                    "tag_description": self.get_tag_description(tag),
                    "generated_by": "llm"
                }
                
                self.pqa_dataset.append(pqa_entry)
            
            # Add delay between products
            if i < len(sampled_products) - 1:
                time.sleep(self.delay_between_requests)
        
        print(f"Generated {len(self.pqa_dataset)} LLM PQA entries")
        return self.pqa_dataset
    
    def analyze_dataset_statistics(self) -> Dict:
        """Analyze the generated dataset statistics"""
        print("Analyzing LLM dataset statistics...")
        
        # Tag distribution
        tag_counts = {}
        for entry in self.pqa_dataset:
            tag = entry['tag']
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Category distribution
        category_counts = {}
        for entry in self.pqa_dataset:
            category = entry['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Brand distribution
        brand_counts = {}
        for entry in self.pqa_dataset:
            brand = entry['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        # Question length analysis
        question_lengths = [len(entry['question']) for entry in self.pqa_dataset]
        
        stats = {
            'total_questions': len(self.pqa_dataset),
            'total_products': len(set(entry['product_id'] for entry in self.pqa_dataset)),
            'total_categories': len(category_counts),
            'total_brands': len(brand_counts),
            'tag_distribution': tag_counts,
            'category_distribution': category_counts,
            'brand_distribution': dict(sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'avg_question_length': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'min_question_length': min(question_lengths) if question_lengths else 0,
            'max_question_length': max(question_lengths) if question_lengths else 0,
            'generated_by': 'llm'
        }
        
        return stats
    
    def save_llm_pqa_dataset(self, output_dir: str = "llm_enhanced_pqa_dataset"):
        """Save the LLM PQA dataset and analysis"""
        print("Saving LLM PQA dataset...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete dataset
        dataset_file = f'{output_dir}/llm_enhanced_pqa_dataset.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_questions': len(self.pqa_dataset),
                    'total_products': len(set(entry['product_id'] for entry in self.pqa_dataset)),
                    'generation_date': '2025-10-26',
                    'includes_cluster_analysis': True,
                    'tagging_system': 'cluster_performance_based',
                    'generated_by': 'llm',
                    'llm_model': self.model,
                    'api_provider': 'avalai'
                },
                'questions': self.pqa_dataset
            }, f, ensure_ascii=False, indent=2)
        
        # Save simplified dataset (just questions and answers)
        simplified_file = f'{output_dir}/llm_enhanced_pqa_dataset_simplified.json'
        simplified_data = []
        for entry in self.pqa_dataset:
            simplified_data.append({
                'question': entry['question'],
                'answer': entry['answer'],
                'tag': entry['tag'],
                'product_title': entry['product_title']
            })
        
        with open(simplified_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, ensure_ascii=False, indent=2)
        
        # Analyze and save statistics
        stats = self.analyze_dataset_statistics()
        stats_file = f'{output_dir}/llm_enhanced_pqa_dataset_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Create sample questions file
        sample_file = f'{output_dir}/llm_sample_questions.txt'
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("LLM-ENHANCED PQA DATASET - SAMPLE QUESTIONS\n")
            f.write("=" * 60 + "\n\n")
            
            # Show examples for each tag
            tag_examples = {}
            for entry in self.pqa_dataset:
                tag = entry['tag']
                if tag not in tag_examples:
                    tag_examples[tag] = entry
            
            for tag, example in tag_examples.items():
                f.write(f"TAG: {tag.upper()}\n")
                f.write(f"Description: {example['tag_description']}\n")
                f.write(f"Question: {example['question']}\n")
                f.write(f"Answer: {example['answer']}\n")
                f.write(f"Product: {example['product_title']}\n")
                f.write(f"Cluster Performance: Rating {example['rating_vs_cluster']}, Price {example['price_vs_cluster']}\n")
                f.write("-" * 60 + "\n\n")
        
        # Create summary report
        summary_file = f'{output_dir}/llm_enhanced_pqa_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("LLM-ENHANCED PQA DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Questions: {stats['total_questions']}\n")
            f.write(f"Total Products: {stats['total_products']}\n")
            f.write(f"Total Categories: {stats['total_categories']}\n")
            f.write(f"Total Brands: {stats['total_brands']}\n")
            f.write(f"Generated by: {stats['generated_by'].upper()}\n")
            f.write(f"LLM Model: {self.model}\n")
            f.write(f"Average Question Length: {stats['avg_question_length']:.1f} characters\n\n")
            
            f.write("TAG DISTRIBUTION:\n")
            for tag, count in stats['tag_distribution'].items():
                percentage = (count / stats['total_questions']) * 100
                f.write(f"  {tag}: {count} questions ({percentage:.1f}%)\n")
            
            f.write("\nCATEGORY DISTRIBUTION:\n")
            for category, count in stats['category_distribution'].items():
                percentage = (count / stats['total_questions']) * 100
                f.write(f"  {category}: {count} questions ({percentage:.1f}%)\n")
            
            f.write("\nTOP 10 BRANDS:\n")
            for brand, count in list(stats['brand_distribution'].items())[:10]:
                percentage = (count / stats['total_questions']) * 100
                f.write(f"  {brand}: {count} questions ({percentage:.1f}%)\n")
        
        print(f"LLM PQA dataset saved to {output_dir}/")
        return stats

def main():
    """Main function to generate the LLM-enhanced PQA dataset"""
    print("Starting LLM-Enhanced PQA Dataset Generation...")
    print("=" * 70)
    
    # Configuration
    enhanced_sample_path = "enhanced_sample_analysis_results/enhanced_sample_products.json"
    output_dir = "llm_enhanced_pqa_dataset"
    questions_per_product = 3
    sample_size = 100  # Process 100 products to manage API costs
    
    # Initialize generator
    generator = LLMEnhancedPQAGenerator(enhanced_sample_path)
    
    try:
        # Load enhanced sample
        generator.load_enhanced_sample()
        
        # Generate LLM PQA dataset
        generator.generate_llm_pqa_dataset(questions_per_product, sample_size)
        
        # Save dataset and analysis
        stats = generator.save_llm_pqa_dataset(output_dir)
        
        print("\n" + "=" * 70)
        print("LLM-Enhanced PQA Dataset Generation Complete!")
        print("=" * 70)
        
        # Print summary
        print(f"\n📊 LLM DATASET SUMMARY:")
        print(f"  Total Questions: {stats['total_questions']}")
        print(f"  Total Products: {stats['total_products']}")
        print(f"  Total Categories: {stats['total_categories']}")
        print(f"  Total Brands: {stats['total_brands']}")
        print(f"  Generated by: {stats['generated_by'].upper()}")
        print(f"  LLM Model: {generator.model}")
        print(f"  Average Question Length: {stats['avg_question_length']:.1f} characters")
        
        print(f"\n🏷️ TAG DISTRIBUTION:")
        for tag, count in stats['tag_distribution'].items():
            percentage = (count / stats['total_questions']) * 100
            print(f"  {tag}: {count} questions ({percentage:.1f}%)")
        
        print(f"\n📁 Files created:")
        print(f"  📄 Complete dataset: {output_dir}/llm_enhanced_pqa_dataset.json")
        print(f"  📄 Simplified dataset: {output_dir}/llm_enhanced_pqa_dataset_simplified.json")
        print(f"  📊 Statistics: {output_dir}/llm_enhanced_pqa_dataset_statistics.json")
        print(f"  📝 Sample questions: {output_dir}/llm_sample_questions.txt")
        print(f"  📋 Summary report: {output_dir}/llm_enhanced_pqa_summary.txt")
        
    except Exception as e:
        print(f"Error during LLM PQA generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
