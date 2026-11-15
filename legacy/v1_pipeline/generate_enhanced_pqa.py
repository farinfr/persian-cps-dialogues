#!/usr/bin/env python3
"""
Enhanced Product Question Answering Dataset Generator
====================================================

This script creates a Product Question Answering (PQA) dataset using the enhanced
sampled dataset with cluster analysis. It generates questions that:
1. Identify specific products using title and properties
2. Tag questions based on cluster performance status
3. Include cluster comparison information in answers

Tagging System:
- "good_quality_product": High rating, low price vs cluster
- "premium_product": High rating, high price vs cluster  
- "budget_product": Low rating, low price vs cluster
- "overpriced_product": Low rating, high price vs cluster
- "average_product": Average rating and price vs cluster
"""

import json
import random
from typing import Dict, List, Tuple, Any
import os

class EnhancedPQAGenerator:
    def __init__(self, enhanced_sample_path: str):
        self.enhanced_sample_path = enhanced_sample_path
        self.sample_data = None
        self.pqa_dataset = []
        
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
    
    def generate_product_identification_questions(self, product: Dict) -> List[Dict]:
        """Generate questions that identify a specific product"""
        questions = []
        
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
        
        # Question templates for product identification
        question_templates = [
            # Title-based questions
            f"محصولی با عنوان «{title}» را معرفی کنید",
            f"در مورد محصول «{title}» چه می‌دانید؟",
            f"محصول «{title}» چه مشخصاتی دارد؟",
            
            # Category-based questions
            f"محصولی در دسته‌بندی {category} با عنوان «{title}» را معرفی کنید",
            f"در دسته‌بندی {category}، محصول «{title}» چه ویژگی‌هایی دارد؟",
            
            # Brand-based questions
            f"محصولی از برند {brand} با عنوان «{title}» را معرفی کنید",
            f"محصول «{title}» از برند {brand} چه مشخصاتی دارد؟",
            
            # Price-based questions
            f"محصولی با قیمت {price_text} و عنوان «{title}» را معرفی کنید",
            f"محصول «{title}» با قیمت {price_text} چه ویژگی‌هایی دارد؟",
            
            # Rating-based questions
            f"محصولی با امتیاز {rating:.1f} و عنوان «{title}» را معرفی کنید",
            f"محصول «{title}» با امتیاز {rating:.1f} چه مشخصاتی دارد؟",
            
            # Combined questions
            f"محصولی از برند {brand} در دسته‌بندی {category} با عنوان «{title}» را معرفی کنید",
            f"محصول «{title}» از برند {brand} با قیمت {price_text} چه ویژگی‌هایی دارد؟",
            f"محصولی با امتیاز {rating:.1f} و قیمت {price_text} و عنوان «{title}» را معرفی کنید",
        ]
        
        # Generate questions
        for question_text in question_templates:
            if question_text and len(question_text.strip()) > 10:  # Ensure meaningful questions
                questions.append({
                    "question": question_text,
                    "product_id": product.get('id'),
                    "product_title": title,
                    "category": category,
                    "brand": brand,
                    "price": price,
                    "rating": rating,
                    "rating_count": rating_count
                })
        
        return questions
    
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
    
    def generate_pqa_dataset(self, num_questions_per_product: int = 3) -> List[Dict]:
        """Generate the complete PQA dataset"""
        print(f"Generating PQA dataset with {num_questions_per_product} questions per product...")
        
        self.pqa_dataset = []
        
        for product in self.sample_data:
            # Determine product tag
            tag = self.determine_product_tag(product)
            
            # Generate questions for this product
            questions = self.generate_product_identification_questions(product)
            
            # Select random questions (up to num_questions_per_product)
            selected_questions = random.sample(
                questions, 
                min(num_questions_per_product, len(questions))
            )
            
            # Create PQA entries
            for question_data in selected_questions:
                answer = self.create_answer_with_cluster_info(product, tag)
                
                pqa_entry = {
                    "question": question_data["question"],
                    "answer": answer,
                    "product_id": question_data["product_id"],
                    "product_title": question_data["product_title"],
                    "category": question_data["category"],
                    "brand": question_data["brand"],
                    "price": question_data["price"],
                    "rating": question_data["rating"],
                    "rating_count": question_data["rating_count"],
                    "cluster_id": product.get('cluster_id'),
                    "cluster_avg_rating": product.get('cluster_avg_rating_average', 0),
                    "cluster_avg_price": product.get('cluster_avg_price', 0),
                    "rating_vs_cluster": product.get('rating_average_vs_cluster', 'unknown'),
                    "price_vs_cluster": product.get('price_vs_cluster', 'unknown'),
                    "rating_diff_percent": product.get('rating_average_diff_percent', 0),
                    "price_diff_percent": product.get('price_diff_percent', 0),
                    "tag": tag,
                    "tag_description": self.get_tag_description(tag)
                }
                
                self.pqa_dataset.append(pqa_entry)
        
        print(f"Generated {len(self.pqa_dataset)} PQA entries")
        return self.pqa_dataset
    
    def analyze_dataset_statistics(self) -> Dict:
        """Analyze the generated dataset statistics"""
        print("Analyzing dataset statistics...")
        
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
        
        # Cluster distribution
        cluster_counts = {}
        for entry in self.pqa_dataset:
            cluster_id = entry['cluster_id']
            if cluster_id is not None:
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        # Price range analysis
        prices = [entry['price'] for entry in self.pqa_dataset if entry['price'] > 0]
        price_ranges = {
            'under_1m': len([p for p in prices if p < 1_000_000]),
            '1m_to_10m': len([p for p in prices if 1_000_000 <= p < 10_000_000]),
            '10m_to_100m': len([p for p in prices if 10_000_000 <= p < 100_000_000]),
            '100m_to_1b': len([p for p in prices if 100_000_000 <= p < 1_000_000_000]),
            'over_1b': len([p for p in prices if p >= 1_000_000_000])
        }
        
        # Rating range analysis
        ratings = [entry['rating'] for entry in self.pqa_dataset if entry['rating'] > 0]
        rating_ranges = {
            'excellent_90_100': len([r for r in ratings if r >= 90]),
            'very_good_80_89': len([r for r in ratings if 80 <= r < 90]),
            'good_70_79': len([r for r in ratings if 70 <= r < 80]),
            'average_60_69': len([r for r in ratings if 60 <= r < 70]),
            'below_average_0_59': len([r for r in ratings if r < 60])
        }
        
        stats = {
            'total_questions': len(self.pqa_dataset),
            'total_products': len(set(entry['product_id'] for entry in self.pqa_dataset)),
            'total_categories': len(category_counts),
            'total_brands': len(brand_counts),
            'total_clusters': len(cluster_counts),
            'tag_distribution': tag_counts,
            'category_distribution': category_counts,
            'brand_distribution': dict(sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'cluster_distribution': dict(sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            'price_ranges': price_ranges,
            'rating_ranges': rating_ranges,
            'avg_price': sum(prices) / len(prices) if prices else 0,
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0
        }
        
        return stats
    
    def save_pqa_dataset(self, output_dir: str = "enhanced_pqa_dataset"):
        """Save the PQA dataset and analysis"""
        print("Saving PQA dataset...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete dataset
        dataset_file = f'{output_dir}/enhanced_pqa_dataset.json'
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_questions': len(self.pqa_dataset),
                    'total_products': len(set(entry['product_id'] for entry in self.pqa_dataset)),
                    'generation_date': '2025-10-26',
                    'includes_cluster_analysis': True,
                    'tagging_system': 'cluster_performance_based'
                },
                'questions': self.pqa_dataset
            }, f, ensure_ascii=False, indent=2)
        
        # Save simplified dataset (just questions and answers)
        simplified_file = f'{output_dir}/enhanced_pqa_dataset_simplified.json'
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
        stats_file = f'{output_dir}/enhanced_pqa_dataset_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # Create sample questions file
        sample_file = f'{output_dir}/enhanced_sample_questions.txt'
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED PQA DATASET - SAMPLE QUESTIONS\n")
            f.write("=" * 50 + "\n\n")
            
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
                f.write("-" * 50 + "\n\n")
        
        # Create summary report
        summary_file = f'{output_dir}/enhanced_pqa_summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED PQA DATASET SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Total Questions: {stats['total_questions']}\n")
            f.write(f"Total Products: {stats['total_products']}\n")
            f.write(f"Total Categories: {stats['total_categories']}\n")
            f.write(f"Total Brands: {stats['total_brands']}\n")
            f.write(f"Total Clusters: {stats['total_clusters']}\n\n")
            
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
        
        print(f"PQA dataset saved to {output_dir}/")
        return stats

def main():
    """Main function to generate the enhanced PQA dataset"""
    print("Starting Enhanced PQA Dataset Generation...")
    print("=" * 60)
    
    # Configuration
    enhanced_sample_path = "enhanced_sample_analysis_results/enhanced_sample_products.json"
    output_dir = "enhanced_pqa_dataset"
    questions_per_product = 3
    
    # Initialize generator
    generator = EnhancedPQAGenerator(enhanced_sample_path)
    
    try:
        # Load enhanced sample
        generator.load_enhanced_sample()
        
        # Generate PQA dataset
        generator.generate_pqa_dataset(questions_per_product)
        
        # Save dataset and analysis
        stats = generator.save_pqa_dataset(output_dir)
        
        print("\n" + "=" * 60)
        print("Enhanced PQA Dataset Generation Complete!")
        print("=" * 60)
        
        # Print summary
        print(f"\n📊 DATASET SUMMARY:")
        print(f"  Total Questions: {stats['total_questions']}")
        print(f"  Total Products: {stats['total_products']}")
        print(f"  Total Categories: {stats['total_categories']}")
        print(f"  Total Brands: {stats['total_brands']}")
        print(f"  Total Clusters: {stats['total_clusters']}")
        
        print(f"\n🏷️ TAG DISTRIBUTION:")
        for tag, count in stats['tag_distribution'].items():
            percentage = (count / stats['total_questions']) * 100
            print(f"  {tag}: {count} questions ({percentage:.1f}%)")
        
        print(f"\n📁 Files created:")
        print(f"  📄 Complete dataset: {output_dir}/enhanced_pqa_dataset.json")
        print(f"  📄 Simplified dataset: {output_dir}/enhanced_pqa_dataset_simplified.json")
        print(f"  📊 Statistics: {output_dir}/enhanced_pqa_dataset_statistics.json")
        print(f"  📝 Sample questions: {output_dir}/enhanced_sample_questions.txt")
        print(f"  📋 Summary report: {output_dir}/enhanced_pqa_summary.txt")
        
    except Exception as e:
        print(f"Error during PQA generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
