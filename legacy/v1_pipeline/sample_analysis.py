#!/usr/bin/env python3
"""
Product Dataset Sampling and Analysis
=====================================

This script samples 4000 products from the complete dataset and performs
comprehensive analysis including:
- Basic statistics
- Category distribution
- Price analysis
- Rating analysis
- Brand analysis
- Specification analysis
- Cluster analysis (if available)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import random
from typing import Dict, List, Any
import os

class ProductSampleAnalyzer:
    def __init__(self, dataset_path: str, sample_size: int = 4000):
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.sample_data = None
        self.analysis_results = {}
        
    def load_dataset(self):
        """Load the complete dataset"""
        print("Loading complete dataset...")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.complete_data = json.load(f)
        
        print(f"Loaded {len(self.complete_data['products'])} products")
        return self.complete_data
    
    def sample_products(self):
        """Sample products from the dataset"""
        print(f"Sampling {self.sample_size} products...")
        
        products = self.complete_data['products']
        
        # Ensure we don't sample more than available
        actual_sample_size = min(self.sample_size, len(products))
        
        # Random sampling
        sampled_indices = random.sample(range(len(products)), actual_sample_size)
        self.sample_data = [products[i] for i in sampled_indices]
        
        print(f"Sampled {len(self.sample_data)} products")
        return self.sample_data
    
    def basic_statistics(self):
        """Calculate basic statistics"""
        print("Calculating basic statistics...")
        
        stats = {
            'total_products': len(self.sample_data),
            'categories': len(set(p.get('category_name', 'Unknown') for p in self.sample_data)),
            'brands': len(set(p.get('brand', 'Unknown') for p in self.sample_data)),
            'products_with_descriptions': sum(1 for p in self.sample_data if p.get('description') and p.get('description', '').strip()),
            'products_with_specifications': sum(1 for p in self.sample_data if p.get('specifications')),
            'products_with_images': sum(1 for p in self.sample_data if p.get('main_image_url')),
        }
        
        # Price statistics
        prices = [p.get('selling_price', 0) or 0 for p in self.sample_data]
        prices = [p for p in prices if p > 0]  # Remove zero prices
        
        if prices:
            stats['price_stats'] = {
                'mean': np.mean(prices),
                'median': np.median(prices),
                'min': np.min(prices),
                'max': np.max(prices),
                'std': np.std(prices),
                'count': len(prices)
            }
        
        # Rating statistics
        ratings = [p.get('rating_average', 0) or 0 for p in self.sample_data]
        ratings = [r for r in ratings if r > 0]  # Remove zero ratings
        
        if ratings:
            stats['rating_stats'] = {
                'mean': np.mean(ratings),
                'median': np.median(ratings),
                'min': np.min(ratings),
                'max': np.max(ratings),
                'std': np.std(ratings),
                'count': len(ratings)
            }
        
        # Rating count statistics
        rating_counts = [p.get('rating_count', 0) or 0 for p in self.sample_data]
        rating_counts = [rc for rc in rating_counts if rc > 0]  # Remove zero counts
        
        if rating_counts:
            stats['rating_count_stats'] = {
                'mean': np.mean(rating_counts),
                'median': np.median(rating_counts),
                'min': np.min(rating_counts),
                'max': np.max(rating_counts),
                'std': np.std(rating_counts),
                'count': len(rating_counts)
            }
        
        self.analysis_results['basic_statistics'] = stats
        return stats
    
    def category_analysis(self):
        """Analyze category distribution"""
        print("Analyzing categories...")
        
        categories = [p.get('category_name', 'Unknown') for p in self.sample_data]
        category_counts = Counter(categories)
        
        category_stats = {}
        for category, count in category_counts.items():
            category_products = [p for p in self.sample_data if p.get('category_name') == category]
            
            # Price analysis for this category
            prices = [p.get('selling_price', 0) or 0 for p in category_products if p.get('selling_price', 0)]
            prices = [p for p in prices if p > 0]
            
            # Rating analysis for this category
            ratings = [p.get('rating_average', 0) or 0 for p in category_products if p.get('rating_average', 0)]
            ratings = [r for r in ratings if r > 0]
            
            category_stats[category] = {
                'count': count,
                'percentage': (count / len(self.sample_data)) * 100,
                'avg_price': np.mean(prices) if prices else 0,
                'avg_rating': np.mean(ratings) if ratings else 0,
                'brands': len(set(p.get('brand', 'Unknown') for p in category_products))
            }
        
        self.analysis_results['category_analysis'] = category_stats
        return category_stats
    
    def brand_analysis(self):
        """Analyze brand distribution"""
        print("Analyzing brands...")
        
        brands = [p.get('brand', 'Unknown') for p in self.sample_data]
        brand_counts = Counter(brands)
        
        # Top brands analysis
        top_brands = brand_counts.most_common(20)
        
        brand_stats = {}
        for brand, count in top_brands:
            brand_products = [p for p in self.sample_data if p.get('brand') == brand]
            
            # Price analysis
            prices = [p.get('selling_price', 0) or 0 for p in brand_products if p.get('selling_price', 0)]
            prices = [p for p in prices if p > 0]
            
            # Rating analysis
            ratings = [p.get('rating_average', 0) or 0 for p in brand_products if p.get('rating_average', 0)]
            ratings = [r for r in ratings if r > 0]
            
            # Categories for this brand
            categories = set(p.get('category_name', 'Unknown') for p in brand_products)
            
            brand_stats[brand] = {
                'count': count,
                'percentage': (count / len(self.sample_data)) * 100,
                'avg_price': np.mean(prices) if prices else 0,
                'avg_rating': np.mean(ratings) if ratings else 0,
                'categories': list(categories),
                'category_count': len(categories)
            }
        
        self.analysis_results['brand_analysis'] = brand_stats
        return brand_stats
    
    def price_analysis(self):
        """Detailed price analysis"""
        print("Analyzing prices...")
        
        prices = [p.get('selling_price', 0) or 0 for p in self.sample_data]
        prices = [p for p in prices if p > 0]  # Remove zero prices
        
        if not prices:
            return {}
        
        # Price ranges
        price_ranges = {
            'under_1m': len([p for p in prices if p < 1_000_000]),
            '1m_to_10m': len([p for p in prices if 1_000_000 <= p < 10_000_000]),
            '10m_to_100m': len([p for p in prices if 10_000_000 <= p < 100_000_000]),
            '100m_to_1b': len([p for p in prices if 100_000_000 <= p < 1_000_000_000]),
            'over_1b': len([p for p in prices if p >= 1_000_000_000])
        }
        
        # Price percentiles
        percentiles = {
            'p25': np.percentile(prices, 25),
            'p50': np.percentile(prices, 50),
            'p75': np.percentile(prices, 75),
            'p90': np.percentile(prices, 90),
            'p95': np.percentile(prices, 95),
            'p99': np.percentile(prices, 99)
        }
        
        price_stats = {
            'total_with_price': len(prices),
            'total_products': len(self.sample_data),
            'price_coverage': (len(prices) / len(self.sample_data)) * 100,
            'ranges': price_ranges,
            'percentiles': percentiles,
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices)
        }
        
        self.analysis_results['price_analysis'] = price_stats
        return price_stats
    
    def rating_analysis(self):
        """Detailed rating analysis"""
        print("Analyzing ratings...")
        
        ratings = [p.get('rating_average', 0) or 0 for p in self.sample_data]
        ratings = [r for r in ratings if r > 0]  # Remove zero ratings
        
        rating_counts = [p.get('rating_count', 0) or 0 for p in self.sample_data]
        rating_counts = [rc for rc in rating_counts if rc > 0]  # Remove zero counts
        
        if not ratings:
            return {}
        
        # Rating ranges
        rating_ranges = {
            'excellent_90_100': len([r for r in ratings if r >= 90]),
            'very_good_80_89': len([r for r in ratings if 80 <= r < 90]),
            'good_70_79': len([r for r in ratings if 70 <= r < 80]),
            'average_60_69': len([r for r in ratings if 60 <= r < 70]),
            'below_average_0_59': len([r for r in ratings if r < 60])
        }
        
        # Rating count ranges
        count_ranges = {
            'no_reviews': len([rc for rc in rating_counts if rc == 0]),
            'few_reviews_1_10': len([rc for rc in rating_counts if 1 <= rc <= 10]),
            'some_reviews_11_100': len([rc for rc in rating_counts if 11 <= rc <= 100]),
            'many_reviews_101_1000': len([rc for rc in rating_counts if 101 <= rc <= 1000]),
            'lots_reviews_over_1000': len([rc for rc in rating_counts if rc > 1000])
        }
        
        rating_stats = {
            'total_with_ratings': len(ratings),
            'total_products': len(self.sample_data),
            'rating_coverage': (len(ratings) / len(self.sample_data)) * 100,
            'rating_ranges': rating_ranges,
            'count_ranges': count_ranges,
            'avg_rating': np.mean(ratings),
            'median_rating': np.median(ratings),
            'avg_count': np.mean(rating_counts) if rating_counts else 0,
            'median_count': np.median(rating_counts) if rating_counts else 0
        }
        
        self.analysis_results['rating_analysis'] = rating_stats
        return rating_stats
    
    def specification_analysis(self):
        """Analyze product specifications"""
        print("Analyzing specifications...")
        
        spec_stats = {
            'products_with_specs': 0,
            'total_specifications': 0,
            'spec_categories': Counter(),
            'spec_keys': Counter(),
            'avg_specs_per_product': 0
        }
        
        for product in self.sample_data:
            specs = product.get('specifications', {})
            if specs:
                spec_stats['products_with_specs'] += 1
                
                for category, category_specs in specs.items():
                    if category_specs:
                        spec_stats['spec_categories'][category] += 1
                        
                        for key, value in category_specs.items():
                            spec_stats['spec_keys'][key] += 1
                            spec_stats['total_specifications'] += 1
        
        spec_stats['avg_specs_per_product'] = (
            spec_stats['total_specifications'] / spec_stats['products_with_specs']
            if spec_stats['products_with_specs'] > 0 else 0
        )
        
        spec_stats['spec_coverage'] = (
            spec_stats['products_with_specs'] / len(self.sample_data)
        ) * 100
        
        # Top specification categories
        spec_stats['top_spec_categories'] = dict(spec_stats['spec_categories'].most_common(10))
        spec_stats['top_spec_keys'] = dict(spec_stats['spec_keys'].most_common(20))
        
        self.analysis_results['specification_analysis'] = spec_stats
        return spec_stats
    
    def cluster_analysis(self):
        """Analyze cluster distribution if available"""
        print("Analyzing clusters...")
        
        # Check if products have cluster information
        cluster_info = [p.get('cluster_id') for p in self.sample_data if 'cluster_id' in p]
        
        if not cluster_info:
            print("No cluster information found in sample")
            return {}
        
        cluster_counts = Counter(cluster_info)
        
        cluster_stats = {
            'total_clusters': len(cluster_counts),
            'cluster_distribution': dict(cluster_counts),
            'avg_products_per_cluster': np.mean(list(cluster_counts.values())),
            'largest_cluster': max(cluster_counts.items(), key=lambda x: x[1]),
            'smallest_cluster': min(cluster_counts.items(), key=lambda x: x[1])
        }
        
        self.analysis_results['cluster_analysis'] = cluster_stats
        return cluster_stats
    
    def generate_visualizations(self, output_dir: str = "sample_analysis_results"):
        """Generate visualizations for the analysis"""
        print("Generating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Category Distribution
        if 'category_analysis' in self.analysis_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            categories = list(self.analysis_results['category_analysis'].keys())
            counts = [self.analysis_results['category_analysis'][cat]['count'] for cat in categories]
            
            ax1.bar(categories, counts)
            ax1.set_title('Product Distribution by Category')
            ax1.set_xlabel('Category')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.pie(counts, labels=categories, autopct='%1.1f%%')
            ax2.set_title('Category Distribution (Percentage)')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/01_category_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Price Analysis
        if 'price_analysis' in self.analysis_results:
            prices = [p.get('selling_price', 0) or 0 for p in self.sample_data if p.get('selling_price', 0) and p.get('selling_price', 0) > 0]
            
            if prices:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.hist(prices, bins=50, alpha=0.7, edgecolor='black')
                ax1.set_title('Price Distribution')
                ax1.set_xlabel('Price (IRR)')
                ax1.set_ylabel('Frequency')
                ax1.set_yscale('log')
                
                # Log scale for better visualization
                log_prices = np.log10(prices)
                ax2.hist(log_prices, bins=50, alpha=0.7, edgecolor='black')
                ax2.set_title('Price Distribution (Log Scale)')
                ax2.set_xlabel('Log10(Price)')
                ax2.set_ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/02_price_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Rating Analysis
        if 'rating_analysis' in self.analysis_results:
            ratings = [p.get('rating_average', 0) or 0 for p in self.sample_data if p.get('rating_average', 0) and p.get('rating_average', 0) > 0]
            
            if ratings:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                ax1.hist(ratings, bins=20, alpha=0.7, edgecolor='black')
                ax1.set_title('Rating Distribution')
                ax1.set_xlabel('Rating')
                ax1.set_ylabel('Frequency')
                
                # Rating ranges pie chart
                ranges = self.analysis_results['rating_analysis']['rating_ranges']
                labels = ['90-100', '80-89', '70-79', '60-69', '0-59']
                values = [ranges['excellent_90_100'], ranges['very_good_80_89'], 
                         ranges['good_70_79'], ranges['average_60_69'], ranges['below_average_0_59']]
                
                ax2.pie(values, labels=labels, autopct='%1.1f%%')
                ax2.set_title('Rating Ranges Distribution')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/03_rating_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 4. Top Brands
        if 'brand_analysis' in self.analysis_results:
            top_brands = list(self.analysis_results['brand_analysis'].items())[:15]
            brands = [item[0] for item in top_brands]
            counts = [item[1]['count'] for item in top_brands]
            
            plt.figure(figsize=(12, 8))
            plt.barh(brands, counts)
            plt.title('Top 15 Brands by Product Count')
            plt.xlabel('Product Count')
            plt.ylabel('Brand')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/04_top_brands.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def save_results(self, output_dir: str = "sample_analysis_results"):
        """Save analysis results"""
        print("Saving analysis results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sample data
        sample_file = f'{output_dir}/sample_products.json'
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'sample_size': len(self.sample_data),
                    'total_products_in_dataset': len(self.complete_data['products']),
                    'sampling_date': pd.Timestamp.now().isoformat()
                },
                'products': self.sample_data
            }, f, ensure_ascii=False, indent=2)
        
        # Save analysis results
        results_file = f'{output_dir}/analysis_results.json'
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        converted_results = convert_numpy_types(self.analysis_results)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(converted_results, f, ensure_ascii=False, indent=2)
        
        # Save summary report
        self._create_summary_report(output_dir)
        
        print(f"Results saved to {output_dir}/")
    
    def _create_summary_report(self, output_dir: str):
        """Create a human-readable summary report"""
        report_file = f'{output_dir}/sample_analysis_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("PRODUCT SAMPLE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            if 'basic_statistics' in self.analysis_results:
                stats = self.analysis_results['basic_statistics']
                f.write("BASIC STATISTICS:\n")
                f.write(f"  Total Products Sampled: {stats['total_products']}\n")
                f.write(f"  Categories: {stats['categories']}\n")
                f.write(f"  Brands: {stats['brands']}\n")
                f.write(f"  Products with Descriptions: {stats['products_with_descriptions']} ({stats['products_with_descriptions']/stats['total_products']*100:.1f}%)\n")
                f.write(f"  Products with Specifications: {stats['products_with_specifications']} ({stats['products_with_specifications']/stats['total_products']*100:.1f}%)\n")
                f.write(f"  Products with Images: {stats['products_with_images']} ({stats['products_with_images']/stats['total_products']*100:.1f}%)\n\n")
                
                if 'price_stats' in stats:
                    price_stats = stats['price_stats']
                    f.write("PRICE STATISTICS:\n")
                    f.write(f"  Mean Price: {price_stats['mean']:,.0f} IRR\n")
                    f.write(f"  Median Price: {price_stats['median']:,.0f} IRR\n")
                    f.write(f"  Min Price: {price_stats['min']:,.0f} IRR\n")
                    f.write(f"  Max Price: {price_stats['max']:,.0f} IRR\n")
                    f.write(f"  Products with Price: {price_stats['count']}\n\n")
                
                if 'rating_stats' in stats:
                    rating_stats = stats['rating_stats']
                    f.write("RATING STATISTICS:\n")
                    f.write(f"  Mean Rating: {rating_stats['mean']:.2f}\n")
                    f.write(f"  Median Rating: {rating_stats['median']:.2f}\n")
                    f.write(f"  Min Rating: {rating_stats['min']:.2f}\n")
                    f.write(f"  Max Rating: {rating_stats['max']:.2f}\n")
                    f.write(f"  Products with Ratings: {rating_stats['count']}\n\n")
            
            # Category analysis
            if 'category_analysis' in self.analysis_results:
                f.write("CATEGORY ANALYSIS:\n")
                for category, data in self.analysis_results['category_analysis'].items():
                    f.write(f"  {category}: {data['count']} products ({data['percentage']:.1f}%)\n")
                    f.write(f"    Avg Price: {data['avg_price']:,.0f} IRR\n")
                    f.write(f"    Avg Rating: {data['avg_rating']:.2f}\n")
                    f.write(f"    Brands: {data['brands']}\n")
                f.write("\n")
            
            # Top brands
            if 'brand_analysis' in self.analysis_results:
                f.write("TOP 10 BRANDS:\n")
                top_brands = sorted(self.analysis_results['brand_analysis'].items(), 
                                  key=lambda x: x[1]['count'], reverse=True)[:10]
                for brand, data in top_brands:
                    f.write(f"  {brand}: {data['count']} products ({data['percentage']:.1f}%)\n")
                    f.write(f"    Avg Price: {data['avg_price']:,.0f} IRR\n")
                    f.write(f"    Avg Rating: {data['avg_rating']:.2f}\n")
                f.write("\n")
            
            # Price analysis
            if 'price_analysis' in self.analysis_results:
                price_analysis = self.analysis_results['price_analysis']
                f.write("PRICE RANGES:\n")
                ranges = price_analysis['ranges']
                f.write(f"  Under 1M: {ranges['under_1m']} products\n")
                f.write(f"  1M - 10M: {ranges['1m_to_10m']} products\n")
                f.write(f"  10M - 100M: {ranges['10m_to_100m']} products\n")
                f.write(f"  100M - 1B: {ranges['100m_to_1b']} products\n")
                f.write(f"  Over 1B: {ranges['over_1b']} products\n\n")
            
            # Rating analysis
            if 'rating_analysis' in self.analysis_results:
                rating_analysis = self.analysis_results['rating_analysis']
                f.write("RATING RANGES:\n")
                ranges = rating_analysis['rating_ranges']
                f.write(f"  Excellent (90-100): {ranges['excellent_90_100']} products\n")
                f.write(f"  Very Good (80-89): {ranges['very_good_80_89']} products\n")
                f.write(f"  Good (70-79): {ranges['good_70_79']} products\n")
                f.write(f"  Average (60-69): {ranges['average_60_69']} products\n")
                f.write(f"  Below Average (0-59): {ranges['below_average_0_59']} products\n\n")
            
            # Specification analysis
            if 'specification_analysis' in self.analysis_results:
                spec_analysis = self.analysis_results['specification_analysis']
                f.write("SPECIFICATION ANALYSIS:\n")
                f.write(f"  Products with Specs: {spec_analysis['products_with_specs']} ({spec_analysis['spec_coverage']:.1f}%)\n")
                f.write(f"  Total Specifications: {spec_analysis['total_specifications']}\n")
                f.write(f"  Avg Specs per Product: {spec_analysis['avg_specs_per_product']:.1f}\n")
                f.write(f"  Top Spec Categories: {list(spec_analysis['top_spec_categories'].keys())[:5]}\n\n")
            
            # Cluster analysis
            if 'cluster_analysis' in self.analysis_results:
                cluster_analysis = self.analysis_results['cluster_analysis']
                f.write("CLUSTER ANALYSIS:\n")
                f.write(f"  Total Clusters: {cluster_analysis['total_clusters']}\n")
                f.write(f"  Avg Products per Cluster: {cluster_analysis['avg_products_per_cluster']:.1f}\n")
                f.write(f"  Largest Cluster: {cluster_analysis['largest_cluster'][0]} ({cluster_analysis['largest_cluster'][1]} products)\n")
                f.write(f"  Smallest Cluster: {cluster_analysis['smallest_cluster'][0]} ({cluster_analysis['smallest_cluster'][1]} products)\n\n")
        
        print(f"Summary report saved to {report_file}")

def main():
    """Main function to run the sample analysis"""
    print("Starting Product Sample Analysis...")
    print("=" * 50)
    
    # Configuration
    dataset_path = "datasets/complete_dataset.json"
    sample_size = 4000
    output_dir = "sample_analysis_results"
    
    # Initialize analyzer
    analyzer = ProductSampleAnalyzer(dataset_path, sample_size)
    
    try:
        # Load dataset
        analyzer.load_dataset()
        
        # Sample products
        analyzer.sample_products()
        
        # Perform analyses
        analyzer.basic_statistics()
        analyzer.category_analysis()
        analyzer.brand_analysis()
        analyzer.price_analysis()
        analyzer.rating_analysis()
        analyzer.specification_analysis()
        analyzer.cluster_analysis()
        
        # Generate visualizations
        analyzer.generate_visualizations(output_dir)
        
        # Save results
        analyzer.save_results(output_dir)
        
        print("\n" + "=" * 50)
        print("Sample Analysis Complete!")
        print("=" * 50)
        
        # Print summary
        if 'basic_statistics' in analyzer.analysis_results:
            stats = analyzer.analysis_results['basic_statistics']
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"  Products Analyzed: {stats['total_products']}")
            print(f"  Categories: {stats['categories']}")
            print(f"  Brands: {stats['brands']}")
            
            if 'price_stats' in stats:
                print(f"  Avg Price: {stats['price_stats']['mean']:,.0f} IRR")
            if 'rating_stats' in stats:
                print(f"  Avg Rating: {stats['rating_stats']['mean']:.2f}")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"📈 Visualizations: {output_dir}/*.png")
        print(f"📄 Summary report: {output_dir}/sample_analysis_summary.txt")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
