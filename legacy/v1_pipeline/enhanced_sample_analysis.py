#!/usr/bin/env python3
"""
Enhanced Product Sample Analysis with Cluster Data
==================================================

This script samples 4000 products from the enhanced dataset that includes
cluster comparison analysis and performs comprehensive analysis including:
- Basic statistics with cluster information
- Category distribution with cluster analysis
- Brand analysis with cluster performance
- Price analysis with cluster comparisons
- Rating analysis with cluster comparisons
- Cluster performance analysis
- High/low performer identification
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

class EnhancedSampleAnalyzer:
    def __init__(self, enhanced_dataset_path: str, sample_size: int = 4000):
        self.enhanced_dataset_path = enhanced_dataset_path
        self.sample_size = sample_size
        self.sample_data = None
        self.analysis_results = {}
        
    def load_enhanced_dataset(self):
        """Load the enhanced dataset with cluster analysis"""
        print("Loading enhanced dataset with cluster analysis...")
        with open(self.enhanced_dataset_path, 'r', encoding='utf-8') as f:
            self.enhanced_data = json.load(f)
        
        print(f"Loaded {len(self.enhanced_data['products'])} enhanced products")
        return self.enhanced_data
    
    def sample_products(self):
        """Sample products from the enhanced dataset"""
        print(f"Sampling {self.sample_size} products from enhanced dataset...")
        
        products = self.enhanced_data['products']
        
        # Ensure we don't sample more than available
        actual_sample_size = min(self.sample_size, len(products))
        
        # Random sampling
        sampled_indices = random.sample(range(len(products)), actual_sample_size)
        self.sample_data = [products[i] for i in sampled_indices]
        
        print(f"Sampled {len(self.sample_data)} products with cluster analysis")
        return self.sample_data
    
    def basic_statistics(self):
        """Calculate basic statistics including cluster information"""
        print("Calculating basic statistics with cluster data...")
        
        stats = {
            'total_products': len(self.sample_data),
            'categories': len(set(p.get('category_name', 'Unknown') for p in self.sample_data)),
            'brands': len(set(p.get('brand', 'Unknown') for p in self.sample_data)),
            'clusters': len(set(p.get('cluster_id') for p in self.sample_data if p.get('cluster_id') is not None)),
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
    
    def cluster_performance_analysis(self):
        """Analyze cluster performance comparisons"""
        print("Analyzing cluster performance comparisons...")
        
        # Count products by comparison categories
        rating_count_comparisons = Counter(p.get('rating_count_vs_cluster', 'unknown') for p in self.sample_data)
        rating_average_comparisons = Counter(p.get('rating_average_vs_cluster', 'unknown') for p in self.sample_data)
        price_comparisons = Counter(p.get('price_vs_cluster', 'unknown') for p in self.sample_data)
        
        # Find high performers (high rating, low price)
        high_performers = [p for p in self.sample_data 
                          if p.get('rating_average_vs_cluster') == 'high' 
                          and p.get('price_vs_cluster') == 'low']
        
        # Find bargains (high rating average, low price)
        bargains = [p for p in self.sample_data 
                   if p.get('rating_average_vs_cluster') == 'high' 
                   and p.get('price_vs_cluster') == 'low']
        
        # Find premium products (high price, high rating)
        premium_products = [p for p in self.sample_data 
                           if p.get('price_vs_cluster') == 'high' 
                           and p.get('rating_average_vs_cluster') == 'high']
        
        # Find underperformers (low rating, high price)
        underperformers = [p for p in self.sample_data 
                          if p.get('rating_average_vs_cluster') == 'low' 
                          and p.get('price_vs_cluster') == 'high']
        
        cluster_performance = {
            'rating_count_comparisons': dict(rating_count_comparisons),
            'rating_average_comparisons': dict(rating_average_comparisons),
            'price_comparisons': dict(price_comparisons),
            'high_performers_count': len(high_performers),
            'bargains_count': len(bargains),
            'premium_products_count': len(premium_products),
            'underperformers_count': len(underperformers),
            'high_performers': high_performers[:10],  # Top 10 examples
            'bargains': bargains[:10],  # Top 10 examples
            'premium_products': premium_products[:10],  # Top 10 examples
            'underperformers': underperformers[:10]  # Top 10 examples
        }
        
        self.analysis_results['cluster_performance_analysis'] = cluster_performance
        return cluster_performance
    
    def category_analysis_with_clusters(self):
        """Analyze category distribution with cluster information"""
        print("Analyzing categories with cluster data...")
        
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
            
            # Cluster analysis for this category
            clusters = [p.get('cluster_id') for p in category_products if p.get('cluster_id') is not None]
            cluster_counts = Counter(clusters)
            
            # Performance analysis for this category
            high_performers = len([p for p in category_products 
                                 if p.get('rating_average_vs_cluster') == 'high' 
                                 and p.get('price_vs_cluster') == 'low'])
            
            category_stats[category] = {
                'count': count,
                'percentage': (count / len(self.sample_data)) * 100,
                'avg_price': np.mean(prices) if prices else 0,
                'avg_rating': np.mean(ratings) if ratings else 0,
                'brands': len(set(p.get('brand', 'Unknown') for p in category_products)),
                'clusters': len(cluster_counts),
                'top_clusters': dict(cluster_counts.most_common(5)),
                'high_performers': high_performers,
                'high_performers_percentage': (high_performers / count) * 100 if count > 0 else 0
            }
        
        self.analysis_results['category_analysis_with_clusters'] = category_stats
        return category_stats
    
    def brand_analysis_with_clusters(self):
        """Analyze brand distribution with cluster performance"""
        print("Analyzing brands with cluster performance...")
        
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
            
            # Cluster performance for this brand
            clusters = [p.get('cluster_id') for p in brand_products if p.get('cluster_id') is not None]
            cluster_counts = Counter(clusters)
            
            # Performance metrics
            high_performers = len([p for p in brand_products 
                                 if p.get('rating_average_vs_cluster') == 'high' 
                                 and p.get('price_vs_cluster') == 'low'])
            
            avg_price_diff = np.mean([p.get('price_diff_percent', 0) or 0 for p in brand_products])
            avg_rating_diff = np.mean([p.get('rating_average_diff_percent', 0) or 0 for p in brand_products])
            
            brand_stats[brand] = {
                'count': count,
                'percentage': (count / len(self.sample_data)) * 100,
                'avg_price': np.mean(prices) if prices else 0,
                'avg_rating': np.mean(ratings) if ratings else 0,
                'categories': list(categories),
                'category_count': len(categories),
                'clusters': len(cluster_counts),
                'top_clusters': dict(cluster_counts.most_common(3)),
                'high_performers': high_performers,
                'high_performers_percentage': (high_performers / count) * 100 if count > 0 else 0,
                'avg_price_diff_percent': avg_price_diff,
                'avg_rating_diff_percent': avg_rating_diff
            }
        
        self.analysis_results['brand_analysis_with_clusters'] = brand_stats
        return brand_stats
    
    def cluster_distribution_analysis(self):
        """Analyze cluster distribution in the sample"""
        print("Analyzing cluster distribution...")
        
        clusters = [p.get('cluster_id') for p in self.sample_data if p.get('cluster_id') is not None]
        cluster_counts = Counter(clusters)
        
        # Cluster statistics
        cluster_sizes = list(cluster_counts.values())
        
        cluster_stats = {
            'total_clusters': len(cluster_counts),
            'cluster_distribution': dict(cluster_counts),
            'avg_products_per_cluster': np.mean(cluster_sizes),
            'median_products_per_cluster': np.median(cluster_sizes),
            'largest_cluster': max(cluster_counts.items(), key=lambda x: x[1]),
            'smallest_cluster': min(cluster_counts.items(), key=lambda x: x[1]),
            'cluster_size_range': (min(cluster_sizes), max(cluster_sizes))
        }
        
        # Top clusters by size
        top_clusters = cluster_counts.most_common(10)
        cluster_details = {}
        
        for cluster_id, count in top_clusters:
            cluster_products = [p for p in self.sample_data if p.get('cluster_id') == cluster_id]
            
            # Get cluster averages from first product (they should be the same)
            if cluster_products:
                first_product = cluster_products[0]
                cluster_details[cluster_id] = {
                    'count': count,
                    'avg_rating_count': first_product.get('cluster_avg_rating_count', 0),
                    'avg_rating_average': first_product.get('cluster_avg_rating_average', 0),
                    'avg_price': first_product.get('cluster_avg_price', 0),
                    'categories': list(set(p.get('category_name', 'Unknown') for p in cluster_products)),
                    'brands': list(set(p.get('brand', 'Unknown') for p in cluster_products))
                }
        
        cluster_stats['top_clusters_details'] = cluster_details
        
        self.analysis_results['cluster_distribution_analysis'] = cluster_stats
        return cluster_stats
    
    def generate_enhanced_visualizations(self, output_dir: str = "enhanced_sample_analysis_results"):
        """Generate enhanced visualizations including cluster data"""
        print("Generating enhanced visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Category Distribution with High Performers
        if 'category_analysis_with_clusters' in self.analysis_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            categories = list(self.analysis_results['category_analysis_with_clusters'].keys())
            counts = [self.analysis_results['category_analysis_with_clusters'][cat]['count'] for cat in categories]
            high_performers = [self.analysis_results['category_analysis_with_clusters'][cat]['high_performers'] for cat in categories]
            
            ax1.bar(categories, counts, alpha=0.7, label='Total Products')
            ax1.bar(categories, high_performers, alpha=0.9, label='High Performers')
            ax1.set_title('Product Distribution by Category (with High Performers)')
            ax1.set_xlabel('Category')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()
            
            # High performers percentage
            percentages = [self.analysis_results['category_analysis_with_clusters'][cat]['high_performers_percentage'] for cat in categories]
            ax2.bar(categories, percentages, alpha=0.7, color='green')
            ax2.set_title('High Performers Percentage by Category')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Percentage (%)')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/01_category_with_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Cluster Performance Comparison
        if 'cluster_performance_analysis' in self.analysis_results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            perf_analysis = self.analysis_results['cluster_performance_analysis']
            
            # Rating count comparisons
            rating_count_data = perf_analysis['rating_count_comparisons']
            ax1.pie(rating_count_data.values(), labels=rating_count_data.keys(), autopct='%1.1f%%')
            ax1.set_title('Rating Count vs Cluster Distribution')
            
            # Rating average comparisons
            rating_avg_data = perf_analysis['rating_average_comparisons']
            ax2.pie(rating_avg_data.values(), labels=rating_avg_data.keys(), autopct='%1.1f%%')
            ax2.set_title('Rating Average vs Cluster Distribution')
            
            # Price comparisons
            price_data = perf_analysis['price_comparisons']
            ax3.pie(price_data.values(), labels=price_data.keys(), autopct='%1.1f%%')
            ax3.set_title('Price vs Cluster Distribution')
            
            # Performance categories
            performance_categories = ['High Performers', 'Bargains', 'Premium', 'Underperformers']
            performance_counts = [
                perf_analysis['high_performers_count'],
                perf_analysis['bargains_count'],
                perf_analysis['premium_products_count'],
                perf_analysis['underperformers_count']
            ]
            
            ax4.bar(performance_categories, performance_counts, alpha=0.7)
            ax4.set_title('Product Performance Categories')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/02_cluster_performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Top Clusters Analysis
        if 'cluster_distribution_analysis' in self.analysis_results:
            cluster_analysis = self.analysis_results['cluster_distribution_analysis']
            top_clusters = list(cluster_analysis['top_clusters_details'].items())[:10]
            
            cluster_ids = [str(item[0]) for item in top_clusters]
            counts = [item[1]['count'] for item in top_clusters]
            avg_ratings = [item[1]['avg_rating_average'] for item in top_clusters]
            avg_prices = [item[1]['avg_price'] for item in top_clusters]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.bar(cluster_ids, counts, alpha=0.7)
            ax1.set_title('Top 10 Clusters by Product Count')
            ax1.set_xlabel('Cluster ID')
            ax1.set_ylabel('Product Count')
            
            ax2.scatter(avg_prices, avg_ratings, s=counts, alpha=0.7)
            ax2.set_title('Cluster Performance: Price vs Rating')
            ax2.set_xlabel('Average Price (IRR)')
            ax2.set_ylabel('Average Rating')
            ax2.set_xscale('log')
            
            # Add cluster ID labels
            for i, cluster_id in enumerate(cluster_ids):
                ax2.annotate(cluster_id, (avg_prices[i], avg_ratings[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/03_top_clusters_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Brand Performance Analysis
        if 'brand_analysis_with_clusters' in self.analysis_results:
            brand_analysis = self.analysis_results['brand_analysis_with_clusters']
            top_brands = list(brand_analysis.items())[:15]
            
            brands = [item[0] for item in top_brands]
            high_performers = [item[1]['high_performers_percentage'] for item in top_brands]
            avg_price_diffs = [item[1]['avg_price_diff_percent'] for item in top_brands]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.barh(brands, high_performers, alpha=0.7)
            ax1.set_title('High Performers Percentage by Brand')
            ax1.set_xlabel('High Performers Percentage (%)')
            
            ax2.barh(brands, avg_price_diffs, alpha=0.7, color='orange')
            ax2.set_title('Average Price Difference vs Cluster by Brand')
            ax2.set_xlabel('Price Difference (%)')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/04_brand_performance_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Enhanced visualizations saved to {output_dir}/")
    
    def save_enhanced_results(self, output_dir: str = "enhanced_sample_analysis_results"):
        """Save enhanced analysis results"""
        print("Saving enhanced analysis results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sample data
        sample_file = f'{output_dir}/enhanced_sample_products.json'
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'sample_size': len(self.sample_data),
                    'total_products_in_dataset': len(self.enhanced_data['products']),
                    'sampling_date': pd.Timestamp.now().isoformat(),
                    'includes_cluster_analysis': True
                },
                'products': self.sample_data
            }, f, ensure_ascii=False, indent=2)
        
        # Save analysis results
        results_file = f'{output_dir}/enhanced_analysis_results.json'
        
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
        self._create_enhanced_summary_report(output_dir)
        
        print(f"Enhanced results saved to {output_dir}/")
    
    def _create_enhanced_summary_report(self, output_dir: str):
        """Create a human-readable enhanced summary report"""
        report_file = f'{output_dir}/enhanced_sample_analysis_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED PRODUCT SAMPLE ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic statistics
            if 'basic_statistics' in self.analysis_results:
                stats = self.analysis_results['basic_statistics']
                f.write("BASIC STATISTICS:\n")
                f.write(f"  Total Products Sampled: {stats['total_products']}\n")
                f.write(f"  Categories: {stats['categories']}\n")
                f.write(f"  Brands: {stats['brands']}\n")
                f.write(f"  Clusters: {stats['clusters']}\n")
                f.write(f"  Products with Descriptions: {stats['products_with_descriptions']} ({stats['products_with_descriptions']/stats['total_products']*100:.1f}%)\n")
                f.write(f"  Products with Specifications: {stats['products_with_specifications']} ({stats['products_with_specifications']/stats['total_products']*100:.1f}%)\n")
                f.write(f"  Products with Images: {stats['products_with_images']} ({stats['products_with_images']/stats['total_products']*100:.1f}%)\n\n")
            
            # Cluster performance analysis
            if 'cluster_performance_analysis' in self.analysis_results:
                perf_analysis = self.analysis_results['cluster_performance_analysis']
                f.write("CLUSTER PERFORMANCE ANALYSIS:\n")
                f.write(f"  High Performers: {perf_analysis['high_performers_count']} products\n")
                f.write(f"  Bargains: {perf_analysis['bargains_count']} products\n")
                f.write(f"  Premium Products: {perf_analysis['premium_products_count']} products\n")
                f.write(f"  Underperformers: {perf_analysis['underperformers_count']} products\n\n")
                
                f.write("PERFORMANCE DISTRIBUTION:\n")
                f.write(f"  Rating Count vs Cluster: {perf_analysis['rating_count_comparisons']}\n")
                f.write(f"  Rating Average vs Cluster: {perf_analysis['rating_average_comparisons']}\n")
                f.write(f"  Price vs Cluster: {perf_analysis['price_comparisons']}\n\n")
            
            # Category analysis with clusters
            if 'category_analysis_with_clusters' in self.analysis_results:
                f.write("CATEGORY ANALYSIS WITH CLUSTER PERFORMANCE:\n")
                for category, data in self.analysis_results['category_analysis_with_clusters'].items():
                    f.write(f"  {category}: {data['count']} products ({data['percentage']:.1f}%)\n")
                    f.write(f"    Avg Price: {data['avg_price']:,.0f} IRR\n")
                    f.write(f"    Avg Rating: {data['avg_rating']:.2f}\n")
                    f.write(f"    Clusters: {data['clusters']}\n")
                    f.write(f"    High Performers: {data['high_performers']} ({data['high_performers_percentage']:.1f}%)\n")
                f.write("\n")
            
            # Top brands with performance
            if 'brand_analysis_with_clusters' in self.analysis_results:
                f.write("TOP 10 BRANDS WITH CLUSTER PERFORMANCE:\n")
                top_brands = sorted(self.analysis_results['brand_analysis_with_clusters'].items(), 
                                  key=lambda x: x[1]['count'], reverse=True)[:10]
                for brand, data in top_brands:
                    f.write(f"  {brand}: {data['count']} products ({data['percentage']:.1f}%)\n")
                    f.write(f"    Avg Price: {data['avg_price']:,.0f} IRR\n")
                    f.write(f"    Avg Rating: {data['avg_rating']:.2f}\n")
                    f.write(f"    High Performers: {data['high_performers']} ({data['high_performers_percentage']:.1f}%)\n")
                    f.write(f"    Avg Price Diff vs Cluster: {data['avg_price_diff_percent']:+.1f}%\n")
                f.write("\n")
            
            # Cluster distribution
            if 'cluster_distribution_analysis' in self.analysis_results:
                cluster_analysis = self.analysis_results['cluster_distribution_analysis']
                f.write("CLUSTER DISTRIBUTION ANALYSIS:\n")
                f.write(f"  Total Clusters: {cluster_analysis['total_clusters']}\n")
                f.write(f"  Avg Products per Cluster: {cluster_analysis['avg_products_per_cluster']:.1f}\n")
                f.write(f"  Largest Cluster: {cluster_analysis['largest_cluster'][0]} ({cluster_analysis['largest_cluster'][1]} products)\n")
                f.write(f"  Smallest Cluster: {cluster_analysis['smallest_cluster'][0]} ({cluster_analysis['smallest_cluster'][1]} products)\n\n")
                
                f.write("TOP 5 CLUSTERS BY SIZE:\n")
                top_clusters = list(cluster_analysis['top_clusters_details'].items())[:5]
                for cluster_id, data in top_clusters:
                    f.write(f"  Cluster {cluster_id}: {data['count']} products\n")
                    f.write(f"    Avg Rating: {data['avg_rating_average']:.2f}\n")
                    f.write(f"    Avg Price: {data['avg_price']:,.0f} IRR\n")
                    f.write(f"    Categories: {data['categories'][:3]}\n")
                f.write("\n")
        
        print(f"Enhanced summary report saved to {report_file}")

def main():
    """Main function to run the enhanced sample analysis"""
    print("Starting Enhanced Product Sample Analysis...")
    print("=" * 60)
    
    # Configuration
    enhanced_dataset_path = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    sample_size = 4000
    output_dir = "enhanced_sample_analysis_results"
    
    # Initialize analyzer
    analyzer = EnhancedSampleAnalyzer(enhanced_dataset_path, sample_size)
    
    try:
        # Load enhanced dataset
        analyzer.load_enhanced_dataset()
        
        # Sample products
        analyzer.sample_products()
        
        # Perform analyses
        analyzer.basic_statistics()
        analyzer.cluster_performance_analysis()
        analyzer.category_analysis_with_clusters()
        analyzer.brand_analysis_with_clusters()
        analyzer.cluster_distribution_analysis()
        
        # Generate visualizations
        analyzer.generate_enhanced_visualizations(output_dir)
        
        # Save results
        analyzer.save_enhanced_results(output_dir)
        
        print("\n" + "=" * 60)
        print("Enhanced Sample Analysis Complete!")
        print("=" * 60)
        
        # Print summary
        if 'basic_statistics' in analyzer.analysis_results:
            stats = analyzer.analysis_results['basic_statistics']
            print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
            print(f"  Products Analyzed: {stats['total_products']}")
            print(f"  Categories: {stats['categories']}")
            print(f"  Brands: {stats['brands']}")
            print(f"  Clusters: {stats['clusters']}")
            
            if 'price_stats' in stats:
                print(f"  Avg Price: {stats['price_stats']['mean']:,.0f} IRR")
            if 'rating_stats' in stats:
                print(f"  Avg Rating: {stats['rating_stats']['mean']:.2f}")
        
        if 'cluster_performance_analysis' in analyzer.analysis_results:
            perf = analyzer.analysis_results['cluster_performance_analysis']
            print(f"\n🎯 CLUSTER PERFORMANCE:")
            print(f"  High Performers: {perf['high_performers_count']}")
            print(f"  Bargains: {perf['bargains_count']}")
            print(f"  Premium Products: {perf['premium_products_count']}")
            print(f"  Underperformers: {perf['underperformers_count']}")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"📈 Enhanced visualizations: {output_dir}/*.png")
        print(f"📄 Enhanced summary report: {output_dir}/enhanced_sample_analysis_summary.txt")
        
    except Exception as e:
        print(f"Error during enhanced analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
