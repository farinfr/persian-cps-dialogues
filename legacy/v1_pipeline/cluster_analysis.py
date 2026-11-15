#!/usr/bin/env python3
"""
Cluster Analysis Script
=======================

This script calculates average rating_count, rating_average, and price for each cluster
and adds comparison columns to show if a product is high or low compared to its cluster average.

Input files:
- final_clustered_products.json: Contains product-cluster mappings
- complete_dataset.json: Contains full product data with ratings and prices

Output:
- Enhanced dataset with cluster comparison columns
- Cluster statistics summary
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import os

class ClusterAnalyzer:
    def __init__(self, clustered_file: str, complete_dataset_file: str):
        self.clustered_file = clustered_file
        self.complete_dataset_file = complete_dataset_file
        self.clustered_products = {}
        self.complete_products = {}
        self.cluster_stats = {}
        
    def load_data(self):
        """Load clustered products and complete dataset"""
        print("Loading clustered products...")
        with open(self.clustered_file, 'r', encoding='utf-8') as f:
            clustered_data = json.load(f)
            
        # Create mapping: product_id -> cluster_id
        for product in clustered_data['products']:
            self.clustered_products[product['id']] = product['cluster']
            
        print(f"Loaded {len(self.clustered_products)} clustered products")
        
        print("Loading complete dataset...")
        with open(self.complete_dataset_file, 'r', encoding='utf-8') as f:
            complete_data = json.load(f)
            
        # Create mapping: product_id -> product_data
        for product in complete_data['products']:
            self.complete_products[product['id']] = product
            
        print(f"Loaded {len(self.complete_products)} complete products")
        
    def calculate_cluster_averages(self):
        """Calculate average rating_count, rating_average, and price for each cluster"""
        print("Calculating cluster averages...")
        
        # Group products by cluster
        cluster_products = defaultdict(list)
        for product_id, cluster_id in self.clustered_products.items():
            if product_id in self.complete_products:
                cluster_products[cluster_id].append(product_id)
        
        # Calculate averages for each cluster
        for cluster_id, product_ids in cluster_products.items():
            rating_counts = []
            rating_averages = []
            prices = []
            
            for product_id in product_ids:
                product = self.complete_products[product_id]
                
                # Extract rating_count
                rating_count = product.get('rating_count', 0)
                if rating_count is not None:
                    rating_counts.append(rating_count)
                
                # Extract rating_average
                rating_average = product.get('rating_average', 0)
                if rating_average is not None:
                    rating_averages.append(rating_average)
                
                # Extract selling_price
                selling_price = product.get('selling_price', 0)
                if selling_price is not None:
                    prices.append(selling_price)
            
            # Calculate averages
            avg_rating_count = np.mean(rating_counts) if rating_counts else 0
            avg_rating_average = np.mean(rating_averages) if rating_averages else 0
            avg_price = np.mean(prices) if prices else 0
            
            self.cluster_stats[cluster_id] = {
                'product_count': len(product_ids),
                'avg_rating_count': avg_rating_count,
                'avg_rating_average': avg_rating_average,
                'avg_price': avg_price,
                'products': product_ids
            }
        
        print(f"Calculated averages for {len(self.cluster_stats)} clusters")
        
    def add_comparison_columns(self):
        """Add comparison columns to products"""
        print("Adding comparison columns...")
        
        enhanced_products = []
        
        for product_id, cluster_id in self.clustered_products.items():
            if product_id not in self.complete_products:
                continue
                
            product = self.complete_products[product_id].copy()
            cluster_stat = self.cluster_stats.get(cluster_id, {})
            
            # Get product values
            product_rating_count = product.get('rating_count', 0) or 0
            product_rating_average = product.get('rating_average', 0) or 0
            product_price = product.get('selling_price', 0) or 0
            
            # Get cluster averages
            cluster_avg_rating_count = cluster_stat.get('avg_rating_count', 0)
            cluster_avg_rating_average = cluster_stat.get('avg_rating_average', 0)
            cluster_avg_price = cluster_stat.get('avg_price', 0)
            
            # Add comparison columns
            product['cluster_id'] = cluster_id
            product['cluster_avg_rating_count'] = cluster_avg_rating_count
            product['cluster_avg_rating_average'] = cluster_avg_rating_average
            product['cluster_avg_price'] = cluster_avg_price
            
            # Determine if product is high or low compared to cluster average
            product['rating_count_vs_cluster'] = self._compare_value(
                product_rating_count, cluster_avg_rating_count
            )
            product['rating_average_vs_cluster'] = self._compare_value(
                product_rating_average, cluster_avg_rating_average
            )
            product['price_vs_cluster'] = self._compare_value(
                product_price, cluster_avg_price
            )
            
            # Add percentage differences
            product['rating_count_diff_percent'] = self._calculate_percentage_diff(
                product_rating_count, cluster_avg_rating_count
            )
            product['rating_average_diff_percent'] = self._calculate_percentage_diff(
                product_rating_average, cluster_avg_rating_average
            )
            product['price_diff_percent'] = self._calculate_percentage_diff(
                product_price, cluster_avg_price
            )
            
            enhanced_products.append(product)
        
        print(f"Enhanced {len(enhanced_products)} products with comparison data")
        return enhanced_products
        
    def _compare_value(self, product_value: float, cluster_avg: float) -> str:
        """Compare product value to cluster average"""
        if cluster_avg == 0:
            return "unknown"
        
        diff_percent = ((product_value - cluster_avg) / cluster_avg) * 100
        
        if diff_percent > 20:
            return "high"
        elif diff_percent < -20:
            return "low"
        else:
            return "average"
    
    def _calculate_percentage_diff(self, product_value: float, cluster_avg: float) -> float:
        """Calculate percentage difference from cluster average"""
        if cluster_avg == 0:
            return 0.0
        return ((product_value - cluster_avg) / cluster_avg) * 100
    
    def save_results(self, enhanced_products: List[Dict], output_dir: str = "cluster_analysis_results"):
        """Save enhanced dataset and cluster statistics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save enhanced products
        enhanced_file = os.path.join(output_dir, "enhanced_products_with_cluster_comparison.json")
        with open(enhanced_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_products': len(enhanced_products),
                    'total_clusters': len(self.cluster_stats),
                    'analysis_date': pd.Timestamp.now().isoformat()
                },
                'products': enhanced_products
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved enhanced products to: {enhanced_file}")
        
        # Save cluster statistics
        cluster_stats_file = os.path.join(output_dir, "cluster_statistics.json")
        with open(cluster_stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.cluster_stats, f, ensure_ascii=False, indent=2)
        
        print(f"Saved cluster statistics to: {cluster_stats_file}")
        
        # Create summary report
        self._create_summary_report(output_dir)
        
    def _create_summary_report(self, output_dir: str):
        """Create a summary report of the analysis"""
        report_file = os.path.join(output_dir, "cluster_analysis_summary.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CLUSTER ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Products Analyzed: {len(self.clustered_products)}\n")
            f.write(f"Total Clusters: {len(self.cluster_stats)}\n\n")
            
            # Cluster size distribution
            cluster_sizes = [stat['product_count'] for stat in self.cluster_stats.values()]
            f.write("CLUSTER SIZE DISTRIBUTION:\n")
            f.write(f"  Average products per cluster: {np.mean(cluster_sizes):.1f}\n")
            f.write(f"  Min products per cluster: {min(cluster_sizes)}\n")
            f.write(f"  Max products per cluster: {max(cluster_sizes)}\n\n")
            
            # Rating statistics
            all_rating_counts = [stat['avg_rating_count'] for stat in self.cluster_stats.values()]
            all_rating_averages = [stat['avg_rating_average'] for stat in self.cluster_stats.values()]
            all_prices = [stat['avg_price'] for stat in self.cluster_stats.values()]
            
            f.write("RATING COUNT STATISTICS:\n")
            f.write(f"  Average across clusters: {np.mean(all_rating_counts):.1f}\n")
            f.write(f"  Min cluster average: {min(all_rating_counts):.1f}\n")
            f.write(f"  Max cluster average: {max(all_rating_counts):.1f}\n\n")
            
            f.write("RATING AVERAGE STATISTICS:\n")
            f.write(f"  Average across clusters: {np.mean(all_rating_averages):.1f}\n")
            f.write(f"  Min cluster average: {min(all_rating_averages):.1f}\n")
            f.write(f"  Max cluster average: {max(all_rating_averages):.1f}\n\n")
            
            f.write("PRICE STATISTICS:\n")
            f.write(f"  Average across clusters: {np.mean(all_prices):,.0f}\n")
            f.write(f"  Min cluster average: {min(all_prices):,.0f}\n")
            f.write(f"  Max cluster average: {max(all_prices):,.0f}\n\n")
            
            # Top clusters by different metrics
            f.write("TOP 5 CLUSTERS BY RATING COUNT:\n")
            top_rating_count = sorted(self.cluster_stats.items(), 
                                    key=lambda x: x[1]['avg_rating_count'], reverse=True)[:5]
            for cluster_id, stats in top_rating_count:
                f.write(f"  Cluster {cluster_id}: {stats['avg_rating_count']:.1f} avg rating count\n")
            
            f.write("\nTOP 5 CLUSTERS BY RATING AVERAGE:\n")
            top_rating_avg = sorted(self.cluster_stats.items(), 
                                  key=lambda x: x[1]['avg_rating_average'], reverse=True)[:5]
            for cluster_id, stats in top_rating_avg:
                f.write(f"  Cluster {cluster_id}: {stats['avg_rating_average']:.1f} avg rating\n")
            
            f.write("\nTOP 5 CLUSTERS BY PRICE:\n")
            top_price = sorted(self.cluster_stats.items(), 
                             key=lambda x: x[1]['avg_price'], reverse=True)[:5]
            for cluster_id, stats in top_price:
                f.write(f"  Cluster {cluster_id}: {stats['avg_price']:,.0f} avg price\n")
        
        print(f"Saved summary report to: {report_file}")

def main():
    """Main function to run the cluster analysis"""
    print("Starting Cluster Analysis...")
    print("=" * 50)
    
    # File paths
    clustered_file = "datasets/final_clustered_products.json"
    complete_dataset_file = "datasets/complete_dataset.json"
    
    # Initialize analyzer
    analyzer = ClusterAnalyzer(clustered_file, complete_dataset_file)
    
    try:
        # Load data
        analyzer.load_data()
        
        # Calculate cluster averages
        analyzer.calculate_cluster_averages()
        
        # Add comparison columns
        enhanced_products = analyzer.add_comparison_columns()
        
        # Save results
        analyzer.save_results(enhanced_products)
        
        print("\n" + "=" * 50)
        print("Cluster Analysis Complete!")
        print("=" * 50)
        
        # Print some sample results
        print("\nSample Enhanced Products:")
        for i, product in enumerate(enhanced_products[:3]):
            print(f"\nProduct {i+1}: {product.get('title', 'Unknown')[:50]}...")
            print(f"  Cluster: {product.get('cluster_id')}")
            print(f"  Rating Count vs Cluster: {product.get('rating_count_vs_cluster')} "
                  f"({product.get('rating_count_diff_percent'):+.1f}%)")
            print(f"  Rating Average vs Cluster: {product.get('rating_average_vs_cluster')} "
                  f"({product.get('rating_average_diff_percent'):+.1f}%)")
            print(f"  Price vs Cluster: {product.get('price_vs_cluster')} "
                  f"({product.get('price_diff_percent'):+.1f}%)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
