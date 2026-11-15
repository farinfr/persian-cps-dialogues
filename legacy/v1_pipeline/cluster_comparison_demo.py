#!/usr/bin/env python3
"""
Cluster Comparison Usage Examples
=================================

This script demonstrates how to use the enhanced dataset with cluster comparison data.
"""

import json
import pandas as pd
from typing import List, Dict

def load_enhanced_dataset(file_path: str) -> Dict:
    """Load the enhanced dataset with cluster comparisons"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_products_by_comparison(data: Dict, 
                              rating_count_vs: str = None,
                              rating_average_vs: str = None, 
                              price_vs: str = None) -> List[Dict]:
    """
    Find products based on their comparison to cluster averages
    
    Args:
        rating_count_vs: 'high', 'low', 'average', or None
        rating_average_vs: 'high', 'low', 'average', or None  
        price_vs: 'high', 'low', 'average', or None
    """
    products = data['products']
    filtered = []
    
    for product in products:
        match = True
        
        if rating_count_vs and product.get('rating_count_vs_cluster') != rating_count_vs:
            match = False
        if rating_average_vs and product.get('rating_average_vs_cluster') != rating_average_vs:
            match = False
        if price_vs and product.get('price_vs_cluster') != price_vs:
            match = False
            
        if match:
            filtered.append(product)
    
    return filtered

def analyze_cluster_performance(data: Dict, cluster_id: int) -> Dict:
    """Analyze performance of products in a specific cluster"""
    products = [p for p in data['products'] if p.get('cluster_id') == cluster_id]
    
    if not products:
        return {"error": f"No products found in cluster {cluster_id}"}
    
    # Count comparisons
    rating_count_stats = {}
    rating_average_stats = {}
    price_stats = {}
    
    for product in products:
        # Rating count comparison
        rc_vs = product.get('rating_count_vs_cluster', 'unknown')
        rating_count_stats[rc_vs] = rating_count_stats.get(rc_vs, 0) + 1
        
        # Rating average comparison
        ra_vs = product.get('rating_average_vs_cluster', 'unknown')
        rating_average_stats[ra_vs] = rating_average_stats.get(ra_vs, 0) + 1
        
        # Price comparison
        p_vs = product.get('price_vs_cluster', 'unknown')
        price_stats[p_vs] = price_stats.get(p_vs, 0) + 1
    
    return {
        "cluster_id": cluster_id,
        "total_products": len(products),
        "rating_count_comparison": rating_count_stats,
        "rating_average_comparison": rating_average_stats,
        "price_comparison": price_stats,
        "cluster_avg_rating_count": products[0].get('cluster_avg_rating_count', 0),
        "cluster_avg_rating_average": products[0].get('cluster_avg_rating_average', 0),
        "cluster_avg_price": products[0].get('cluster_avg_price', 0)
    }

def get_top_performers(data: Dict, metric: str = 'rating_average', limit: int = 10) -> List[Dict]:
    """
    Get top performing products based on cluster comparison
    
    Args:
        metric: 'rating_count', 'rating_average', or 'price'
        limit: number of products to return
    """
    products = data['products']
    
    # Filter products that are 'high' compared to their cluster
    high_performers = []
    
    for product in products:
        if metric == 'rating_count' and product.get('rating_count_vs_cluster') == 'high':
            high_performers.append(product)
        elif metric == 'rating_average' and product.get('rating_average_vs_cluster') == 'high':
            high_performers.append(product)
        elif metric == 'price' and product.get('price_vs_cluster') == 'high':
            high_performers.append(product)
    
    # Sort by the actual metric value
    if metric == 'rating_count':
        high_performers.sort(key=lambda x: x.get('rating_count', 0), reverse=True)
    elif metric == 'rating_average':
        high_performers.sort(key=lambda x: x.get('rating_average', 0), reverse=True)
    elif metric == 'price':
        high_performers.sort(key=lambda x: x.get('selling_price', 0), reverse=True)
    
    return high_performers[:limit]

def main():
    """Demonstrate usage of the enhanced dataset"""
    print("Cluster Comparison Analysis Demo")
    print("=" * 50)
    
    # Load enhanced dataset
    enhanced_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    data = load_enhanced_dataset(enhanced_file)
    
    print(f"Loaded {data['metadata']['total_products']} products from {data['metadata']['total_clusters']} clusters")
    
    # Example 1: Find products that are high in rating count but low in price
    print("\n1. Products with HIGH rating count but LOW price:")
    high_rating_low_price = find_products_by_comparison(
        data, rating_count_vs='high', price_vs='low'
    )
    
    for i, product in enumerate(high_rating_low_price[:5]):
        print(f"  {i+1}. {product.get('title', 'Unknown')[:60]}...")
        print(f"     Cluster: {product.get('cluster_id')}")
        print(f"     Rating Count: {product.get('rating_count')} (vs cluster avg: {product.get('cluster_avg_rating_count', 0):.1f})")
        selling_price = product.get('selling_price', 0) or 0
        cluster_avg_price = product.get('cluster_avg_price', 0) or 0
        print(f"     Price: {selling_price:,.0f} (vs cluster avg: {cluster_avg_price:,.0f})")
        print(f"     Rating Count Diff: {product.get('rating_count_diff_percent', 0):+.1f}%")
        print(f"     Price Diff: {product.get('price_diff_percent', 0):+.1f}%")
        print()
    
    # Example 2: Analyze a specific cluster
    print("\n2. Analysis of Cluster 23 (Xiaomi phones):")
    cluster_analysis = analyze_cluster_performance(data, 23)
    if 'error' not in cluster_analysis:
        print(f"  Total products: {cluster_analysis['total_products']}")
        print(f"  Cluster avg rating count: {cluster_analysis['cluster_avg_rating_count']:.1f}")
        print(f"  Cluster avg rating: {cluster_analysis['cluster_avg_rating_average']:.1f}")
        print(f"  Cluster avg price: {cluster_analysis['cluster_avg_price']:,.0f}")
        print(f"  Rating count comparison: {cluster_analysis['rating_count_comparison']}")
        print(f"  Rating average comparison: {cluster_analysis['rating_average_comparison']}")
        print(f"  Price comparison: {cluster_analysis['price_comparison']}")
    
    # Example 3: Top performers by rating average
    print("\n3. Top 5 products with HIGH rating average vs their cluster:")
    top_rated = get_top_performers(data, 'rating_average', 5)
    
    for i, product in enumerate(top_rated):
        print(f"  {i+1}. {product.get('title', 'Unknown')[:60]}...")
        print(f"     Cluster: {product.get('cluster_id')}")
        print(f"     Rating: {product.get('rating_average', 0):.1f} (vs cluster avg: {product.get('cluster_avg_rating_average', 0):.1f})")
        print(f"     Rating Diff: {product.get('rating_average_diff_percent', 0):+.1f}%")
        print()
    
    # Example 4: Find bargain products (high rating, low price)
    print("\n4. Bargain products (HIGH rating average, LOW price vs cluster):")
    bargains = find_products_by_comparison(
        data, rating_average_vs='high', price_vs='low'
    )
    
    # Sort by rating average
    bargains.sort(key=lambda x: x.get('rating_average', 0), reverse=True)
    
    for i, product in enumerate(bargains[:5]):
        print(f"  {i+1}. {product.get('title', 'Unknown')[:60]}...")
        print(f"     Cluster: {product.get('cluster_id')}")
        print(f"     Rating: {product.get('rating_average', 0):.1f} (vs cluster avg: {product.get('cluster_avg_rating_average', 0):.1f})")
        selling_price = product.get('selling_price', 0) or 0
        cluster_avg_price = product.get('cluster_avg_price', 0) or 0
        print(f"     Price: {selling_price:,.0f} (vs cluster avg: {cluster_avg_price:,.0f})")
        print(f"     Rating Diff: {product.get('rating_average_diff_percent', 0):+.1f}%")
        print(f"     Price Diff: {product.get('price_diff_percent', 0):+.1f}%")
        print()

if __name__ == "__main__":
    main()
