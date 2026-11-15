#!/usr/bin/env python3
"""
Batch Sampling Script - 500 Products with 5 Features Each
=========================================================

This script performs batch sampling:
- 500 products
- 5 features per product
- Includes feature sampling order
"""

import json
import os
from product_feature_sampler import ProductFeatureSampler

def main():
    """Run batch sampling for 500 products"""
    print("=" * 70)
    print("BATCH SAMPLING: 500 Products, 5 Features Each")
    print("=" * 70)
    
    # Configuration
    products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    sampling_probabilities_file = "cluster_sampling_probability_results_custome/cluster_sampling_probabilities.json"
    output_dir = "sampled_products"
    output_file = f"{output_dir}/batch_500_products_5_features.json"
    
    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = ProductFeatureSampler(products_file, sampling_probabilities_file)
    
    try:
        # Load data
        print("Loading data...")
        sampler.load_data()
        
        # Batch sample: 500 products, 5 features each
        print(f"\nSampling 500 products with 5 features each...")
        print("This may take a few minutes...\n")
        
        samples = sampler.batch_sample(
            num_samples=500,
            num_features=5,
            unique_products=True,
            min_probability=0.001  # Only sample features with normalized_probability >= 0.001
        )
        
        print(f"\nSuccessfully sampled {len(samples)} products")
        
        # Show statistics
        clusters = {}
        total_features = 0
        successful_samples = 0
        failed_samples = 0
        
        for sample in samples:
            if 'error' not in sample:
                cluster_id = sample['cluster_id']
                clusters[cluster_id] = clusters.get(cluster_id, 0) + 1
                feature_count = len(sample.get('sampled_features', {}))
                total_features += feature_count
                if feature_count == 5:  # Successfully sampled all 5 features
                    successful_samples += 1
                else:
                    failed_samples += 1
            else:
                failed_samples += 1
        
        print(f"\nStatistics:")
        print(f"  Total products sampled: {len(samples)}")
        print(f"  Successfully sampled (5 features): {successful_samples}")
        print(f"  Partially sampled (< 5 features): {failed_samples}")
        print(f"  Total features across all products: {total_features}")
        if successful_samples + failed_samples > 0:
            print(f"  Average features per product: {total_features / (successful_samples + failed_samples):.2f}")
        print(f"  Unique clusters represented: {len(clusters)}")
        
        # Show top clusters
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Top 10 clusters by product count:")
        for i, (cluster_id, count) in enumerate(sorted_clusters, 1):
            print(f"    {i:2d}. Cluster {cluster_id}: {count} products")
        
        # Show sample of results with order
        print(f"\nSample Results (first 3 products):")
        print("=" * 70)
        for i, sample in enumerate(samples[:3], 1):
            if 'error' not in sample:
                print(f"\nProduct {i}:")
                print(f"  Title: {sample['product']['title'][:60]}...")
                print(f"  Cluster: {sample['cluster_id']}")
                print(f"  Features sampled: {len(sample['sampled_features'])}")
                print(f"  Feature sampling order:")
                order = sample.get('feature_sampling_order', [])
                for idx, feature_name in enumerate(order, 1):
                    value = sample['sampled_features'].get(feature_name, 'N/A')
                    print(f"    {idx}. {feature_name}: {value}")
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        sampler.save_samples(samples, output_file)
        
        print("\n" + "=" * 70)
        print("BATCH SAMPLING COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Results saved to: {output_file}")
        print(f"📊 Total products: {len(samples)}")
        print(f"🔢 Features per product: 5")
        print(f"📋 Feature sampling order included for each product")
        
    except Exception as e:
        print(f"\nError during batch sampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
