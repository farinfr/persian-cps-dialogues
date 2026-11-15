#!/usr/bin/env python3
"""
Random Batch Sampling Script - 500 Products with 5 Features Each
================================================================

This script performs completely random sampling (no probabilities):
- 500 products (random selection)
- 5 features per product (random selection, not weighted)
- Includes feature sampling order
"""

import json
import os
from product_feature_sampler import ProductFeatureSampler

def main():
    """Run random batch sampling for 500 products"""
    print("=" * 70)
    print("RANDOM BATCH SAMPLING: 500 Products, 5 Features Each (NO PROBABILITIES)")
    print("=" * 70)
    
    # Configuration
    products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    sampling_probabilities_file = "cluster_sampling_probability_results_custome/cluster_sampling_probabilities.json"
    output_dir = "sampled_products"
    output_file = f"{output_dir}/random_batch_500_products_5_features.json"
    
    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = ProductFeatureSampler(products_file, sampling_probabilities_file)
    
    try:
        # Load data
        print("Loading data...")
        sampler.load_data()
        
        # Random batch sample: 500 products, 5 features each (completely random)
        print(f"\nRandomly sampling 500 products with 5 features each...")
        print("(No probability weights - pure random selection)\n")
        
        samples = sampler.batch_sample_random(
            num_samples=500,
            num_features=5,
            unique_products=True
        )
        
        print(f"\nSuccessfully sampled {len(samples)} products")
        
        # Show statistics
        clusters = {}
        total_features = 0
        errors = 0
        for sample in samples:
            if 'error' not in sample:
                cluster_id = sample['cluster_id']
                clusters[cluster_id] = clusters.get(cluster_id, 0) + 1
                total_features += len(sample.get('sampled_features', {}))
            else:
                errors += 1
        
        print(f"\nStatistics:")
        print(f"  Total products sampled: {len(samples)}")
        print(f"  Successful samples: {len(samples) - errors}")
        print(f"  Errors: {errors}")
        print(f"  Total features across all products: {total_features}")
        print(f"  Average features per product: {total_features / max(len(samples) - errors, 1):.2f}")
        print(f"  Unique clusters represented: {len(clusters)}")
        
        # Show top clusters
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  Top 10 clusters by product count:")
        for i, (cluster_id, count) in enumerate(sorted_clusters, 1):
            print(f"    {i:2d}. Cluster {cluster_id}: {count} products")
        
        # Show sample of results with order
        print(f"\nSample Results (first 3 products):")
        print("=" * 70)
        shown = 0
        for sample in samples:
            if 'error' not in sample and shown < 3:
                shown += 1
                print(f"\nProduct {shown}:")
                print(f"  Title: {sample['product']['title'][:60]}...")
                print(f"  Cluster: {sample['cluster_id']}")
                print(f"  Features sampled: {len(sample['sampled_features'])}")
                print(f"  Feature sampling order (RANDOM):")
                order = sample.get('feature_sampling_order', [])
                for idx, feature_name in enumerate(order, 1):
                    value = sample['sampled_features'].get(feature_name, 'N/A')
                    # Truncate long values for display
                    display_value = value[:60] + "..." if len(str(value)) > 60 else value
                    print(f"    {idx}. {feature_name}")
                    print(f"       Value: {display_value}")
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        sampler.save_samples(samples, output_file)
        
        print("\n" + "=" * 70)
        print("RANDOM BATCH SAMPLING COMPLETE!")
        print("=" * 70)
        print(f"\n📁 Results saved to: {output_file}")
        print(f"📊 Total products: {len(samples)}")
        print(f"🔢 Features per product: 5 (randomly selected)")
        print(f"📋 Feature sampling order included for each product")
        print(f"⚡ Sampling method: Pure random (no probability weights)")
        
    except Exception as e:
        print(f"\nError during random batch sampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
