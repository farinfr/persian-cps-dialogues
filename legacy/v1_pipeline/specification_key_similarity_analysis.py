#!/usr/bin/env python3
"""
Product Specification Key Similarity Analysis
===========================================

This script calculates the similarity of specification keys (names) between products
within each cluster. It analyzes how similar products are in terms of their
specification structure (what specifications they have) rather than their values.

Features:
- Extracts specification keys from products
- Calculates key-based similarity between products within clusters
- Uses Jaccard similarity for key sets
- Generates cluster key similarity reports
- Identifies most/least similar clusters based on specification structure

Example:
json1 = {"key1": "value1", "key2": "value2"}
json2 = {"key1": "value11", "key3": "value3"}
set1 = {"key1", "key2"}
set2 = {"key1", "key3"}
Jaccard similarity = |intersection| / |union| = 1 / 3 = 0.333
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Set
import os
import matplotlib.pyplot as plt
import seaborn as sns

class SpecificationKeySimilarityAnalyzer:
    def __init__(self, enhanced_dataset_path: str):
        self.enhanced_dataset_path = enhanced_dataset_path
        self.products_data = None
        self.cluster_similarities = {}
        self.similarity_results = {}
        
    def load_enhanced_dataset(self):
        """Load the enhanced dataset with cluster information"""
        print("Loading enhanced dataset...")
        with open(self.enhanced_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.products_data = data['products']
        print(f"Loaded {len(self.products_data)} products")
        return self.products_data
    
    def extract_specification_keys(self, product: Dict) -> Set[str]:
        """Extract specification keys (names) as a set"""
        specs = product.get('specifications', {})
        if not specs:
            return set()
        
        keys = set()
        
        for category, category_specs in specs.items():
            if not category_specs:
                continue
                
            for key, value in category_specs.items():
                if value:  # Only include keys with non-empty values
                    # Create unique key identifier
                    full_key = f"{category}_{key}"
                    keys.add(full_key)
        
        return keys
    
    def calculate_jaccard_similarity(self, keys1: Set[str], keys2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets of keys"""
        if not keys1 and not keys2:
            return 1.0  # Both empty sets are identical
        if not keys1 or not keys2:
            return 0.0  # One empty, one non-empty
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        return len(intersection) / len(union)
    
    def analyze_cluster_key_similarities(self):
        """Analyze key similarities within each cluster"""
        print("Analyzing specification key similarities within clusters...")
        
        # Group products by cluster
        cluster_products = defaultdict(list)
        for product in self.products_data:
            cluster_id = product.get('cluster_id')
            if cluster_id is not None:
                cluster_products[cluster_id].append(product)
        
        print(f"Found {len(cluster_products)} clusters")
        
        cluster_stats = {}
        
        for cluster_id, products in cluster_products.items():
            if len(products) < 2:
                # Skip clusters with only one product
                cluster_stats[cluster_id] = {
                    'product_count': len(products),
                    'avg_key_similarity': 0.0,
                    'similarity_std': 0.0,
                    'max_similarity': 0.0,
                    'min_similarity': 0.0,
                    'avg_keys_per_product': 0.0,
                    'total_unique_keys': 0,
                    'key_coverage': 0.0
                }
                continue
            
            print(f"Analyzing cluster {cluster_id} with {len(products)} products...")
            
            # Extract specification keys for each product
            product_keys = []
            for product in products:
                keys = self.extract_specification_keys(product)
                product_keys.append(keys)
            
            # Calculate Jaccard similarities between all pairs
            similarities = []
            for i in range(len(product_keys)):
                for j in range(i + 1, len(product_keys)):
                    similarity = self.calculate_jaccard_similarity(product_keys[i], product_keys[j])
                    similarities.append(similarity)
            
            # Calculate key statistics
            all_keys = set().union(*product_keys)
            avg_keys_per_product = np.mean([len(keys) for keys in product_keys])
            
            # Calculate key coverage (how many products have each key)
            key_coverage_scores = []
            for key in all_keys:
                products_with_key = sum(1 for keys in product_keys if key in keys)
                coverage = products_with_key / len(products)
                key_coverage_scores.append(coverage)
            
            avg_key_coverage = np.mean(key_coverage_scores) if key_coverage_scores else 0.0
            
            # Store cluster statistics
            cluster_stats[cluster_id] = {
                'product_count': len(products),
                'avg_key_similarity': np.mean(similarities) if similarities else 0.0,
                'similarity_std': np.std(similarities) if similarities else 0.0,
                'max_similarity': np.max(similarities) if similarities else 0.0,
                'min_similarity': np.min(similarities) if similarities else 0.0,
                'avg_keys_per_product': avg_keys_per_product,
                'total_unique_keys': len(all_keys),
                'key_coverage': avg_key_coverage,
                'similarities': similarities,
                'product_keys': product_keys
            }
        
        self.cluster_similarities = cluster_stats
        return cluster_stats
    
    def generate_key_similarity_statistics(self) -> Dict:
        """Generate overall key similarity statistics"""
        print("Generating key similarity statistics...")
        
        if not self.cluster_similarities:
            return {}
        
        # Extract metrics
        key_similarities = [stats['avg_key_similarity'] for stats in self.cluster_similarities.values()]
        product_counts = [stats['product_count'] for stats in self.cluster_similarities.values()]
        avg_keys_per_product = [stats['avg_keys_per_product'] for stats in self.cluster_similarities.values()]
        key_coverages = [stats['key_coverage'] for stats in self.cluster_similarities.values()]
        
        # Calculate overall statistics
        stats = {
            'total_clusters': len(self.cluster_similarities),
            'clusters_with_multiple_products': len([c for c in self.cluster_similarities.values() if c['product_count'] > 1]),
            'avg_cluster_size': np.mean(product_counts),
            'median_cluster_size': np.median(product_counts),
            'min_cluster_size': np.min(product_counts),
            'max_cluster_size': np.max(product_counts),
            
            # Key similarity statistics
            'avg_key_similarity': np.mean(key_similarities),
            'median_key_similarity': np.median(key_similarities),
            'std_key_similarity': np.std(key_similarities),
            'min_key_similarity': np.min(key_similarities),
            'max_key_similarity': np.max(key_similarities),
            
            # Key structure statistics
            'avg_keys_per_product': np.mean(avg_keys_per_product),
            'median_keys_per_product': np.median(avg_keys_per_product),
            'std_keys_per_product': np.std(avg_keys_per_product),
            'min_keys_per_product': np.min(avg_keys_per_product),
            'max_keys_per_product': np.max(avg_keys_per_product),
            
            # Key coverage statistics
            'avg_key_coverage': np.mean(key_coverages),
            'median_key_coverage': np.median(key_coverages),
            'std_key_coverage': np.std(key_coverages),
            'min_key_coverage': np.min(key_coverages),
            'max_key_coverage': np.max(key_coverages)
        }
        
        # Find top and bottom clusters
        sorted_by_similarity = sorted(self.cluster_similarities.items(), 
                                    key=lambda x: x[1]['avg_key_similarity'], reverse=True)
        
        # Convert to JSON-serializable format
        stats['top_5_key_similarity_clusters'] = [
            {
                'cluster_id': cluster_id,
                'avg_key_similarity': cluster_stats['avg_key_similarity'],
                'product_count': cluster_stats['product_count']
            }
            for cluster_id, cluster_stats in sorted_by_similarity[:5]
        ]
        stats['bottom_5_key_similarity_clusters'] = [
            {
                'cluster_id': cluster_id,
                'avg_key_similarity': cluster_stats['avg_key_similarity'],
                'product_count': cluster_stats['product_count']
            }
            for cluster_id, cluster_stats in sorted_by_similarity[-5:]
        ]
        
        self.similarity_results = stats
        return stats
    
    def generate_key_similarity_visualizations(self, output_dir: str = "key_similarity_results"):
        """Generate visualizations for key similarity analysis"""
        print("Generating key similarity visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Key Similarity Distribution
        key_similarities = [stats['avg_key_similarity'] for stats in self.cluster_similarities.values()]
        
        plt.figure(figsize=(12, 8))
        plt.hist(key_similarities, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Average Key Similarity Across Clusters')
        plt.xlabel('Average Key Similarity (Jaccard)')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/01_key_similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Keys per Product Distribution
        keys_per_product = [stats['avg_keys_per_product'] for stats in self.cluster_similarities.values()]
        
        plt.figure(figsize=(12, 8))
        plt.hist(keys_per_product, bins=30, alpha=0.7, edgecolor='black', color='green')
        plt.title('Distribution of Average Keys per Product Across Clusters')
        plt.xlabel('Average Keys per Product')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/02_keys_per_product_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Key Coverage Distribution
        key_coverages = [stats['key_coverage'] for stats in self.cluster_similarities.values()]
        
        plt.figure(figsize=(12, 8))
        plt.hist(key_coverages, bins=30, alpha=0.7, edgecolor='black', color='orange')
        plt.title('Distribution of Key Coverage Across Clusters')
        plt.xlabel('Average Key Coverage')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/03_key_coverage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Top 20 Most Similar Clusters
        top_clusters = sorted(self.cluster_similarities.items(), 
                            key=lambda x: x[1]['avg_key_similarity'], reverse=True)[:20]
        
        cluster_ids = [str(item[0]) for item in top_clusters]
        similarities = [item[1]['avg_key_similarity'] for item in top_clusters]
        
        plt.figure(figsize=(15, 8))
        plt.bar(cluster_ids, similarities, alpha=0.7)
        plt.title('Top 20 Most Similar Clusters (Key Similarity)')
        plt.xlabel('Cluster ID')
        plt.ylabel('Average Key Similarity (Jaccard)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/04_top_20_key_similar_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Key Similarity vs Cluster Size
        cluster_sizes = [stats['product_count'] for stats in self.cluster_similarities.values()]
        
        plt.figure(figsize=(12, 8))
        plt.scatter(cluster_sizes, key_similarities, alpha=0.6)
        plt.xlabel('Cluster Size (Number of Products)')
        plt.ylabel('Average Key Similarity (Jaccard)')
        plt.title('Key Similarity vs Cluster Size')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/05_key_similarity_vs_cluster_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Key similarity visualizations saved to {output_dir}/")
    
    def save_key_similarity_results(self, output_dir: str = "key_similarity_results"):
        """Save key similarity analysis results"""
        print("Saving key similarity analysis results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert NumPy types to Python native types
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
            elif isinstance(obj, set):
                return list(obj)  # Convert sets to lists for JSON serialization
            elif isinstance(obj, tuple):
                return list(obj)  # Convert tuples to lists for JSON serialization
            else:
                return obj
        
        # Save cluster similarities (simplified for JSON)
        cluster_similarities_simplified = {}
        for cluster_id, stats in self.cluster_similarities.items():
            cluster_similarities_simplified[cluster_id] = {
                'product_count': stats['product_count'],
                'avg_key_similarity': stats['avg_key_similarity'],
                'similarity_std': stats['similarity_std'],
                'max_similarity': stats['max_similarity'],
                'min_similarity': stats['min_similarity'],
                'avg_keys_per_product': stats['avg_keys_per_product'],
                'total_unique_keys': stats['total_unique_keys'],
                'key_coverage': stats['key_coverage']
            }
        
        cluster_file = f'{output_dir}/cluster_key_similarities.json'
        with open(cluster_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(cluster_similarities_simplified), f, ensure_ascii=False, indent=2)
        
        # Save overall statistics
        stats_file = f'{output_dir}/key_similarity_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(self.similarity_results), f, ensure_ascii=False, indent=2)
        
        # Create summary report
        self._create_key_similarity_summary_report(output_dir)
        
        print(f"Key similarity results saved to {output_dir}/")
    
    def _create_key_similarity_summary_report(self, output_dir: str):
        """Create a human-readable summary report for key similarity analysis"""
        report_file = f'{output_dir}/key_similarity_analysis_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SPECIFICATION KEY SIMILARITY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            if not self.similarity_results:
                f.write("No key similarity results available.\n")
                return
            
            stats = self.similarity_results
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total Clusters: {stats['total_clusters']}\n")
            f.write(f"  Clusters with Multiple Products: {stats['clusters_with_multiple_products']}\n")
            f.write(f"  Average Cluster Size: {stats['avg_cluster_size']:.1f} products\n")
            f.write(f"  Median Cluster Size: {stats['median_cluster_size']:.1f} products\n")
            f.write(f"  Cluster Size Range: {stats['min_cluster_size']} - {stats['max_cluster_size']} products\n\n")
            
            f.write("KEY SIMILARITY STATISTICS:\n")
            f.write(f"  Average Similarity: {stats['avg_key_similarity']:.4f}\n")
            f.write(f"  Median Similarity: {stats['median_key_similarity']:.4f}\n")
            f.write(f"  Standard Deviation: {stats['std_key_similarity']:.4f}\n")
            f.write(f"  Range: {stats['min_key_similarity']:.4f} - {stats['max_key_similarity']:.4f}\n\n")
            
            f.write("KEY STRUCTURE STATISTICS:\n")
            f.write(f"  Average Keys per Product: {stats['avg_keys_per_product']:.1f}\n")
            f.write(f"  Median Keys per Product: {stats['median_keys_per_product']:.1f}\n")
            f.write(f"  Standard Deviation: {stats['std_keys_per_product']:.1f}\n")
            f.write(f"  Range: {stats['min_keys_per_product']:.1f} - {stats['max_keys_per_product']:.1f}\n\n")
            
            f.write("KEY COVERAGE STATISTICS:\n")
            f.write(f"  Average Coverage: {stats['avg_key_coverage']:.4f}\n")
            f.write(f"  Median Coverage: {stats['median_key_coverage']:.4f}\n")
            f.write(f"  Standard Deviation: {stats['std_key_coverage']:.4f}\n")
            f.write(f"  Range: {stats['min_key_coverage']:.4f} - {stats['max_key_coverage']:.4f}\n\n")
            
            f.write("TOP 5 MOST SIMILAR CLUSTERS (Key Similarity):\n")
            for cluster_info in stats['top_5_key_similarity_clusters']:
                f.write(f"  Cluster {cluster_info['cluster_id']}: {cluster_info['avg_key_similarity']:.4f} "
                       f"({cluster_info['product_count']} products)\n")
            
            f.write("\nTOP 5 LEAST SIMILAR CLUSTERS (Key Similarity):\n")
            for cluster_info in stats['bottom_5_key_similarity_clusters']:
                f.write(f"  Cluster {cluster_info['cluster_id']}: {cluster_info['avg_key_similarity']:.4f} "
                       f"({cluster_info['product_count']} products)\n")
        
        print(f"Key similarity summary report saved to {report_file}")

def main():
    """Main function to run the specification key similarity analysis"""
    print("Starting Specification Key Similarity Analysis...")
    print("=" * 60)
    
    # Configuration
    enhanced_dataset_path = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    output_dir = "key_similarity_results"
    
    # Initialize analyzer
    analyzer = SpecificationKeySimilarityAnalyzer(enhanced_dataset_path)
    
    try:
        # Load enhanced dataset
        analyzer.load_enhanced_dataset()
        
        # Analyze cluster key similarities
        cluster_stats = analyzer.analyze_cluster_key_similarities()
        
        # Generate key similarity statistics
        overall_stats = analyzer.generate_key_similarity_statistics()
        
        # Generate visualizations
        analyzer.generate_key_similarity_visualizations(output_dir)
        
        # Save results
        analyzer.save_key_similarity_results(output_dir)
        
        print("\n" + "=" * 60)
        print("Specification Key Similarity Analysis Complete!")
        print("=" * 60)
        
        # Print summary
        if overall_stats:
            print(f"\n📊 KEY SIMILARITY ANALYSIS SUMMARY:")
            print(f"  Total Clusters: {overall_stats['total_clusters']}")
            print(f"  Clusters with Multiple Products: {overall_stats['clusters_with_multiple_products']}")
            print(f"  Average Cluster Size: {overall_stats['avg_cluster_size']:.1f} products")
            
            print(f"\n🔑 KEY SIMILARITY:")
            print(f"  Average: {overall_stats['avg_key_similarity']:.4f}")
            print(f"  Range: {overall_stats['min_key_similarity']:.4f} - {overall_stats['max_key_similarity']:.4f}")
            
            print(f"\n📋 KEY STRUCTURE:")
            print(f"  Average Keys per Product: {overall_stats['avg_keys_per_product']:.1f}")
            print(f"  Range: {overall_stats['min_keys_per_product']:.1f} - {overall_stats['max_keys_per_product']:.1f}")
            
            print(f"\n📊 KEY COVERAGE:")
            print(f"  Average: {overall_stats['avg_key_coverage']:.4f}")
            print(f"  Range: {overall_stats['min_key_coverage']:.4f} - {overall_stats['max_key_coverage']:.4f}")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"📈 Visualizations: {output_dir}/*.png")
        print(f"📄 Summary report: {output_dir}/key_similarity_analysis_summary.txt")
        
    except Exception as e:
        print(f"Error during key similarity analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
