#!/usr/bin/env python3
"""
Random Cluster Key Similarity Analysis
=====================================

This script randomly clusters products and then calculates the key similarity
of specifications within each random cluster. This helps us understand how
random clustering compares to our optimized clustering results.

Features:
- Randomly assigns products to clusters
- Calculates key similarity within each random cluster
- Compares results with optimized clustering
- Generates statistical analysis and visualizations
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Set
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns

class RandomClusterKeySimilarityAnalyzer:
    def __init__(self, enhanced_dataset_path: str):
        self.enhanced_dataset_path = enhanced_dataset_path
        self.products_data = None
        self.random_cluster_similarities = {}
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
    
    def create_random_clusters(self, num_clusters: int = 170, seed: int = 42):
        """Create random clusters by randomly assigning products to clusters"""
        print(f"Creating {num_clusters} random clusters...")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Get total number of products
        total_products = len(self.products_data)
        
        # Randomly assign cluster IDs to products
        random_cluster_assignments = np.random.randint(0, num_clusters, total_products)
        
        # Group products by random cluster
        random_cluster_products = defaultdict(list)
        for i, product in enumerate(self.products_data):
            cluster_id = random_cluster_assignments[i]
            random_cluster_products[cluster_id].append(product)
        
        print(f"Created {len(random_cluster_products)} random clusters")
        
        # Print cluster size distribution
        cluster_sizes = [len(products) for products in random_cluster_products.values()]
        print(f"Random cluster sizes - Min: {min(cluster_sizes)}, Max: {max(cluster_sizes)}, Avg: {np.mean(cluster_sizes):.1f}")
        
        return random_cluster_products
    
    def analyze_random_cluster_key_similarities(self, random_clusters: Dict[int, List[Dict]]):
        """Analyze key similarities within each random cluster"""
        print("Analyzing specification key similarities within random clusters...")
        
        cluster_stats = {}
        
        for cluster_id, products in random_clusters.items():
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
            
            if cluster_id % 20 == 0:  # Progress indicator
                print(f"Analyzing random cluster {cluster_id} with {len(products)} products...")
            
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
        
        self.random_cluster_similarities = cluster_stats
        return cluster_stats
    
    def generate_random_similarity_statistics(self) -> Dict:
        """Generate overall random similarity statistics"""
        print("Generating random similarity statistics...")
        
        if not self.random_cluster_similarities:
            return {}
        
        # Extract metrics
        key_similarities = [stats['avg_key_similarity'] for stats in self.random_cluster_similarities.values()]
        product_counts = [stats['product_count'] for stats in self.random_cluster_similarities.values()]
        avg_keys_per_product = [stats['avg_keys_per_product'] for stats in self.random_cluster_similarities.values()]
        key_coverages = [stats['key_coverage'] for stats in self.random_cluster_similarities.values()]
        
        # Calculate overall statistics
        stats = {
            'total_clusters': len(self.random_cluster_similarities),
            'clusters_with_multiple_products': len([c for c in self.random_cluster_similarities.values() if c['product_count'] > 1]),
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
        sorted_by_similarity = sorted(self.random_cluster_similarities.items(), 
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
    
    def compare_with_optimized_clustering(self, output_dir: str = "random_cluster_results"):
        """Compare random clustering results with optimized clustering"""
        print("Comparing with optimized clustering results...")
        
        try:
            # Load optimized clustering results
            with open('key_similarity_results/key_similarity_statistics.json', 'r', encoding='utf-8') as f:
                optimized_stats = json.load(f)
            
            # Create comparison
            comparison = {
                'random_clustering': {
                    'avg_key_similarity': self.similarity_results['avg_key_similarity'],
                    'median_key_similarity': self.similarity_results['median_key_similarity'],
                    'std_key_similarity': self.similarity_results['std_key_similarity'],
                    'min_key_similarity': self.similarity_results['min_key_similarity'],
                    'max_key_similarity': self.similarity_results['max_key_similarity'],
                    'avg_keys_per_product': self.similarity_results['avg_keys_per_product'],
                    'avg_key_coverage': self.similarity_results['avg_key_coverage']
                },
                'optimized_clustering': {
                    'avg_key_similarity': optimized_stats['avg_key_similarity'],
                    'median_key_similarity': optimized_stats['median_key_similarity'],
                    'std_key_similarity': optimized_stats['std_key_similarity'],
                    'min_key_similarity': optimized_stats['min_key_similarity'],
                    'max_key_similarity': optimized_stats['max_key_similarity'],
                    'avg_keys_per_product': optimized_stats['avg_keys_per_product'],
                    'avg_key_coverage': optimized_stats['avg_key_coverage']
                }
            }
            
            # Calculate improvement
            improvement = {
                'similarity_improvement': optimized_stats['avg_key_similarity'] - self.similarity_results['avg_key_similarity'],
                'similarity_improvement_pct': ((optimized_stats['avg_key_similarity'] - self.similarity_results['avg_key_similarity']) / self.similarity_results['avg_key_similarity']) * 100,
                'coverage_improvement': optimized_stats['avg_key_coverage'] - self.similarity_results['avg_key_coverage'],
                'coverage_improvement_pct': ((optimized_stats['avg_key_coverage'] - self.similarity_results['avg_key_coverage']) / self.similarity_results['avg_key_coverage']) * 100
            }
            
            comparison['improvement'] = improvement
            
            # Save comparison
            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/clustering_comparison.json', 'w', encoding='utf-8') as f:
                json.dump(comparison, f, ensure_ascii=False, indent=2)
            
            return comparison
            
        except FileNotFoundError:
            print("Optimized clustering results not found. Skipping comparison.")
            return None
    
    def generate_random_similarity_visualizations(self, output_dir: str = "random_cluster_results"):
        """Generate visualizations for random similarity analysis"""
        print("Generating random similarity visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Random Key Similarity Distribution
        key_similarities = [stats['avg_key_similarity'] for stats in self.random_cluster_similarities.values()]
        
        plt.figure(figsize=(12, 8))
        plt.hist(key_similarities, bins=30, alpha=0.7, edgecolor='black', color='red')
        plt.title('Distribution of Average Key Similarity Across Random Clusters')
        plt.xlabel('Average Key Similarity (Jaccard)')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/01_random_key_similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Random vs Optimized Comparison
        try:
            with open('key_similarity_results/key_similarity_statistics.json', 'r', encoding='utf-8') as f:
                optimized_stats = json.load(f)
            
            # Load optimized similarities
            with open('key_similarity_results/cluster_key_similarities.json', 'r', encoding='utf-8') as f:
                optimized_clusters = json.load(f)
            
            optimized_similarities = [cluster['avg_key_similarity'] for cluster in optimized_clusters.values()]
            
            plt.figure(figsize=(15, 8))
            plt.hist([key_similarities, optimized_similarities], bins=30, alpha=0.7, 
                    label=['Random Clustering', 'Optimized Clustering'], 
                    color=['red', 'blue'], edgecolor='black')
            plt.title('Key Similarity Distribution: Random vs Optimized Clustering')
            plt.xlabel('Average Key Similarity (Jaccard)')
            plt.ylabel('Number of Clusters')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{output_dir}/02_random_vs_optimized_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except FileNotFoundError:
            print("Optimized clustering data not found. Skipping comparison plot.")
        
        # 3. Top 20 Random Clusters
        top_clusters = sorted(self.random_cluster_similarities.items(), 
                            key=lambda x: x[1]['avg_key_similarity'], reverse=True)[:20]
        
        cluster_ids = [str(item[0]) for item in top_clusters]
        similarities = [item[1]['avg_key_similarity'] for item in top_clusters]
        
        plt.figure(figsize=(15, 8))
        plt.bar(cluster_ids, similarities, alpha=0.7, color='red')
        plt.title('Top 20 Most Similar Random Clusters (Key Similarity)')
        plt.xlabel('Random Cluster ID')
        plt.ylabel('Average Key Similarity (Jaccard)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/03_top_20_random_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Random Key Similarity vs Cluster Size
        cluster_sizes = [stats['product_count'] for stats in self.random_cluster_similarities.values()]
        
        plt.figure(figsize=(12, 8))
        plt.scatter(cluster_sizes, key_similarities, alpha=0.6, color='red')
        plt.xlabel('Random Cluster Size (Number of Products)')
        plt.ylabel('Average Key Similarity (Jaccard)')
        plt.title('Random Key Similarity vs Cluster Size')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/04_random_similarity_vs_cluster_size.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Random similarity visualizations saved to {output_dir}/")
    
    def save_random_similarity_results(self, output_dir: str = "random_cluster_results"):
        """Save random similarity analysis results"""
        print("Saving random similarity analysis results...")
        
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
        for cluster_id, stats in self.random_cluster_similarities.items():
            # Convert cluster_id to string for JSON serialization
            cluster_similarities_simplified[str(cluster_id)] = {
                'product_count': stats['product_count'],
                'avg_key_similarity': stats['avg_key_similarity'],
                'similarity_std': stats['similarity_std'],
                'max_similarity': stats['max_similarity'],
                'min_similarity': stats['min_similarity'],
                'avg_keys_per_product': stats['avg_keys_per_product'],
                'total_unique_keys': stats['total_unique_keys'],
                'key_coverage': stats['key_coverage']
            }
        
        cluster_file = f'{output_dir}/random_cluster_key_similarities.json'
        with open(cluster_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(cluster_similarities_simplified), f, ensure_ascii=False, indent=2)
        
        # Save overall statistics
        stats_file = f'{output_dir}/random_key_similarity_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(self.similarity_results), f, ensure_ascii=False, indent=2)
        
        # Create summary report
        self._create_random_similarity_summary_report(output_dir)
        
        print(f"Random similarity results saved to {output_dir}/")
    
    def _create_random_similarity_summary_report(self, output_dir: str):
        """Create a human-readable summary report for random similarity analysis"""
        report_file = f'{output_dir}/random_key_similarity_analysis_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("RANDOM CLUSTER KEY SIMILARITY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            if not self.similarity_results:
                f.write("No random similarity results available.\n")
                return
            
            stats = self.similarity_results
            
            f.write("RANDOM CLUSTERING STATISTICS:\n")
            f.write(f"  Total Random Clusters: {stats['total_clusters']}\n")
            f.write(f"  Clusters with Multiple Products: {stats['clusters_with_multiple_products']}\n")
            f.write(f"  Average Cluster Size: {stats['avg_cluster_size']:.1f} products\n")
            f.write(f"  Median Cluster Size: {stats['median_cluster_size']:.1f} products\n")
            f.write(f"  Cluster Size Range: {stats['min_cluster_size']} - {stats['max_cluster_size']} products\n\n")
            
            f.write("RANDOM KEY SIMILARITY STATISTICS:\n")
            f.write(f"  Average Similarity: {stats['avg_key_similarity']:.4f}\n")
            f.write(f"  Median Similarity: {stats['median_key_similarity']:.4f}\n")
            f.write(f"  Standard Deviation: {stats['std_key_similarity']:.4f}\n")
            f.write(f"  Range: {stats['min_key_similarity']:.4f} - {stats['max_key_similarity']:.4f}\n\n")
            
            f.write("RANDOM KEY STRUCTURE STATISTICS:\n")
            f.write(f"  Average Keys per Product: {stats['avg_keys_per_product']:.1f}\n")
            f.write(f"  Median Keys per Product: {stats['median_keys_per_product']:.1f}\n")
            f.write(f"  Standard Deviation: {stats['std_keys_per_product']:.1f}\n")
            f.write(f"  Range: {stats['min_keys_per_product']:.1f} - {stats['max_keys_per_product']:.1f}\n\n")
            
            f.write("RANDOM KEY COVERAGE STATISTICS:\n")
            f.write(f"  Average Coverage: {stats['avg_key_coverage']:.4f}\n")
            f.write(f"  Median Coverage: {stats['median_key_coverage']:.4f}\n")
            f.write(f"  Standard Deviation: {stats['std_key_coverage']:.4f}\n")
            f.write(f"  Range: {stats['min_key_coverage']:.4f} - {stats['max_key_coverage']:.4f}\n\n")
            
            f.write("TOP 5 MOST SIMILAR RANDOM CLUSTERS:\n")
            for cluster_info in stats['top_5_key_similarity_clusters']:
                f.write(f"  Random Cluster {cluster_info['cluster_id']}: {cluster_info['avg_key_similarity']:.4f} "
                       f"({cluster_info['product_count']} products)\n")
            
            f.write("\nTOP 5 LEAST SIMILAR RANDOM CLUSTERS:\n")
            for cluster_info in stats['bottom_5_key_similarity_clusters']:
                f.write(f"  Random Cluster {cluster_info['cluster_id']}: {cluster_info['avg_key_similarity']:.4f} "
                       f"({cluster_info['product_count']} products)\n")
            
            # Add comparison if available
            try:
                with open('key_similarity_results/key_similarity_statistics.json', 'r', encoding='utf-8') as comp_f:
                    optimized_stats = json.load(comp_f)
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("COMPARISON WITH OPTIMIZED CLUSTERING:\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("KEY SIMILARITY COMPARISON:\n")
                f.write(f"  Random Clustering Average: {stats['avg_key_similarity']:.4f}\n")
                f.write(f"  Optimized Clustering Average: {optimized_stats['avg_key_similarity']:.4f}\n")
                improvement = optimized_stats['avg_key_similarity'] - stats['avg_key_similarity']
                improvement_pct = (improvement / stats['avg_key_similarity']) * 100
                f.write(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}%)\n\n")
                
                f.write("KEY COVERAGE COMPARISON:\n")
                f.write(f"  Random Clustering Average: {stats['avg_key_coverage']:.4f}\n")
                f.write(f"  Optimized Clustering Average: {optimized_stats['avg_key_coverage']:.4f}\n")
                coverage_improvement = optimized_stats['avg_key_coverage'] - stats['avg_key_coverage']
                coverage_improvement_pct = (coverage_improvement / stats['avg_key_coverage']) * 100
                f.write(f"  Improvement: {coverage_improvement:.4f} ({coverage_improvement_pct:.1f}%)\n\n")
                
                f.write("CONCLUSION:\n")
                if improvement > 0:
                    f.write("  ✅ Optimized clustering performs BETTER than random clustering\n")
                    f.write(f"  ✅ Key similarity improved by {improvement_pct:.1f}%\n")
                else:
                    f.write("  ❌ Random clustering performs better than optimized clustering\n")
                    f.write(f"  ❌ Key similarity decreased by {abs(improvement_pct):.1f}%\n")
                
            except FileNotFoundError:
                f.write("\nNote: Optimized clustering results not available for comparison.\n")
        
        print(f"Random similarity summary report saved to {report_file}")

def main():
    """Main function to run the random cluster key similarity analysis"""
    print("Starting Random Cluster Key Similarity Analysis...")
    print("=" * 60)
    
    # Configuration
    enhanced_dataset_path = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    output_dir = "random_cluster_results"
    num_random_clusters = 170  # Same as optimized clustering
    
    # Initialize analyzer
    analyzer = RandomClusterKeySimilarityAnalyzer(enhanced_dataset_path)
    
    try:
        # Load enhanced dataset
        analyzer.load_enhanced_dataset()
        
        # Create random clusters
        random_clusters = analyzer.create_random_clusters(num_clusters=num_random_clusters)
        
        # Analyze random cluster key similarities
        cluster_stats = analyzer.analyze_random_cluster_key_similarities(random_clusters)
        
        # Generate random similarity statistics
        overall_stats = analyzer.generate_random_similarity_statistics()
        
        # Compare with optimized clustering
        comparison = analyzer.compare_with_optimized_clustering(output_dir)
        
        # Generate visualizations
        analyzer.generate_random_similarity_visualizations(output_dir)
        
        # Save results
        analyzer.save_random_similarity_results(output_dir)
        
        print("\n" + "=" * 60)
        print("Random Cluster Key Similarity Analysis Complete!")
        print("=" * 60)
        
        # Print summary
        if overall_stats:
            print(f"\n📊 RANDOM CLUSTERING SUMMARY:")
            print(f"  Total Random Clusters: {overall_stats['total_clusters']}")
            print(f"  Clusters with Multiple Products: {overall_stats['clusters_with_multiple_products']}")
            print(f"  Average Cluster Size: {overall_stats['avg_cluster_size']:.1f} products")
            
            print(f"\n🔑 RANDOM KEY SIMILARITY:")
            print(f"  Average: {overall_stats['avg_key_similarity']:.4f}")
            print(f"  Range: {overall_stats['min_key_similarity']:.4f} - {overall_stats['max_key_similarity']:.4f}")
            
            print(f"\n📋 RANDOM KEY STRUCTURE:")
            print(f"  Average Keys per Product: {overall_stats['avg_keys_per_product']:.1f}")
            print(f"  Range: {overall_stats['min_keys_per_product']:.1f} - {overall_stats['max_keys_per_product']:.1f}")
            
            print(f"\n📊 RANDOM KEY COVERAGE:")
            print(f"  Average: {overall_stats['avg_key_coverage']:.4f}")
            print(f"  Range: {overall_stats['min_key_coverage']:.4f} - {overall_stats['max_key_coverage']:.4f}")
        
        # Print comparison if available
        if comparison:
            print(f"\n🆚 COMPARISON WITH OPTIMIZED CLUSTERING:")
            print(f"  Random Average: {comparison['random_clustering']['avg_key_similarity']:.4f}")
            print(f"  Optimized Average: {comparison['optimized_clustering']['avg_key_similarity']:.4f}")
            print(f"  Improvement: {comparison['improvement']['similarity_improvement']:.4f} "
                  f"({comparison['improvement']['similarity_improvement_pct']:.1f}%)")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"📈 Visualizations: {output_dir}/*.png")
        print(f"📄 Summary report: {output_dir}/random_key_similarity_analysis_summary.txt")
        
    except Exception as e:
        print(f"Error during random similarity analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
