#!/usr/bin/env python3
"""
Cluster Key-Value Analysis Tool
===============================

This script analyzes keys and their values within each cluster of products.
It calculates:
1. How many times each key exists in products of each cluster
2. What values each key has across products in the cluster
3. How many times each value appears for each key in the cluster
4. Generates comprehensive JSON output and analysis

Features:
- Extracts all specification keys from products in each cluster
- Counts key frequency per cluster
- Analyzes value distributions for each key
- Identifies most/least common keys and values
- Generates detailed statistics and visualizations
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Set, Union
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ClusterKeyValueAnalyzer:
    def __init__(self, enhanced_dataset_path: str):
        self.enhanced_dataset_path = enhanced_dataset_path
        self.products_data = None
        self.cluster_analysis = {}
        self.global_statistics = {}
        
    def load_enhanced_dataset(self):
        """Load the enhanced dataset with cluster information"""
        print("Loading enhanced dataset...")
        with open(self.enhanced_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.products_data = data['products']
        print(f"Loaded {len(self.products_data)} products")
        return self.products_data
    
    def extract_specification_keys_and_values(self, product: Dict) -> Dict[str, Union[str, List[str]]]:
        """Extract specification keys and their values from a product"""
        specs = product.get('specifications', {})
        if not specs:
            return {}
        
        key_values = {}
        
        for category, category_specs in specs.items():
            if not category_specs:
                continue
                
            for key, value in category_specs.items():
                if value:  # Only include keys with non-empty values
                    # Create unique key identifier
                    full_key = f"{category}_{key}"
                    key_values[full_key] = value
        
        return key_values
    
    def analyze_cluster_keys_and_values(self):
        """Analyze keys and values within each cluster"""
        print("Analyzing keys and values within clusters...")
        
        # Group products by cluster
        cluster_products = defaultdict(list)
        for product in self.products_data:
            cluster_id = product.get('cluster_id')
            if cluster_id is not None:
                cluster_products[cluster_id].append(product)
        
        print(f"Found {len(cluster_products)} clusters")
        
        cluster_stats = {}
        
        for cluster_id, products in cluster_products.items():
            print(f"Analyzing cluster {cluster_id} with {len(products)} products...")
            
            # Extract all key-value pairs from products in this cluster
            all_key_values = []
            for product in products:
                key_values = self.extract_specification_keys_and_values(product)
                all_key_values.append(key_values)
            
            # Count key frequency in this cluster
            key_frequency = Counter()
            key_value_frequency = defaultdict(Counter)
            
            for product_key_values in all_key_values:
                for key, value in product_key_values.items():
                    key_frequency[key] += 1
                    
                    # Handle both single values and lists
                    if isinstance(value, list):
                        for v in value:
                            key_value_frequency[key][str(v).strip()] += 1
                    else:
                        key_value_frequency[key][str(value).strip()] += 1
            
            # Calculate key coverage (percentage of products that have each key)
            key_coverage = {}
            for key, count in key_frequency.items():
                key_coverage[key] = count / len(products)
            
            # Analyze value diversity for each key
            value_diversity = {}
            for key, value_counter in key_value_frequency.items():
                total_values = sum(value_counter.values())
                unique_values = len(value_counter)
                value_diversity[key] = {
                    'unique_values': unique_values,
                    'total_occurrences': total_values,
                    'most_common_value': value_counter.most_common(1)[0] if value_counter else None,
                    'value_distribution': dict(value_counter.most_common(10))  # Top 10 values
                }
            
            # Store cluster statistics
            cluster_stats[cluster_id] = {
                'product_count': len(products),
                'total_unique_keys': len(key_frequency),
                'key_frequency': dict(key_frequency),
                'key_coverage': key_coverage,
                'key_value_frequency': {k: dict(v) for k, v in key_value_frequency.items()},
                'value_diversity': value_diversity,
                'most_common_keys': key_frequency.most_common(10),
                'least_common_keys': key_frequency.most_common()[-10:] if len(key_frequency) >= 10 else key_frequency.most_common(),
                'keys_with_high_coverage': {k: v for k, v in key_coverage.items() if v >= 0.8},
                'keys_with_low_coverage': {k: v for k, v in key_coverage.items() if v <= 0.2}
            }
        
        self.cluster_analysis = cluster_stats
        return cluster_stats
    
    def generate_global_statistics(self) -> Dict:
        """Generate global statistics across all clusters"""
        print("Generating global statistics...")
        
        if not self.cluster_analysis:
            return {}
        
        # Collect all keys across all clusters
        all_keys = set()
        all_key_frequencies = Counter()
        cluster_sizes = []
        keys_per_cluster = []
        
        for cluster_id, stats in self.cluster_analysis.items():
            cluster_sizes.append(stats['product_count'])
            keys_per_cluster.append(stats['total_unique_keys'])
            
            for key, frequency in stats['key_frequency'].items():
                all_keys.add(key)
                all_key_frequencies[key] += frequency
        
        # Calculate global key statistics
        global_stats = {
            'total_clusters': len(self.cluster_analysis),
            'total_unique_keys_globally': len(all_keys),
            'avg_cluster_size': np.mean(cluster_sizes),
            'median_cluster_size': np.median(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
            'avg_keys_per_cluster': np.mean(keys_per_cluster),
            'median_keys_per_cluster': np.median(keys_per_cluster),
            'min_keys_per_cluster': np.min(keys_per_cluster),
            'max_keys_per_cluster': np.max(keys_per_cluster),
            'most_common_keys_globally': all_key_frequencies.most_common(20),
            'least_common_keys_globally': all_key_frequencies.most_common()[-20:] if len(all_key_frequencies) >= 20 else all_key_frequencies.most_common(),
            'key_frequency_distribution': dict(all_key_frequencies)
        }
        
        # Analyze clusters with most/least diverse keys
        clusters_by_key_diversity = sorted(self.cluster_analysis.items(), 
                                         key=lambda x: x[1]['total_unique_keys'], reverse=True)
        
        global_stats['clusters_with_most_keys'] = [
            {
                'cluster_id': cluster_id,
                'total_unique_keys': stats['total_unique_keys'],
                'product_count': stats['product_count']
            }
            for cluster_id, stats in clusters_by_key_diversity[:10]
        ]
        
        global_stats['clusters_with_least_keys'] = [
            {
                'cluster_id': cluster_id,
                'total_unique_keys': stats['total_unique_keys'],
                'product_count': stats['product_count']
            }
            for cluster_id, stats in clusters_by_key_diversity[-10:]
        ]
        
        self.global_statistics = global_stats
        return global_stats
    
    def generate_visualizations(self, output_dir: str = "cluster_key_value_results"):
        """Generate visualizations for key-value analysis"""
        print("Generating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Keys per Cluster Distribution
        keys_per_cluster = [stats['total_unique_keys'] for stats in self.cluster_analysis.values()]
        
        plt.figure(figsize=(12, 8))
        plt.hist(keys_per_cluster, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Unique Keys per Cluster')
        plt.xlabel('Number of Unique Keys')
        plt.ylabel('Number of Clusters')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/01_keys_per_cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Values per Key Distribution (for most common keys)
        all_value_counts = []
        for stats in self.cluster_analysis.values():
            for key, diversity in stats['value_diversity'].items():
                all_value_counts.append(diversity['unique_values'])
        
        plt.figure(figsize=(12, 8))
        plt.hist(all_value_counts, bins=30, alpha=0.7, edgecolor='black', color='green')
        plt.title('Distribution of Unique Values per Key')
        plt.xlabel('Number of Unique Values')
        plt.ylabel('Number of Key-Cluster Combinations')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/02_values_per_key_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Key Coverage Distribution
        all_key_coverages = []
        for stats in self.cluster_analysis.values():
            all_key_coverages.extend(stats['key_coverage'].values())
        
        plt.figure(figsize=(12, 8))
        plt.hist(all_key_coverages, bins=30, alpha=0.7, edgecolor='black', color='orange')
        plt.title('Distribution of Key Coverage Across Clusters')
        plt.xlabel('Key Coverage (Percentage of Products)')
        plt.ylabel('Number of Key-Cluster Combinations')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/03_key_coverage_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Value Diversity Distribution
        plt.figure(figsize=(12, 8))
        plt.hist(all_value_counts, bins=30, alpha=0.7, edgecolor='black', color='purple')
        plt.title('Distribution of Value Diversity per Key')
        plt.xlabel('Value Diversity (Unique Values per Key)')
        plt.ylabel('Number of Key-Cluster Combinations')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/04_value_diversity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Most Common Keys Globally
        if self.global_statistics and 'most_common_keys_globally' in self.global_statistics:
            top_keys = self.global_statistics['most_common_keys_globally'][:20]
            keys = [item[0] for item in top_keys]
            frequencies = [item[1] for item in top_keys]
            
            plt.figure(figsize=(15, 8))
            plt.bar(range(len(keys)), frequencies, alpha=0.7)
            plt.title('Top 20 Most Common Keys Globally')
            plt.xlabel('Keys')
            plt.ylabel('Total Frequency Across All Clusters')
            plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/05_most_common_keys.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Most Diverse Keys (by unique values)
        key_diversity = defaultdict(int)
        for stats in self.cluster_analysis.values():
            for key, diversity in stats['value_diversity'].items():
                key_diversity[key] += diversity['unique_values']
        
        top_diverse_keys = sorted(key_diversity.items(), key=lambda x: x[1], reverse=True)[:20]
        
        plt.figure(figsize=(15, 8))
        keys = [item[0] for item in top_diverse_keys]
        diversities = [item[1] for item in top_diverse_keys]
        
        plt.bar(range(len(keys)), diversities, alpha=0.7, color='red')
        plt.title('Top 20 Most Diverse Keys (by Unique Values)')
        plt.xlabel('Keys')
        plt.ylabel('Total Unique Values Across All Clusters')
        plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/06_most_diverse_keys.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def save_analysis_results(self, output_dir: str = "cluster_key_value_results"):
        """Save comprehensive analysis results"""
        print("Saving analysis results...")
        
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
                return list(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj
        
        # Save detailed cluster analysis
        cluster_file = f'{output_dir}/cluster_key_value_analysis.json'
        with open(cluster_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(self.cluster_analysis), f, ensure_ascii=False, indent=2)
        
        # Save global statistics
        stats_file = f'{output_dir}/key_value_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(self.global_statistics), f, ensure_ascii=False, indent=2)
        
        # Create summary report
        self._create_summary_report(output_dir)
        
        print(f"Analysis results saved to {output_dir}/")
    
    def _create_summary_report(self, output_dir: str):
        """Create a human-readable summary report"""
        report_file = f'{output_dir}/key_value_analysis_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CLUSTER KEY-VALUE ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not self.global_statistics:
                f.write("No analysis results available.\n")
                return
            
            stats = self.global_statistics
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total Clusters: {stats['total_clusters']}\n")
            f.write(f"  Total Unique Keys Globally: {stats['total_unique_keys_globally']}\n")
            f.write(f"  Average Cluster Size: {stats['avg_cluster_size']:.1f} products\n")
            f.write(f"  Median Cluster Size: {stats['median_cluster_size']:.1f} products\n")
            f.write(f"  Cluster Size Range: {stats['min_cluster_size']} - {stats['max_cluster_size']} products\n\n")
            
            f.write("KEY STATISTICS:\n")
            f.write(f"  Average Keys per Cluster: {stats['avg_keys_per_cluster']:.1f}\n")
            f.write(f"  Median Keys per Cluster: {stats['median_keys_per_cluster']:.1f}\n")
            f.write(f"  Keys per Cluster Range: {stats['min_keys_per_cluster']} - {stats['max_keys_per_cluster']}\n\n")
            
            f.write("TOP 10 MOST COMMON KEYS GLOBALLY:\n")
            for i, (key, frequency) in enumerate(stats['most_common_keys_globally'][:10], 1):
                f.write(f"  {i:2d}. {key}: {frequency} occurrences\n")
            
            f.write("\nTOP 10 CLUSTERS WITH MOST KEYS:\n")
            for i, cluster_info in enumerate(stats['clusters_with_most_keys'][:10], 1):
                f.write(f"  {i:2d}. Cluster {cluster_info['cluster_id']}: {cluster_info['total_unique_keys']} keys "
                       f"({cluster_info['product_count']} products)\n")
            
            f.write("\nTOP 10 CLUSTERS WITH LEAST KEYS:\n")
            for i, cluster_info in enumerate(stats['clusters_with_least_keys'][:10], 1):
                f.write(f"  {i:2d}. Cluster {cluster_info['cluster_id']}: {cluster_info['total_unique_keys']} keys "
                       f"({cluster_info['product_count']} products)\n")
        
        print(f"Summary report saved to {report_file}")

def main():
    """Main function to run the cluster key-value analysis"""
    print("Starting Cluster Key-Value Analysis...")
    print("=" * 60)
    
    # Configuration
    enhanced_dataset_path = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    output_dir = "cluster_key_value_results"
    
    # Initialize analyzer
    analyzer = ClusterKeyValueAnalyzer(enhanced_dataset_path)
    
    try:
        # Load enhanced dataset
        analyzer.load_enhanced_dataset()
        
        # Analyze cluster keys and values
        cluster_stats = analyzer.analyze_cluster_keys_and_values()
        
        # Generate global statistics
        global_stats = analyzer.generate_global_statistics()
        
        # Generate visualizations
        analyzer.generate_visualizations(output_dir)
        
        # Save results
        analyzer.save_analysis_results(output_dir)
        
        print("\n" + "=" * 60)
        print("Cluster Key-Value Analysis Complete!")
        print("=" * 60)
        
        # Print summary
        if global_stats:
            print(f"\n📊 KEY-VALUE ANALYSIS SUMMARY:")
            print(f"  Total Clusters: {global_stats['total_clusters']}")
            print(f"  Total Unique Keys Globally: {global_stats['total_unique_keys_globally']}")
            print(f"  Average Cluster Size: {global_stats['avg_cluster_size']:.1f} products")
            print(f"  Average Keys per Cluster: {global_stats['avg_keys_per_cluster']:.1f}")
            
            print(f"\n🔑 TOP 5 MOST COMMON KEYS:")
            for i, (key, frequency) in enumerate(global_stats['most_common_keys_globally'][:5], 1):
                print(f"  {i}. {key}: {frequency} occurrences")
            
            print(f"\n📈 CLUSTERS WITH MOST KEYS:")
            for i, cluster_info in enumerate(global_stats['clusters_with_most_keys'][:5], 1):
                print(f"  {i}. Cluster {cluster_info['cluster_id']}: {cluster_info['total_unique_keys']} keys")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"📈 Visualizations: {output_dir}/*.png")
        print(f"📄 Summary report: {output_dir}/key_value_analysis_summary.txt")
        print(f"📊 Detailed analysis: {output_dir}/cluster_key_value_analysis.json")
        
    except Exception as e:
        print(f"Error during key-value analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
