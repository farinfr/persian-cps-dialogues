#!/usr/bin/env python3
"""
Cluster Sampling Probability Analyzer
=====================================

This script calculates optimal sampling probabilities for product features in each cluster.
Based on key frequency and value distributions, it generates probabilities that can be
used for intelligent product sampling and feature selection.

Probability Calculation Strategy:
1. Key-Level Probability: Based on key coverage (how many products have the key)
2. Value-Level Probability: Based on value frequency within the key
3. Combined Probability: Product of key and value probabilities

Output: JSON with all clusters, their features, and sampling probabilities
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import math

class ClusterSamplingProbabilityAnalyzer:
    def __init__(self, analysis_file_path: str):
        self.analysis_file_path = analysis_file_path
        self.analysis_data = None
        self.cluster_sampling_probs = {}
        self.global_stats = {}
        
    def load_analysis_data(self):
        """Load the cluster key-value analysis data"""
        print("Loading cluster key-value analysis data...")
        with open(self.analysis_file_path, 'r', encoding='utf-8') as f:
            self.analysis_data = json.load(f)
        
        print(f"Loaded analysis data for {len(self.analysis_data)} clusters")
        return self.analysis_data
    
    def calculate_custom_sampling_probabilities(self):
        """Calculate custom sampling probabilities for each cluster"""
        print("Calculating custom sampling probabilities for each cluster...")
        # loop on clusters
        for cluster_id, cluster_data in self.analysis_data.items():
            print(f"Processing cluster {cluster_id}...")
            
            product_count_cluster = cluster_data['product_count']
            key_frequency = cluster_data['key_frequency']
            key_value_frequency = cluster_data['key_value_frequency']
            
            cluster_features = {}
            sum_custom_metric = 0;
            # loop on features
            for key, product_count in key_frequency.items():
                value_frequencies= key_value_frequency.get(key, {})
                partition_count = len(value_frequencies) + 1
                mean_product_partition_count = product_count_cluster / partition_count;
                if partition_count > 0:
                    sum_difrence = 0;
                    
                    # loop on partitions
                    for value, frequency in value_frequencies.items():
                        sum_difrence += (frequency - mean_product_partition_count)**2
                    sum_difrence += (product_count_cluster - product_count - mean_product_partition_count)**2
                    std_deviation = sum_difrence / partition_count
                    standard_deviation = math.sqrt(std_deviation)
                    hyper_parameter = mean_product_partition_count*product_count_cluster
                    custom_metric = 1/(standard_deviation + (hyper_parameter*(abs(math.sqrt(product_count)-partition_count))))
                    cluster_features[key] = {
                        'partition_count': partition_count,
                        'mean_product_partition_count': mean_product_partition_count,
                        'std_deviation': round(standard_deviation, 6),
                        'custom_metric': round(custom_metric, 6),
                        'value_frequencies': value_frequencies,
                    }
                    sum_custom_metric += custom_metric
                else:
                    print(f"No values found for key {key} in cluster {cluster_id}")
                    cluster_features[key] = {
                        'partition_count': partition_count,
                        'mean_product_partition_count': mean_product_partition_count,
                        'std_deviation': 0,
                        'custom_metric': 0,
                    }
            # Store cluster probabilities
            self.cluster_sampling_probs[cluster_id] = {
                'cluster_id': cluster_id,
                'product_count': product_count_cluster,
                'total_features': len(cluster_features),
                'sum_custom_metric': round(sum_custom_metric, 6),
                'features': cluster_features,
            }


  
    
    def normalized_probability(self):
        """Normalize the probability for features and clusters"""
        # First, normalize feature probabilities within each cluster
        for cluster_id, cluster_data in self.cluster_sampling_probs.items():
            for feature_key, feature_data in cluster_data['features'].items():
                if cluster_data['sum_custom_metric'] > 0:
                    feature_data['normalized_probability'] = feature_data['custom_metric'] / cluster_data['sum_custom_metric']
                    feature_data['normalized_probability'] = round(feature_data['normalized_probability'], 6)
                else:
                    feature_data['normalized_probability'] = 0.0
        
        # Calculate total sum_custom_metric across all clusters for cluster-level normalization
        total_cluster_sum = sum(cluster_data['sum_custom_metric'] for cluster_data in self.cluster_sampling_probs.values())
        
        # Add normalized probability to each cluster
        for cluster_id, cluster_data in self.cluster_sampling_probs.items():
            if total_cluster_sum > 0:
                cluster_data['normalized_probability'] = round(cluster_data['sum_custom_metric'] / total_cluster_sum, 6)
            else:
                cluster_data['normalized_probability'] = 0.0
        
        print(f"Total cluster sum_custom_metric: {total_cluster_sum}")
        print(f"Added cluster-level normalized_probability to {len(self.cluster_sampling_probs)} clusters")
               
    def generate_global_statistics(self):
        """Generate global statistics about sampling probabilities"""
        print("Generating global statistics...")
        
        all_features = set()
        feature_coverage_scores = []
        feature_entropies = []
        feature_sampling_weights = []
        
        for cluster_id, cluster_data in self.cluster_sampling_probs.items():
            for feature_key, feature_data in cluster_data['features'].items():
                all_features.add(feature_key)
                
                feature_coverage_scores.append(feature_data['key_coverage'])
                feature_entropies.append(feature_data.get('information_value', 0))
                feature_sampling_weights.append(feature_data.get('sampling_weight', 0))
        
        # Find best features by different criteria
        best_by_coverage = sorted(self.cluster_sampling_probs.items(), 
                                 key=lambda x: sum(f['key_coverage'] for f in x[1]['features'].values()),
                                 reverse=True)[:10]
        
        best_by_information = sorted(self.cluster_sampling_probs.items(),
                                   key=lambda x: sum(f.get('information_value', 0) for f in x[1]['features'].values()),
                                   reverse=True)[:10]
        
        best_by_weight = sorted(self.cluster_sampling_probs.items(),
                              key=lambda x: sum(f.get('sampling_weight', 0) for f in x[1]['features'].values()),
                              reverse=True)[:10]
        
        self.global_stats = {
            'total_clusters': len(self.cluster_sampling_probs),
            'total_unique_features_globally': len(all_features),
            'avg_key_coverage': np.mean(feature_coverage_scores) if feature_coverage_scores else 0,
            'median_key_coverage': np.median(feature_coverage_scores) if feature_coverage_scores else 0,
            'avg_information_value': np.mean(feature_entropies) if feature_entropies else 0,
            'median_information_value': np.median(feature_entropies) if feature_entropies else 0,
            'avg_sampling_weight': np.mean(feature_sampling_weights) if feature_sampling_weights else 0,
            'median_sampling_weight': np.median(feature_sampling_weights) if feature_sampling_weights else 0,
            'clusters_best_by_coverage': [
                {
                    'cluster_id': int(cluster_id),
                    'cluster_id_str': cluster_id,
                    'total_features': data['total_features'],
                    'avg_coverage': sum(f['key_coverage'] for f in data['features'].values()) / len(data['features']) if data['features'] else 0
                }
                for cluster_id, data in best_by_coverage
            ],
            'clusters_best_by_information': [
                {
                    'cluster_id': int(cluster_id),
                    'cluster_id_str': cluster_id,
                    'total_features': data['total_features'],
                    'total_information_value': sum(f.get('information_value', 0) for f in data['features'].values())
                }
                for cluster_id, data in best_by_information
            ],
            'clusters_best_by_weight': [
                {
                    'cluster_id': int(cluster_id),
                    'cluster_id_str': cluster_id,
                    'total_features': data['total_features'],
                    'total_sampling_weight': sum(f.get('sampling_weight', 0) for f in data['features'].values())
                }
                for cluster_id, data in best_by_weight
            ]
        }
        
        return self.global_stats
    
    def save_results(self, output_dir: str = "cluster_sampling_probability_results"):
        """Save sampling probability results"""
        print("Saving sampling probability results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert numpy types
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
        
        # Save main results
        main_file = f'{output_dir}/cluster_sampling_probabilities.json'
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(self.cluster_sampling_probs), f, ensure_ascii=False, indent=2)
        
        # Save statistics
        stats_file = f'{output_dir}/sampling_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(self.global_stats), f, ensure_ascii=False, indent=2)
        
        # Create summary report
        self._create_summary_report(output_dir)
        
        print(f"Results saved to {output_dir}/")
    
    def _create_summary_report(self, output_dir: str):
        """Create human-readable summary report"""
        report_file = f'{output_dir}/sampling_probability_summary.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("CLUSTER SAMPLING PROBABILITY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not self.global_stats:
                f.write("No statistics available.\n")
                return
            
            stats = self.global_stats
            
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total Clusters: {stats['total_clusters']}\n")
            f.write(f"  Total Unique Features Globally: {stats['total_unique_features_globally']}\n")
            f.write(f"  Average Key Coverage: {stats['avg_key_coverage']:.4f}\n")
            f.write(f"  Median Key Coverage: {stats['median_key_coverage']:.4f}\n")
            f.write(f"  Average Information Value: {stats['avg_information_value']:.4f}\n")
            f.write(f"  Median Information Value: {stats['median_information_value']:.4f}\n")
            f.write(f"  Average Sampling Weight: {stats['avg_sampling_weight']:.4f}\n")
            f.write(f"  Median Sampling Weight: {stats['median_sampling_weight']:.4f}\n\n")
            
            f.write("TOP 10 CLUSTERS BY COVERAGE (Best for common features):\n")
            for i, cluster_info in enumerate(stats['clusters_best_by_coverage'][:10], 1):
                f.write(f"  {i:2d}. Cluster {cluster_info['cluster_id']}: {cluster_info['total_features']} features, "
                       f"avg coverage: {cluster_info['avg_coverage']:.4f}\n")
            
            f.write("\nTOP 10 CLUSTERS BY INFORMATION VALUE (Most diverse features):\n")
            for i, cluster_info in enumerate(stats['clusters_best_by_information'][:10], 1):
                f.write(f"  {i:2d}. Cluster {cluster_info['cluster_id']}: {cluster_info['total_features']} features, "
                       f"total info value: {cluster_info['total_information_value']:.4f}\n")
            
            f.write("\nTOP 10 CLUSTERS BY SAMPLING WEIGHT (Best overall for sampling):\n")
            for i, cluster_info in enumerate(stats['clusters_best_by_weight'][:10], 1):
                f.write(f"  {i:2d}. Cluster {cluster_info['cluster_id']}: {cluster_info['total_features']} features, "
                       f"total weight: {cluster_info['total_sampling_weight']:.4f}\n")
        
        print(f"Summary report saved to {report_file}")

def main():
    """Main function to run the sampling probability analysis"""
    print("Starting Cluster Sampling Probability Analysis...")
    print("=" * 60)
    
    # Configuration
    analysis_file = "cluster_key_value_results/cluster_key_value_analysis.json"
    output_dir = "cluster_sampling_probability_results_custome"
    
    # Initialize analyzer
    analyzer = ClusterSamplingProbabilityAnalyzer(analysis_file)
    
    try:
        # Load analysis data
        analyzer.load_analysis_data()
        analyzer.calculate_custom_sampling_probabilities()
        analyzer.normalized_probability()
        analyzer.save_results(output_dir)
       
        
        print("\n" + "=" * 60)
        print("Cluster Sampling Probability Analysis Complete!")
        print("=" * 60)
        
        # Print summary
        if analyzer.global_stats:
            stats = analyzer.global_stats
            print(f"\n📊 SAMPLING PROBABILITY SUMMARY:")
            print(f"  Total Clusters: {stats['total_clusters']}")
            print(f"  Total Unique Features: {stats['total_unique_features_globally']}")
            print(f"  Average Key Coverage: {stats['avg_key_coverage']:.4f}")
            print(f"  Average Information Value: {stats['avg_information_value']:.4f}")
            print(f"  Average Sampling Weight: {stats['avg_sampling_weight']:.4f}")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print(f"📊 Main file: {output_dir}/cluster_sampling_probabilities.json")
        print(f"📈 Statistics: {output_dir}/sampling_statistics.json")
        print(f"📄 Summary report: {output_dir}/sampling_probability_summary.txt")
        
    except Exception as e:
        print(f"Error during sampling probability analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
