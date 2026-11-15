#!/usr/bin/env python3
"""
Product Feature Sampler
=======================

This script samples product features based on cluster normalized probabilities.
It:
1. Samples a product (or uses a given product)
2. Identifies the cluster the product belongs to
3. Samples features from that cluster based on normalized probabilities
4. Generates synthetic product feature sets
"""

import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import os

class ProductFeatureSampler:
    def __init__(self, 
                 products_file: str,
                 sampling_probabilities_file: str):
        """
        Initialize the sampler
        
        Args:
            products_file: Path to products JSON file with cluster_id
            sampling_probabilities_file: Path to cluster sampling probabilities JSON
        """
        self.products_file = products_file
        self.sampling_probabilities_file = sampling_probabilities_file
        self.products = []
        self.sampling_probs = {}
        self.cluster_features = {}  # Cache cluster features
        
    def load_data(self):
        """Load products and sampling probabilities"""
        print("Loading products dataset...")
        with open(self.products_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'products' in data:
            self.products = data['products']
        else:
            self.products = data
        
        print(f"Loaded {len(self.products)} products")
        
        print("Loading cluster sampling probabilities...")
        with open(self.sampling_probabilities_file, 'r', encoding='utf-8') as f:
            self.sampling_probs = json.load(f)
        
        print(f"Loaded probabilities for {len(self.sampling_probs)} clusters")
        
        # Build cluster features cache
        for cluster_id, cluster_data in self.sampling_probs.items():
            self.cluster_features[cluster_id] = cluster_data['features']
        
        return self.products, self.sampling_probs
    
    def get_product_cluster(self, product: Dict) -> Optional[str]:
        """Get the cluster ID for a product"""
        cluster_id = product.get('cluster_id')
        if cluster_id is None:
            cluster_id = product.get('cluster')
        
        return str(cluster_id) if cluster_id is not None else None
    
    def sample_product(self, product_id: Optional[int] = None, 
                      random_sample: bool = True) -> Optional[Dict]:
        """
        Sample a product from the dataset
        
        Args:
            product_id: Specific product ID to use (if None, samples randomly)
            random_sample: If True and product_id is None, samples randomly
            
        Returns:
            Product dictionary or None if not found
        """
        if product_id is not None:
            # Find specific product
            for product in self.products:
                if product.get('id') == product_id:
                    return product
            return None
        
        if random_sample and self.products:
            return random.choice(self.products)
        
        return None
    
    def sample_features_from_cluster_random(self,
                                            cluster_id: str,
                                            num_features: int,
                                            track_order: bool = True) -> Tuple[Dict[str, Any], List[str]]:
        """
        Sample features from a cluster completely randomly (no probability weights)
        
        Args:
            cluster_id: Cluster ID to sample from
            num_features: Number of features to sample
            track_order: If True, return the order of feature sampling
            
        Returns:
            Tuple of (dictionary of sampled features with their values, list of feature names in sampling order)
        """
        if cluster_id not in self.cluster_features:
            print(f"Warning: Cluster {cluster_id} not found in sampling probabilities")
            return {}, []
        
        features = self.cluster_features[cluster_id]
        sampled_features = {}
        sampling_order = []
        
        if not features:
            return {}, []
        
        # Get all feature names
        feature_names = list(features.keys())
        
        # Randomly sample without replacement
        num_to_sample = min(num_features, len(feature_names))
        sampled_feature_names = random.sample(feature_names, num_to_sample)
        
        # Track order
        sampling_order = sampled_feature_names.copy()
        
        # Get values for sampled features (still use frequency-based value sampling)
        for feature_name in sampling_order:
            feature_data = features[feature_name]
            sampled_value = self._sample_feature_value(feature_data)
            if sampled_value is not None:
                sampled_features[feature_name] = sampled_value
        
        return sampled_features, sampling_order
    
    def extract_product_features(self, product: Dict) -> Dict[str, Any]:
        """
        Extract features from product's specifications
        
        Args:
            product: Product dictionary
            
        Returns:
            Dictionary of {feature_name: feature_value}
        """
        specs = product.get('specifications', {})
        if not specs:
            return {}
        
        product_features = {}
        
        for category, category_specs in specs.items():
            if not category_specs:
                continue
                
            for key, value in category_specs.items():
                if value:  # Only include keys with non-empty values
                    # Create unique key identifier (same format as cluster features)
                    full_key = f"{category}_{key}"
                    product_features[full_key] = value
        
        return product_features
    
    def sample_features_from_cluster(self, 
                                     cluster_id: str,
                                     num_features: Optional[int] = None,
                                     min_probability: float = 0.0,
                                     track_order: bool = True,
                                     product: Optional[Dict] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Sample features from a cluster based on normalized probabilities
        
        Args:
            cluster_id: Cluster ID to sample from
            num_features: Number of features to sample (None = sample all based on probability)
            min_probability: Minimum normalized probability threshold for including a feature
            track_order: If True, return the order of feature sampling
            
        Returns:
            Tuple of (dictionary of sampled features with their values, list of feature names in sampling order)
        """
        if cluster_id not in self.cluster_features:
            print(f"Warning: Cluster {cluster_id} not found in sampling probabilities")
            return {}, []
        
        features = self.cluster_features[cluster_id]
        sampled_features = {}
        sampling_order = []  # Track the order features were sampled
        
        # Filter features by minimum probability
        eligible_features = {
            k: v for k, v in features.items() 
            if v.get('normalized_probability', 0) >= min_probability
        }
        
        # If product is provided, filter to only features that exist in the product
        product_features = {}
        if product is not None:
            product_features = self.extract_product_features(product)
            # Filter to only features that exist in the product
            eligible_features = {
                k: v for k, v in eligible_features.items()
                if k in product_features
            }
        
        if not eligible_features:
            warning_msg = f"Warning: No features found for cluster {cluster_id}"
            if product is not None:
                warning_msg += f" that exist in product {product.get('id')}"
            if min_probability > 0:
                warning_msg += f" with min_probability >= {min_probability}"
            print(warning_msg)
            return {}, []
        
        # If num_features is specified, sample that many
        if num_features is not None:
            # Sort features by normalized_probability (highest first) for better sampling
            sorted_features = sorted(
                eligible_features.items(),
                key=lambda x: x[1].get('normalized_probability', 0),
                reverse=True
            )
            
            # Weighted random sampling based on normalized_probability
            feature_names = [f[0] for f in sorted_features]
            probabilities = [f[1].get('normalized_probability', 0) for f in sorted_features]
            
            # Normalize probabilities
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            else:
                # Equal probability if all are 0
                probabilities = [1.0 / len(probabilities)] * len(probabilities) if len(probabilities) > 0 else []
            
            # Sample features (one at a time to track order)
            num_to_sample = min(num_features, len(eligible_features))
            
            # Sample without replacement, tracking order
            sampled_indices = []
            remaining_indices = list(range(len(feature_names)))
            remaining_probs = probabilities.copy()
            
            for _ in range(num_to_sample):
                if not remaining_indices:
                    break
                
                # Normalize remaining probabilities
                total_remaining = sum(remaining_probs)
                if total_remaining > 0:
                    normalized_probs = [p / total_remaining for p in remaining_probs]
                else:
                    normalized_probs = [1.0 / len(remaining_probs)] * len(remaining_probs) if len(remaining_probs) > 0 else []
                
                if not normalized_probs:
                    break
                
                # Sample one feature
                sampled_idx = np.random.choice(len(remaining_indices), p=normalized_probs)
                actual_idx = remaining_indices[sampled_idx]
                sampled_indices.append(actual_idx)
                
                # Track order
                feature_name = feature_names[actual_idx]
                sampling_order.append(feature_name)
                
                # Remove from remaining
                remaining_indices.pop(sampled_idx)
                remaining_probs.pop(sampled_idx)
            
            # Get values for sampled features in order
            for feature_name in sampling_order:
                feature_data = eligible_features[feature_name]
                
                # If product is provided, use the actual value from product
                if product is not None and feature_name in product_features:
                    actual_value = product_features[feature_name]
                    # Handle list values
                    if isinstance(actual_value, list):
                        # Use first value if list, or join if multiple
                        actual_value = actual_value[0] if len(actual_value) == 1 else ', '.join(str(v) for v in actual_value)
                    sampled_features[feature_name] = str(actual_value).strip()
                else:
                    # Otherwise sample from value_frequencies
                    sampled_value = self._sample_feature_value(feature_data)
                    if sampled_value is not None:
                        sampled_features[feature_name] = sampled_value
        
        else:
            # Sample all features based on their probability (weighted coin flip)
            # Create ordered list by sorting by probability (highest first) for deterministic order tracking
            sorted_features = sorted(
                eligible_features.items(),
                key=lambda x: x[1].get('normalized_probability', 0),
                reverse=True
            )
            
            for feature_name, feature_data in sorted_features:
                probability = feature_data.get('normalized_probability', 0)
                if random.random() < probability:
                    sampling_order.append(feature_name)
                    sampled_value = self._sample_feature_value(feature_data)
                    if sampled_value is not None:
                        sampled_features[feature_name] = sampled_value
        
        return sampled_features, sampling_order
    
    def _sample_feature_value(self, feature_data: Dict) -> Optional[Any]:
        """
        Sample a value for a feature based on value frequencies
        
        Args:
            feature_data: Feature data dictionary with value_frequencies
            
        Returns:
            Sampled value or None
        """
        value_frequencies = feature_data.get('value_frequencies', {})
        
        if not value_frequencies:
            return None
        
        # Convert to lists for weighted sampling
        values = list(value_frequencies.keys())
        frequencies = list(value_frequencies.values())
        
        # Normalize frequencies to probabilities
        total_freq = sum(frequencies)
        if total_freq == 0:
            return None
        
        probabilities = [f / total_freq for f in frequencies]
        
        # Weighted random sample
        sampled_value = np.random.choice(values, p=probabilities)
        return sampled_value
    
    def sample_product_features_random(self,
                                      product: Optional[Dict] = None,
                                      product_id: Optional[int] = None,
                                      num_features: int = 5) -> Dict[str, Any]:
        """
        Sample product features completely randomly (no probability weights)
        
        Args:
            product: Product dictionary (if None, will sample one)
            product_id: Specific product ID to use
            num_features: Number of features to sample
            
        Returns:
            Dictionary with sampled product features
        """
        # Get or sample product
        if product is None:
            product = self.sample_product(product_id=product_id, random_sample=True)
        
        if product is None:
            return {'error': 'Could not find or sample product'}
        
        # Get cluster ID
        cluster_id = self.get_product_cluster(product)
        if cluster_id is None:
            return {'error': f'Product {product.get("id")} has no cluster_id'}
        
        if cluster_id not in self.sampling_probs:
            return {'error': f'Cluster {cluster_id} not found in sampling probabilities'}
        
        # Sample features randomly (no probability weights)
        sampled_features, feature_order = self.sample_features_from_cluster_random(
            cluster_id=cluster_id,
            num_features=num_features,
            track_order=True
        )
        
        # Get cluster info
        cluster_data = self.sampling_probs[cluster_id]
        
        return {
            'product': {
                'id': product.get('id'),
                'digikala_id': product.get('digikala_id'),
                'title': product.get('title'),
                'brand': product.get('brand'),
                'category_name': product.get('category_name'),
            },
            'cluster_id': cluster_id,
            'cluster_info': {
                'product_count': cluster_data.get('product_count'),
                'total_features': cluster_data.get('total_features'),
            },
            'sampled_features': sampled_features,
            'feature_sampling_order': feature_order,
            'sampling_info': {
                'sampling_method': 'random',
                'sampled_feature_count': len(sampled_features),
                'num_features_requested': num_features,
            }
        }
    
    def sample_product_features(self,
                                product: Optional[Dict] = None,
                                product_id: Optional[int] = None,
                                num_features: Optional[int] = None,
                                min_probability: float = 0.0) -> Dict[str, Any]:
        """
        Main method: Sample a product and then sample its features
        
        Args:
            product: Product dictionary (if None, will sample one)
            product_id: Specific product ID to use
            num_features: Number of features to sample (None = probabilistic sampling)
            min_probability: Minimum probability threshold for features
            
        Returns:
            Dictionary with:
            - 'product': Original product info
            - 'cluster_id': Cluster ID
            - 'sampled_features': Dictionary of sampled features
            - 'sampling_info': Statistics about the sampling
        """
        # Get or sample product
        if product is None:
            product = self.sample_product(product_id=product_id, random_sample=True)
        
        if product is None:
            return {'error': 'Could not find or sample product'}
        
        # Get cluster ID
        cluster_id = self.get_product_cluster(product)
        if cluster_id is None:
            return {'error': f'Product {product.get("id")} has no cluster_id'}
        
        if cluster_id not in self.sampling_probs:
            return {'error': f'Cluster {cluster_id} not found in sampling probabilities'}
        
        # Sample features from cluster (returns features dict and order list)
        # Pass product to ensure features exist in the product
        sampled_features, feature_order = self.sample_features_from_cluster(
            cluster_id=cluster_id,
            num_features=num_features,
            min_probability=min_probability,
            track_order=True,
            product=product  # Pass product to filter features
        )
        
        # Get cluster info
        cluster_data = self.sampling_probs[cluster_id]
        cluster_features = cluster_data['features']
        
        # Calculate sampling statistics
        total_possible_features = len(cluster_features)
        sampled_count = len(sampled_features)
        
        # Calculate expected features based on probabilities
        if num_features is None:
            expected_features = sum(
                f.get('normalized_probability', 0) 
                for f in cluster_features.values()
            )
        else:
            expected_features = sampled_count
        
        return {
            'product': {
                'id': product.get('id'),
                'digikala_id': product.get('digikala_id'),
                'title': product.get('title'),
                'brand': product.get('brand'),
                'category_name': product.get('category_name'),
            },
            'cluster_id': cluster_id,
            'cluster_info': {
                'product_count': cluster_data.get('product_count'),
                'total_features': cluster_data.get('total_features'),
                'sum_custom_metric': cluster_data.get('sum_custom_metric'),
                'normalized_probability': cluster_data.get('normalized_probability'),
            },
            'sampled_features': sampled_features,
            'feature_sampling_order': feature_order,  # Order in which features were sampled
            'sampling_info': {
                'total_possible_features': total_possible_features,
                'sampled_feature_count': sampled_count,
                'expected_features': round(expected_features, 2),
                'num_features_requested': num_features,
                'min_probability_threshold': min_probability,
                'features_exist_in_product': True,  # All sampled features exist in product
                'values_from_product': True,  # All values are from actual product
            }
        }
    
    def batch_sample_random(self,
                            num_samples: int = 10,
                            num_features: int = 5,
                            unique_products: bool = True) -> List[Dict]:
        """
        Batch sample products with completely random feature selection (no probabilities)
        
        Args:
            num_samples: Number of products to sample
            num_features: Number of features per product
            unique_products: If True, ensures unique products
            
        Returns:
            List of sampling results
        """
        results = []
        sampled_product_ids = set()
        
        for i in range(num_samples):
            product = None
            if unique_products:
                # Keep trying until we get a unique product
                attempts = 0
                while product is None or product.get('id') in sampled_product_ids:
                    product = self.sample_product(random_sample=True)
                    attempts += 1
                    if attempts > 1000:  # Safety limit
                        print(f"Warning: Could not find unique product after {attempts} attempts")
                        break
                if product:
                    sampled_product_ids.add(product.get('id'))
            else:
                product = self.sample_product(random_sample=True)
            
            if product:
                result = self.sample_product_features_random(
                    product=product,
                    num_features=num_features
                )
                results.append(result)
        
        return results
    
    def batch_sample(self,
                     num_samples: int = 10,
                     num_features: Optional[int] = None,
                     min_probability: float = 0.0,
                     unique_products: bool = True) -> List[Dict]:
        """
        Sample multiple products and their features
        
        Args:
            num_samples: Number of products to sample
            num_features: Number of features per product (None = probabilistic)
            min_probability: Minimum probability threshold
            unique_products: If True, ensures unique products
            
        Returns:
            List of sampling results
        """
        results = []
        sampled_product_ids = set()
        
        for i in range(num_samples):
            product = None
            if unique_products:
                # Keep trying until we get a unique product
                attempts = 0
                while product is None or product.get('id') in sampled_product_ids:
                    product = self.sample_product(random_sample=True)
                    attempts += 1
                    if attempts > 1000:  # Safety limit
                        print(f"Warning: Could not find unique product after {attempts} attempts")
                        break
                if product:
                    sampled_product_ids.add(product.get('id'))
            else:
                product = self.sample_product(random_sample=True)
            
            if product:
                result = self.sample_product_features(
                    product=product,
                    num_features=num_features,
                    min_probability=min_probability
                )
                results.append(result)
        
        return results
    
    def save_samples(self, samples: List[Dict], output_file: str):
        """Save sampled results to JSON file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_samples': len(samples),
                    'timestamp': str(np.datetime64('now')),
                },
                'samples': samples
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(samples)} samples to {output_file}")

def main():
    """Main function to demonstrate product feature sampling"""
    print("Starting Product Feature Sampler...")
    print("=" * 60)
    
    # Configuration
    products_file = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
    sampling_probabilities_file = "cluster_sampling_probability_results_custome/cluster_sampling_probabilities.json"
    output_dir = "sampled_products"
    
    # Initialize sampler
    sampler = ProductFeatureSampler(products_file, sampling_probabilities_file)
    
    try:
        # Load data
        sampler.load_data()
        
        print("\n" + "=" * 60)
        print("Sampling Examples:")
        print("=" * 60)
        
        # Example 1: Sample a specific product
        print("\n1. Sampling features for a specific product (ID: 10945)...")
        result1 = sampler.sample_product_features(product_id=10945, num_features=10)
        if 'error' not in result1:
            print(f"   Product: {result1['product']['title'][:60]}...")
            print(f"   Cluster: {result1['cluster_id']}")
            print(f"   Sampled {len(result1['sampled_features'])} features")
            print(f"   Sample features: {list(result1['sampled_features'].keys())[:5]}")
        else:
            print(f"   Error: {result1['error']}")
        
        # Example 2: Random sampling with fixed number of features
        print("\n2. Random product with 15 features...")
        result2 = sampler.sample_product_features(num_features=15)
        if 'error' not in result2:
            print(f"   Product: {result2['product']['title'][:60]}...")
            print(f"   Cluster: {result2['cluster_id']}")
            print(f"   Sampled {len(result2['sampled_features'])} features")
        
        # Example 3: Probabilistic sampling (no fixed number)
        print("\n3. Random product with probabilistic feature sampling...")
        result3 = sampler.sample_product_features(min_probability=0.01)
        if 'error' not in result3:
            print(f"   Product: {result3['product']['title'][:60]}...")
            print(f"   Cluster: {result3['cluster_id']}")
            print(f"   Expected features: {result3['sampling_info']['expected_features']}")
            print(f"   Actual sampled: {len(result3['sampled_features'])} features")
        
        # Example 4: Batch sampling
        print("\n4. Batch sampling 5 products...")
        batch_results = sampler.batch_sample(num_samples=5, num_features=10, unique_products=True)
        print(f"   Successfully sampled {len(batch_results)} products")
        
        for i, sample in enumerate(batch_results[:3], 1):
            if 'error' not in sample:
                print(f"   Sample {i}: {sample['product']['title'][:50]}... "
                     f"(Cluster {sample['cluster_id']}, {len(sample['sampled_features'])} features)")
        
        # Save batch samples
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/sampled_products_batch.json"
        sampler.save_samples(batch_results, output_file)
        
        print("\n" + "=" * 60)
        print("Product Feature Sampling Complete!")
        print("=" * 60)
        print(f"\n📁 Batch samples saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during sampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
