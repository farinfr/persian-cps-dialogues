#!/usr/bin/env python3
"""
Key Similarity Analysis Demonstration
===================================

This script demonstrates how the key similarity analysis works
by showing examples of specification key comparisons.
"""

import json

def extract_specification_keys(product):
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

def calculate_jaccard_similarity(keys1, keys2):
    """Calculate Jaccard similarity between two sets of keys"""
    if not keys1 and not keys2:
        return 1.0  # Both empty sets are identical
    if not keys1 or not keys2:
        return 0.0  # One empty, one non-empty
    
    intersection = keys1.intersection(keys2)
    union = keys1.union(keys2)
    
    return len(intersection) / len(union)

def demonstrate_key_similarity():
    """Demonstrate key similarity with examples"""
    print("KEY SIMILARITY ANALYSIS DEMONSTRATION")
    print("=" * 50)
    print()
    
    # Example 1: Simple case
    print("📋 EXAMPLE 1: Simple Key Comparison")
    print("-" * 40)
    
    json1 = {"key1": "value1", "key2": "value2"}
    json2 = {"key1": "value11", "key3": "value3"}
    
    # Extract keys
    keys1 = set(json1.keys())
    keys2 = set(json2.keys())
    
    print(f"JSON1: {json1}")
    print(f"JSON2: {json2}")
    print(f"Keys1: {keys1}")
    print(f"Keys2: {keys2}")
    
    similarity = calculate_jaccard_similarity(keys1, keys2)
    print(f"Jaccard Similarity: {similarity:.4f}")
    print(f"Explanation: |intersection| / |union| = {len(keys1.intersection(keys2))} / {len(keys1.union(keys2))} = {similarity:.4f}")
    print()
    
    # Example 2: Real product specifications
    print("📋 EXAMPLE 2: Real Product Specifications")
    print("-" * 45)
    
    # Load some real products
    try:
        with open('cluster_analysis_results/enhanced_products_with_cluster_comparison.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = data['products'][:5]  # Get first 5 products
        
        print("Comparing specification keys between products:")
        print()
        
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                product1 = products[i]
                product2 = products[j]
                
                keys1 = extract_specification_keys(product1)
                keys2 = extract_specification_keys(product2)
                
                similarity = calculate_jaccard_similarity(keys1, keys2)
                
                print(f"Product {i+1} vs Product {j+1}:")
                print(f"  Product 1: {product1.get('title', 'Unknown')[:50]}...")
                print(f"  Product 2: {product2.get('title', 'Unknown')[:50]}...")
                print(f"  Keys 1: {len(keys1)} keys")
                print(f"  Keys 2: {len(keys2)} keys")
                print(f"  Common keys: {len(keys1.intersection(keys2))}")
                print(f"  Total unique keys: {len(keys1.union(keys2))}")
                print(f"  Key Similarity: {similarity:.4f}")
                print()
                
                # Show some example keys
                if keys1 and keys2:
                    common_keys = keys1.intersection(keys2)
                    unique_keys1 = keys1 - keys2
                    unique_keys2 = keys2 - keys1
                    
                    if common_keys:
                        print(f"  Common keys: {list(common_keys)[:3]}...")
                    if unique_keys1:
                        print(f"  Unique to Product 1: {list(unique_keys1)[:3]}...")
                    if unique_keys2:
                        print(f"  Unique to Product 2: {list(unique_keys2)[:3]}...")
                    print()
    
    except FileNotFoundError:
        print("Enhanced dataset not found. Using mock data instead.")
        print()
        
        # Mock example
        mock_product1 = {
            "specifications": {
                "general": {"brand": "Samsung", "model": "Galaxy S21"},
                "display": {"size": "6.2 inch", "resolution": "1080x2400"}
            }
        }
        
        mock_product2 = {
            "specifications": {
                "general": {"brand": "Apple", "model": "iPhone 13"},
                "display": {"size": "6.1 inch", "resolution": "1170x2532"},
                "camera": {"megapixels": "12MP"}
            }
        }
        
        keys1 = extract_specification_keys(mock_product1)
        keys2 = extract_specification_keys(mock_product2)
        
        print("Mock Product Comparison:")
        print(f"Product 1 Keys: {keys1}")
        print(f"Product 2 Keys: {keys2}")
        
        similarity = calculate_jaccard_similarity(keys1, keys2)
        print(f"Key Similarity: {similarity:.4f}")
        print()
    
    # Example 3: Cluster analysis results
    print("📊 EXAMPLE 3: Cluster Analysis Results")
    print("-" * 40)
    
    try:
        with open('key_similarity_results/key_similarity_statistics.json', 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print("Overall Key Similarity Statistics:")
        print(f"  Average Key Similarity: {stats['avg_key_similarity']:.4f}")
        print(f"  Median Key Similarity: {stats['median_key_similarity']:.4f}")
        print(f"  Range: {stats['min_key_similarity']:.4f} - {stats['max_key_similarity']:.4f}")
        print()
        
        print("Key Structure Statistics:")
        print(f"  Average Keys per Product: {stats['avg_keys_per_product']:.1f}")
        print(f"  Range: {stats['min_keys_per_product']:.1f} - {stats['max_keys_per_product']:.1f}")
        print()
        
        print("Top 3 Most Similar Clusters:")
        for i, cluster in enumerate(stats['top_5_key_similarity_clusters'][:3]):
            print(f"  {i+1}. Cluster {cluster['cluster_id']}: {cluster['avg_key_similarity']:.4f} "
                  f"({cluster['product_count']} products)")
        print()
        
        print("Top 3 Least Similar Clusters:")
        for i, cluster in enumerate(stats['bottom_5_key_similarity_clusters'][:3]):
            print(f"  {i+1}. Cluster {cluster['cluster_id']}: {cluster['avg_key_similarity']:.4f} "
                  f"({cluster['product_count']} products)")
        print()
    
    except FileNotFoundError:
        print("Key similarity results not found.")
    
    print("🎯 INTERPRETATION:")
    print("-" * 20)
    print("Key Similarity measures how similar products are in terms of")
    print("their specification structure (what specifications they have).")
    print()
    print("• High similarity (0.8-1.0): Products have very similar")
    print("  specification structures")
    print("• Medium similarity (0.4-0.8): Products share some common")
    print("  specifications but also have unique ones")
    print("• Low similarity (0.0-0.4): Products have very different")
    print("  specification structures")
    print()
    print("This is different from value similarity, which would compare")
    print("the actual values of the specifications.")

if __name__ == "__main__":
    demonstrate_key_similarity()
