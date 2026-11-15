#!/usr/bin/env python3
"""
Analyze statistics of generated dialogues in full_dialogues_dataset
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any

def analyze_dialogues(directory: str = "full_dialogues_dataset"):
    """Analyze all dialogue files and generate statistics"""
    
    print("=" * 70)
    print("Dialogue Statistics Analysis")
    print("=" * 70)
    
    # Get all dialogue files
    dir_path = Path(directory)
    dialogue_files = sorted(
        [f for f in dir_path.glob("dialogue_*.json")],
        key=lambda x: int(x.stem.split('_')[1]) if x.stem.split('_')[1].isdigit() else 0
    )
    
    print(f"\n📁 Found {len(dialogue_files)} dialogue files in {directory}\n")
    
    # Statistics containers
    total_dialogues = 0
    total_rounds = 0
    total_features_discussed = 0
    total_selected_features = 0
    total_custom_metric = 0.0
    
    categories = Counter()
    brands = Counter()
    clusters = Counter()
    dialogue_rounds_distribution = []
    features_per_dialogue = []
    custom_metrics_per_dialogue = []
    
    products_with_price = 0
    products_without_price = 0
    products_with_similar = 0
    products_without_similar = 0
    
    similar_products_count = []
    
    failed_files = []
    
    # Analyze each dialogue
    for dialogue_file in dialogue_files:
        try:
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                dialogue = json.load(f)
            
            total_dialogues += 1
            
            # Basic stats
            rounds = dialogue.get('dialogue_rounds', 0)
            total_rounds += rounds
            dialogue_rounds_distribution.append(rounds)
            
            # Category and brand
            category = dialogue.get('category', 'Unknown')
            brand = dialogue.get('brand', 'Unknown')
            cluster_id = dialogue.get('cluster_id', 'Unknown')
            
            categories[category] += 1
            brands[brand] += 1
            clusters[str(cluster_id)] += 1
            
            # Features
            features_discussed = dialogue.get('features_discussed', [])
            selected_features = dialogue.get('selected_features', [])
            
            total_features_discussed += len(features_discussed)
            total_selected_features += len(selected_features)
            features_per_dialogue.append(len(features_discussed))
            
            # Custom metrics
            custom_metric_sum = sum(
                f.get('custom_metric', 0.0) for f in selected_features
            )
            total_custom_metric += custom_metric_sum
            custom_metrics_per_dialogue.append(custom_metric_sum)
            
            # Price
            price = dialogue.get('price')
            if price is not None and price != 0:
                products_with_price += 1
            else:
                products_without_price += 1
            
            # Similar products
            similar_products = dialogue.get('similar_products', [])
            if similar_products:
                products_with_similar += 1
                similar_products_count.append(len(similar_products))
            else:
                products_without_similar += 1
                
        except Exception as e:
            failed_files.append((dialogue_file.name, str(e)))
            print(f"⚠️  Error reading {dialogue_file.name}: {e}")
    
    # Calculate averages
    avg_rounds = total_rounds / total_dialogues if total_dialogues > 0 else 0
    avg_features_discussed = total_features_discussed / total_dialogues if total_dialogues > 0 else 0
    avg_selected_features = total_selected_features / total_dialogues if total_dialogues > 0 else 0
    avg_custom_metric = total_custom_metric / total_dialogues if total_dialogues > 0 else 0
    
    # Print statistics
    print("=" * 70)
    print("📊 OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total Dialogues: {total_dialogues}")
    print(f"Total Dialogue Rounds: {total_rounds}")
    print(f"Average Dialogue Rounds: {avg_rounds:.2f}")
    print(f"Min Dialogue Rounds: {min(dialogue_rounds_distribution) if dialogue_rounds_distribution else 0}")
    print(f"Max Dialogue Rounds: {max(dialogue_rounds_distribution) if dialogue_rounds_distribution else 0}")
    
    print(f"\n📈 FEATURES STATISTICS")
    print("=" * 70)
    print(f"Total Features Discussed: {total_features_discussed}")
    print(f"Average Features Discussed per Dialogue: {avg_features_discussed:.2f}")
    print(f"Total Selected Features: {total_selected_features}")
    print(f"Average Selected Features per Dialogue: {avg_selected_features:.2f}")
    print(f"Average Custom Metric per Dialogue: {avg_custom_metric:.6f}")
    print(f"Min Features Discussed: {min(features_per_dialogue) if features_per_dialogue else 0}")
    print(f"Max Features Discussed: {max(features_per_dialogue) if features_per_dialogue else 0}")
    
    print(f"\n💰 PRICE STATISTICS")
    print("=" * 70)
    print(f"Products with Price: {products_with_price} ({products_with_price/total_dialogues*100:.1f}%)")
    print(f"Products without Price: {products_without_price} ({products_without_price/total_dialogues*100:.1f}%)")
    
    print(f"\n🔗 SIMILAR PRODUCTS STATISTICS")
    print("=" * 70)
    print(f"Products with Similar Products: {products_with_similar} ({products_with_similar/total_dialogues*100:.1f}%)")
    print(f"Products without Similar Products: {products_without_similar} ({products_without_similar/total_dialogues*100:.1f}%)")
    if similar_products_count:
        avg_similar = sum(similar_products_count) / len(similar_products_count)
        print(f"Average Similar Products per Dialogue: {avg_similar:.2f}")
        print(f"Min Similar Products: {min(similar_products_count)}")
        print(f"Max Similar Products: {max(similar_products_count)}")
    
    print(f"\n📂 CATEGORY DISTRIBUTION")
    print("=" * 70)
    for category, count in categories.most_common(10):
        percentage = (count / total_dialogues) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print(f"\n🏷️  TOP BRANDS")
    print("=" * 70)
    for brand, count in brands.most_common(10):
        percentage = (count / total_dialogues) * 100
        print(f"  {brand}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔢 CLUSTER DISTRIBUTION")
    print("=" * 70)
    print(f"Total Unique Clusters: {len(clusters)}")
    for cluster, count in clusters.most_common(10):
        percentage = (count / total_dialogues) * 100
        print(f"  Cluster {cluster}: {count} ({percentage:.1f}%)")
    
    print(f"\n📊 DIALOGUE ROUNDS DISTRIBUTION")
    print("=" * 70)
    rounds_counter = Counter(dialogue_rounds_distribution)
    for rounds in sorted(rounds_counter.keys()):
        count = rounds_counter[rounds]
        percentage = (count / total_dialogues) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {rounds:2d} rounds: {count:3d} ({percentage:5.1f}%) {bar}")
    
    if failed_files:
        print(f"\n⚠️  FAILED FILES")
        print("=" * 70)
        for filename, error in failed_files:
            print(f"  {filename}: {error}")
    
    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)
    
    return {
        'total_dialogues': total_dialogues,
        'total_rounds': total_rounds,
        'avg_rounds': avg_rounds,
        'categories': dict(categories),
        'brands': dict(brands),
        'clusters': dict(clusters),
        'dialogue_rounds_distribution': dialogue_rounds_distribution,
        'features_per_dialogue': features_per_dialogue,
        'custom_metrics_per_dialogue': custom_metrics_per_dialogue
    }


def main():
    """Main function"""
    directory = "full_dialogues_dataset"
    
    if not os.path.exists(directory):
        print(f"❌ Directory '{directory}' not found!")
        return
    
    stats = analyze_dialogues(directory)
    
    # Optionally save statistics to file
    output_file = os.path.join(directory, "dialogue_statistics.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_dialogues': stats['total_dialogues'],
                'total_rounds': stats['total_rounds'],
                'avg_rounds': stats['avg_rounds']
            },
            'categories': stats['categories'],
            'brands': stats['brands'],
            'clusters': stats['clusters']
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Statistics saved to: {output_file}")


if __name__ == "__main__":
    main()


