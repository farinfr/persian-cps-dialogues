#!/usr/bin/env python3
"""
Calculate statistics for each cluster
Analyze rating_average, rating_count, and price per cluster
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load clustered products and original dataset"""
    logger.info("Loading data...")
    
    # Load clustered products
    with open('fine_grained_results/final_clustered_products.json', 'r', encoding='utf-8') as f:
        clustered_data = json.load(f)
    
    df_clusters = pd.DataFrame(clustered_data['products'])
    
    # Load original dataset with all details
    with open('../datasets/complete_dataset.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    df_original = pd.DataFrame(original_data['products'])
    
    logger.info(f"Loaded {len(df_clusters)} clustered products")
    logger.info(f"Loaded {len(df_original)} original products")
    
    # Merge to get full data with clusters
    df_merged = df_clusters.merge(
        df_original[['digikala_id', 'rating_average', 'rating_count', 'price', 'selling_price', 
                    'discount_percent', 'specifications', 'is_available']],
        on='digikala_id',
        how='left'
    )
    
    logger.info(f"Merged dataset: {len(df_merged)} products")
    
    return df_merged, clustered_data['metadata']


def calculate_cluster_statistics(df):
    """Calculate statistics for each cluster"""
    logger.info("Calculating cluster statistics...")
    
    # Group by cluster
    cluster_stats = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        stats = {
            'cluster_id': int(cluster_id),
            'total_products': len(cluster_data),
            
            # Rating statistics
            'rating_average_mean': float(cluster_data['rating_average'].mean()) if cluster_data['rating_average'].notna().any() else None,
            'rating_average_std': float(cluster_data['rating_average'].std()) if cluster_data['rating_average'].notna().any() else None,
            'rating_average_min': float(cluster_data['rating_average'].min()) if cluster_data['rating_average'].notna().any() else None,
            'rating_average_max': float(cluster_data['rating_average'].max()) if cluster_data['rating_average'].notna().any() else None,
            
            # Rating count statistics
            'rating_count_mean': float(cluster_data['rating_count'].mean()) if cluster_data['rating_count'].notna().any() else None,
            'rating_count_std': float(cluster_data['rating_count'].std()) if cluster_data['rating_count'].notna().any() else None,
            'rating_count_total': int(cluster_data['rating_count'].sum()) if cluster_data['rating_count'].notna().any() else 0,
            
            # Price statistics
            'price_mean': float(cluster_data['price'].mean()) if cluster_data['price'].notna().any() else None,
            'price_std': float(cluster_data['price'].std()) if cluster_data['price'].notna().any() else None,
            'price_min': float(cluster_data['price'].min()) if cluster_data['price'].notna().any() else None,
            'price_max': float(cluster_data['price'].max()) if cluster_data['price'].notna().any() else None,
            
            # Selling price statistics
            'selling_price_mean': float(cluster_data['selling_price'].mean()) if cluster_data['selling_price'].notna().any() else None,
            'selling_price_std': float(cluster_data['selling_price'].std()) if cluster_data['selling_price'].notna().any() else None,
            
            # Discount
            'discount_percent_mean': float(cluster_data['discount_percent'].mean()) if cluster_data['discount_percent'].notna().any() else None,
            
            # Top categories
            'top_category': cluster_data['category_name'].mode()[0] if len(cluster_data['category_name'].mode()) > 0 else None,
            'category_distribution': cluster_data['category_name'].value_counts().to_dict() if 'category_name' in cluster_data.columns else {},
            
            # Top brands
            'top_brand': cluster_data['brand'].mode()[0] if len(cluster_data['brand'].mode()) > 0 else None,
            'top_3_brands': cluster_data['brand'].value_counts().head(3).to_dict() if 'brand' in cluster_data.columns else {},
            
            # Availability
            'availability_rate': float(cluster_data['is_available'].mean()) if cluster_data['is_available'].notna().any() else None
        }
        
        cluster_stats.append(stats)
    
    logger.info(f"✅ Calculated statistics for {len(cluster_stats)} clusters")
    
    return cluster_stats


def visualize_cluster_statistics(cluster_stats):
    """Create visualizations for cluster statistics"""
    logger.info("Creating statistical visualizations...")
    
    df_stats = pd.DataFrame(cluster_stats)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Mean Rating Average per Cluster
    ax = fig.add_subplot(gs[0, 0])
    df_stats_sorted = df_stats.sort_values('rating_average_mean', ascending=False).head(20)
    ax.barh(range(len(df_stats_sorted)), df_stats_sorted['rating_average_mean'], color='#3498db', alpha=0.7)
    ax.set_yticks(range(len(df_stats_sorted)))
    ax.set_yticklabels(df_stats_sorted['cluster_id'].astype(int))
    ax.set_xlabel('Mean Rating Average', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cluster ID', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Clusters by Rating Average', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Mean Rating Count per Cluster
    ax = fig.add_subplot(gs[0, 1])
    df_stats_sorted = df_stats.sort_values('rating_count_mean', ascending=False).head(20)
    ax.barh(range(len(df_stats_sorted)), df_stats_sorted['rating_count_mean'], color='#e74c3c', alpha=0.7)
    ax.set_yticks(range(len(df_stats_sorted)))
    ax.set_yticklabels(df_stats_sorted['cluster_id'].astype(int))
    ax.set_xlabel('Mean Rating Count', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cluster ID', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Clusters by Rating Count', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Mean Price per Cluster
    ax = fig.add_subplot(gs[0, 2])
    df_stats_sorted = df_stats.sort_values('price_mean', ascending=False).head(20)
    ax.barh(range(len(df_stats_sorted)), df_stats_sorted['price_mean'], color='#2ecc71', alpha=0.7)
    ax.set_yticks(range(len(df_stats_sorted)))
    ax.set_yticklabels(df_stats_sorted['cluster_id'].astype(int))
    ax.set_xlabel('Mean Price', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cluster ID', fontsize=11, fontweight='bold')
    ax.set_title('Top 20 Clusters by Price', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Cluster Size Distribution
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(df_stats['cluster_id'], df_stats['total_products'], color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Cluster ID', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
    ax.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Rating Average Distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(df_stats['rating_average_mean'].dropna(), bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Rating Average', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Mean Rating Averages', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Price Distribution
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(df_stats['price_mean'].dropna(), bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Mean Price', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Clusters', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Mean Prices', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 7. Price vs Rating scatter
    ax = fig.add_subplot(gs[2, :2])
    scatter = ax.scatter(
        df_stats['price_mean'],
        df_stats['rating_average_mean'],
        s=df_stats['total_products']*2,
        c=df_stats['rating_count_mean'],
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Mean Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Rating Average', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Analysis: Price vs Rating\n(Size = cluster size, Color = rating count)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Mean Rating Count')
    
    # 8. Top Statistics Table
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Get top clusters
    top_rated = df_stats.nlargest(5, 'rating_average_mean')
    
    table_data = [['Metric', 'Value']]
    table_data.append(['Total Clusters', f"{len(df_stats)}"])
    table_data.append(['Total Products', f"{df_stats['total_products'].sum()}"])
    table_data.append(['', ''])
    table_data.append(['Best Rated Cluster', f"#{int(top_rated.iloc[0]['cluster_id'])}"])
    table_data.append(['  Rating', f"{top_rated.iloc[0]['rating_average_mean']:.2f}"])
    table_data.append(['  Size', f"{int(top_rated.iloc[0]['total_products'])}"])
    table_data.append(['', ''])
    table_data.append(['Avg Price Range', f"{df_stats['price_mean'].min():.0f} - {df_stats['price_mean'].max():.0f}"])
    table_data.append(['Avg Rating Range', f"{df_stats['rating_average_mean'].min():.2f} - {df_stats['rating_average_mean'].max():.2f}"])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(table_data)):
        cell = table[(i, 0)]
        cell.set_facecolor('#ecf0f1')
        if i == 0:
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
            table[(i, 1)].set_facecolor('#34495e')
            table[(i, 1)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    # 9. Heatmap of key metrics
    ax = fig.add_subplot(gs[3, :])
    
    # Prepare data for heatmap (top 30 clusters)
    top_30 = df_stats.nlargest(30, 'total_products')
    heatmap_data = top_30[['cluster_id', 'rating_average_mean', 'rating_count_mean', 'price_mean']].set_index('cluster_id')
    
    # Normalize for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    heatmap_normalized = pd.DataFrame(
        scaler.fit_transform(heatmap_data.fillna(0)),
        columns=heatmap_data.columns,
        index=heatmap_data.index
    )
    
    sns.heatmap(
        heatmap_normalized.T,
        annot=False,
        cmap='RdYlGn',
        center=0,
        cbar_kws={'label': 'Normalized Value'},
        ax=ax
    )
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    ax.set_yticklabels(['Rating Average', 'Rating Count', 'Price'], rotation=0)
    ax.set_title('Top 30 Clusters - Normalized Metrics Heatmap\n(Green=High, Red=Low)', 
                fontsize=13, fontweight='bold')
    
    plt.suptitle('Cluster Statistics Analysis', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('fine_grained_results/06_cluster_statistics.png', dpi=300, bbox_inches='tight')
    logger.info("✅ Saved: fine_grained_results/06_cluster_statistics.png")
    plt.close()


def create_detailed_statistics_report(cluster_stats):
    """Create detailed statistics report"""
    logger.info("Creating detailed statistics report...")
    
    df_stats = pd.DataFrame(cluster_stats)
    
    # Create detailed CSV
    df_stats.to_csv('fine_grained_results/detailed_cluster_statistics.csv', index=False)
    logger.info("✅ Saved: fine_grained_results/detailed_cluster_statistics.csv")
    
    # Create summary report
    summary = {
        'overall_statistics': {
            'total_clusters': len(cluster_stats),
            'total_products': sum(s['total_products'] for s in cluster_stats),
            'average_cluster_size': np.mean([s['total_products'] for s in cluster_stats]),
            'median_cluster_size': np.median([s['total_products'] for s in cluster_stats])
        },
        'rating_statistics': {
            'overall_mean_rating_average': float(df_stats['rating_average_mean'].mean()),
            'overall_mean_rating_count': float(df_stats['rating_count_mean'].mean()),
            'best_rated_cluster': {
                'cluster_id': int(df_stats.loc[df_stats['rating_average_mean'].idxmax(), 'cluster_id']),
                'rating': float(df_stats['rating_average_mean'].max())
            },
            'most_reviewed_cluster': {
                'cluster_id': int(df_stats.loc[df_stats['rating_count_mean'].idxmax(), 'cluster_id']),
                'avg_reviews': float(df_stats['rating_count_mean'].max())
            }
        },
        'price_statistics': {
            'overall_mean_price': float(df_stats['price_mean'].mean()),
            'price_range': {
                'min': float(df_stats['price_mean'].min()),
                'max': float(df_stats['price_mean'].max())
            },
            'highest_price_cluster': {
                'cluster_id': int(df_stats.loc[df_stats['price_mean'].idxmax(), 'cluster_id']),
                'mean_price': float(df_stats['price_mean'].max())
            },
            'lowest_price_cluster': {
                'cluster_id': int(df_stats.loc[df_stats['price_mean'].idxmin(), 'cluster_id']),
                'mean_price': float(df_stats['price_mean'].min())
            }
        },
        'cluster_statistics': cluster_stats
    }
    
    with open('fine_grained_results/statistics_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ Saved: fine_grained_results/statistics_summary.json")
    
    return summary


def print_summary(summary):
    """Print summary to console"""
    print("\n" + "=" * 80)
    print("CLUSTER STATISTICS SUMMARY")
    print("=" * 80)
    print(f"\nOverall:")
    print(f"  Total Clusters: {summary['overall_statistics']['total_clusters']}")
    print(f"  Total Products: {summary['overall_statistics']['total_products']:,}")
    print(f"  Average Cluster Size: {summary['overall_statistics']['average_cluster_size']:.1f}")
    
    print(f"\nRating Statistics:")
    print(f"  Overall Mean Rating: {summary['rating_statistics']['overall_mean_rating_average']:.2f}")
    print(f"  Overall Mean Review Count: {summary['rating_statistics']['overall_mean_rating_count']:.1f}")
    print(f"  Best Rated Cluster: #{summary['rating_statistics']['best_rated_cluster']['cluster_id']} "
          f"(Rating: {summary['rating_statistics']['best_rated_cluster']['rating']:.2f})")
    print(f"  Most Reviewed Cluster: #{summary['rating_statistics']['most_reviewed_cluster']['cluster_id']} "
          f"({summary['rating_statistics']['most_reviewed_cluster']['avg_reviews']:.0f} reviews)")
    
    print(f"\nPrice Statistics:")
    print(f"  Overall Mean Price: {summary['price_statistics']['overall_mean_price']:,.0f}")
    print(f"  Price Range: {summary['price_statistics']['price_range']['min']:,.0f} - "
          f"{summary['price_statistics']['price_range']['max']:,.0f}")
    print(f"  Highest Price Cluster: #{summary['price_statistics']['highest_price_cluster']['cluster_id']} "
          f"({summary['price_statistics']['highest_price_cluster']['mean_price']:,.0f})")
    print(f"  Lowest Price Cluster: #{summary['price_statistics']['lowest_price_cluster']['cluster_id']} "
          f"({summary['price_statistics']['lowest_price_cluster']['mean_price']:,.0f})")
    
    print("\n" + "=" * 80)
    print("\nTop 10 Clusters by Rating Average:")
    print("-" * 80)
    
    df_stats = pd.DataFrame(summary['cluster_statistics'])
    top_10_rating = df_stats.nlargest(10, 'rating_average_mean')
    
    for idx, row in top_10_rating.iterrows():
        print(f"Cluster {row['cluster_id']:3d}: Rating={row['rating_average_mean']:5.2f}, "
              f"Reviews={row['rating_count_mean']:6.0f}, "
              f"Price={row['price_mean']:10,.0f}, "
              f"Size={row['total_products']:4d} products")
    
    print("\n" + "=" * 80)


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("CLUSTER STATISTICS ANALYSIS")
    logger.info("=" * 80)
    
    # Load data
    df, metadata = load_data()
    
    logger.info(f"\nClustering Metadata:")
    logger.info(f"  k clusters: {metadata.get('k_clusters', 'N/A')}")
    logger.info(f"  Model: {metadata.get('model', 'N/A')}")
    
    # Calculate statistics
    cluster_stats = calculate_cluster_statistics(df)
    
    # Create visualizations
    visualize_cluster_statistics(cluster_stats)
    
    # Create detailed report
    summary = create_detailed_statistics_report(cluster_stats)
    
    # Print summary
    print_summary(summary)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info("Generated files:")
    logger.info("  • fine_grained_results/detailed_cluster_statistics.csv")
    logger.info("  • fine_grained_results/statistics_summary.json")
    logger.info("  • fine_grained_results/06_cluster_statistics.png")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

