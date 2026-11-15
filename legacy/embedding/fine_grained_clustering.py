#!/usr/bin/env python3
"""
Fine-Grained K-Means Clustering (k=150-200)
Detailed product segmentation using embeddings
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FineGrainedClustering:
    """Fine-grained clustering analysis with k=150-200"""
    
    def __init__(self, embeddings_path='embeddings', output_dir='fine_grained_results'):
        self.embeddings_path = embeddings_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.embeddings = None
        self.df = None
        self.best_model = None
        self.best_k = None
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def load_data(self):
        """Load embeddings and product data"""
        logger.info("Loading embeddings and product data...")
        
        self.embeddings = np.load(f'{self.embeddings_path}/embeddings.npy')
        
        with open(f'{self.embeddings_path}/embeddings_full.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.df = pd.DataFrame(data['products'])
        
        logger.info(f"✅ Loaded: {len(self.embeddings)} products, {self.embeddings.shape[1]} dimensions")
        return self.embeddings
    
    def find_optimal_k(self, k_range=range(150, 201, 5)):
        """Find optimal k in the fine-grained range"""
        logger.info(f"Finding optimal k in range {min(k_range)}-{max(k_range)}...")
        logger.info("This will take approximately 20-30 minutes...")
        
        results = []
        
        for k in tqdm(k_range, desc="Evaluating k values"):
            logger.info(f"  Evaluating k={k}")
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(self.embeddings)
            
            silhouette = silhouette_score(self.embeddings, labels)
            davies_bouldin = davies_bouldin_score(self.embeddings, labels)
            calinski = calinski_harabasz_score(self.embeddings, labels)
            
            results.append({
                'k': k,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski,
                'inertia': kmeans.inertia_
            })
            
            logger.info(f"    Silhouette: {silhouette:.4f}, DB: {davies_bouldin:.4f}, CH: {calinski:.2f}")
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'{self.output_dir}/fine_grained_metrics.csv', index=False)
        
        # Find best k
        best_by_silhouette = df_results.loc[df_results['silhouette_score'].idxmax()]
        best_by_db = df_results.loc[df_results['davies_bouldin_score'].idxmin()]
        best_by_ch = df_results.loc[df_results['calinski_harabasz_score'].idxmax()]
        
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMAL K ANALYSIS (Fine-Grained)")
        logger.info("=" * 80)
        logger.info(f"🏆 Best by Silhouette: k={int(best_by_silhouette['k'])} (score={best_by_silhouette['silhouette_score']:.4f})")
        logger.info(f"🏆 Best by Davies-Bouldin: k={int(best_by_db['k'])} (score={best_by_db['davies_bouldin_score']:.4f})")
        logger.info(f"🏆 Best by Calinski-Harabasz: k={int(best_by_ch['k'])} (score={best_by_ch['calinski_harabasz_score']:.2f})")
        logger.info("=" * 80)
        
        # Use silhouette as primary metric
        self.best_k = int(best_by_silhouette['k'])
        
        return df_results, self.best_k
    
    def plot_metrics_trends(self, df_results):
        """Plot metric trends for fine-grained k range"""
        logger.info("Creating metrics trend plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Silhouette Score
        ax = axes[0, 0]
        ax.plot(df_results['k'], df_results['silhouette_score'], 'o-', linewidth=2, markersize=6, color='#2ecc71')
        best_idx = df_results['silhouette_score'].idxmax()
        best_k = df_results.loc[best_idx, 'k']
        ax.plot(best_k, df_results.loc[best_idx, 'silhouette_score'], 'r*', markersize=20, 
               label=f'Best: k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax.set_title('Silhouette Score (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Davies-Bouldin Index
        ax = axes[0, 1]
        ax.plot(df_results['k'], df_results['davies_bouldin_score'], 'o-', linewidth=2, markersize=6, color='#e74c3c')
        best_idx = df_results['davies_bouldin_score'].idxmin()
        best_k = df_results.loc[best_idx, 'k']
        ax.plot(best_k, df_results.loc[best_idx, 'davies_bouldin_score'], 'g*', markersize=20,
               label=f'Best: k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
        ax.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Calinski-Harabasz Score
        ax = axes[1, 0]
        ax.plot(df_results['k'], df_results['calinski_harabasz_score'], 'o-', linewidth=2, markersize=6, color='#3498db')
        best_idx = df_results['calinski_harabasz_score'].idxmax()
        best_k = df_results.loc[best_idx, 'k']
        ax.plot(best_k, df_results.loc[best_idx, 'calinski_harabasz_score'], 'r*', markersize=20,
               label=f'Best: k={int(best_k)}')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
        ax.set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Inertia
        ax = axes[1, 1]
        ax.plot(df_results['k'], df_results['inertia'], 'o-', linewidth=2, markersize=6, color='#9b59b6')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inertia', fontsize=12, fontweight='bold')
        ax.set_title('Inertia (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Fine-Grained Clustering Analysis (k=150-200)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_metrics_trends.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/01_metrics_trends.png")
        plt.close()
    
    def plot_pca_best_k(self, k):
        """PCA visualization for best k"""
        logger.info(f"Creating PCA visualization for k={k}...")
        
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(self.embeddings)
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab20c',
            alpha=0.6,
            s=15,
            edgecolors='none'
        )
        
        silhouette = silhouette_score(self.embeddings, labels)
        
        ax.set_title(f'PCA Visualization - K-Means (k={k})\nSilhouette Score: {silhouette:.4f}', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=13)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=13)
        
        plt.colorbar(scatter, ax=ax, label='Cluster ID', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_pca_visualization.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/02_pca_visualization.png")
        plt.close()
        
        return labels
    
    def plot_cluster_size_distribution(self, k, labels):
        """Analyze cluster size distribution"""
        logger.info(f"Creating cluster size distribution (k={k})...")
        
        unique, counts = np.unique(labels, return_counts=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
        ax1.bar(unique, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
        ax1.set_title(f'Cluster Size Distribution (k={k})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_size = counts.mean()
        ax1.axhline(y=mean_size, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_size:.0f}')
        ax1.legend(fontsize=11)
        
        # Histogram
        ax2.hist(counts, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Cluster Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title(f'Distribution of Cluster Sizes (k={k})', fontsize=14, fontweight='bold')
        ax2.axvline(x=mean_size, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_size:.0f}')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=11)
        
        # Statistics
        logger.info(f"Cluster size statistics:")
        logger.info(f"  Mean: {counts.mean():.1f}")
        logger.info(f"  Std: {counts.std():.1f}")
        logger.info(f"  Min: {counts.min()}")
        logger.info(f"  Max: {counts.max()}")
        logger.info(f"  Median: {np.median(counts):.1f}")
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_cluster_size_distribution.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/03_cluster_size_distribution.png")
        plt.close()
    
    def plot_tsne_best_k(self, k, n_samples=8000):
        """t-SNE visualization for best k"""
        logger.info(f"Creating t-SNE visualization for k={k}...")
        
        # Use subset for faster t-SNE
        if len(self.embeddings) > n_samples:
            indices = np.random.choice(len(self.embeddings), n_samples, replace=False)
            embeddings_subset = self.embeddings[indices]
            df_subset = self.df.iloc[indices]
        else:
            embeddings_subset = self.embeddings
            df_subset = self.df
        
        # K-Means on subset
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels_subset = kmeans.fit_predict(embeddings_subset)
        
        # t-SNE
        logger.info("  Computing t-SNE (may take 3-5 minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, n_jobs=-1)
        embeddings_tsne = tsne.fit_transform(embeddings_subset)
        
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        
        scatter = ax.scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            c=labels_subset,
            cmap='tab20c',
            alpha=0.7,
            s=25,
            edgecolors='black',
            linewidth=0.3
        )
        
        ax.set_title(f't-SNE Visualization - K-Means (k={k})\n{len(embeddings_subset)} products', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
        
        plt.colorbar(scatter, ax=ax, label='Cluster ID', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_tsne_visualization.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/04_tsne_visualization.png")
        plt.close()
    
    def analyze_cluster_quality(self, k, labels):
        """Detailed cluster quality analysis"""
        logger.info(f"Analyzing cluster quality for k={k}...")
        
        # Per-cluster silhouette scores
        sample_silhouette_values = silhouette_samples(self.embeddings, labels)
        
        cluster_silhouettes = []
        cluster_sizes = []
        
        for i in range(k):
            cluster_mask = labels == i
            cluster_silhouettes.append(sample_silhouette_values[cluster_mask].mean())
            cluster_sizes.append(cluster_mask.sum())
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Cluster quality
        colors = ['green' if s > 0.3 else 'orange' if s > 0.15 else 'red' for s in cluster_silhouettes]
        ax1.bar(range(k), cluster_silhouettes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax1.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Weak threshold')
        ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Good threshold')
        ax1.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Silhouette Score', fontsize=12, fontweight='bold')
        ax1.set_title(f'Cluster Quality Analysis (k={k})\nGreen=Good, Orange=Weak, Red=Poor', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Size vs Quality scatter
        ax2.scatter(cluster_sizes, cluster_silhouettes, s=100, alpha=0.6, c=colors, edgecolors='black')
        ax2.set_xlabel('Cluster Size (number of products)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Silhouette Score', fontsize=12, fontweight='bold')
        ax2.set_title('Cluster Size vs Quality', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax2.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.3, color='green', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_cluster_quality.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/05_cluster_quality.png")
        plt.close()
        
        # Quality statistics
        good_clusters = sum(1 for s in cluster_silhouettes if s > 0.3)
        weak_clusters = sum(1 for s in cluster_silhouettes if 0.15 < s <= 0.3)
        poor_clusters = sum(1 for s in cluster_silhouettes if s <= 0.15)
        
        logger.info(f"Cluster quality breakdown:")
        logger.info(f"  Good clusters (>0.3): {good_clusters} ({good_clusters/k*100:.1f}%)")
        logger.info(f"  Weak clusters (0.15-0.3): {weak_clusters} ({weak_clusters/k*100:.1f}%)")
        logger.info(f"  Poor clusters (≤0.15): {poor_clusters} ({poor_clusters/k*100:.1f}%)")
    
    def create_final_model(self, k):
        """Train final model and save results"""
        logger.info(f"Training final K-Means model with k={k}...")
        
        self.best_model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = self.best_model.fit_predict(self.embeddings)
        
        # Calculate metrics
        silhouette = silhouette_score(self.embeddings, labels)
        davies_bouldin = davies_bouldin_score(self.embeddings, labels)
        calinski = calinski_harabasz_score(self.embeddings, labels)
        
        # Add cluster labels to products
        self.df['cluster'] = labels
        
        # Save clustered products
        products_with_clusters = []
        for idx, row in self.df.iterrows():
            products_with_clusters.append({
                'id': int(row['id']) if pd.notna(row['id']) else None,
                'digikala_id': int(row['digikala_id']) if pd.notna(row['digikala_id']) else None,
                'title': row['title'],
                'brand': row['brand'],
                'category_name': row.get('category_name'),
                'cluster': int(labels[idx])
            })
        
        # Save detailed results
        output = {
            'metadata': {
                'model': 'K-Means on HuggingFace Embeddings',
                'embedding_model': 'paraphrase-multilingual-MiniLM-L12-v2',
                'k_clusters': k,
                'total_products': len(labels),
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski),
                'inertia': float(self.best_model.inertia_)
            },
            'products': products_with_clusters
        }
        
        with open(f'{self.output_dir}/final_clustered_products.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved: {self.output_dir}/final_clustered_products.json")
        
        # Save cluster statistics
        cluster_stats = []
        for cluster_id in range(k):
            cluster_products = self.df[self.df['cluster'] == cluster_id]
            
            stats = {
                'cluster_id': cluster_id,
                'size': len(cluster_products),
                'top_brands': cluster_products['brand'].value_counts().head(5).to_dict() if 'brand' in cluster_products.columns else {},
                'categories': cluster_products['category_name'].value_counts().to_dict() if 'category_name' in cluster_products.columns else {}
            }
            cluster_stats.append(stats)
        
        with open(f'{self.output_dir}/cluster_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved: {self.output_dir}/cluster_statistics.json")
        
        return labels
    
    def create_summary_report(self, df_metrics, best_k, labels):
        """Create final summary report"""
        logger.info("Creating summary report...")
        
        silhouette = silhouette_score(self.embeddings, labels)
        davies_bouldin = davies_bouldin_score(self.embeddings, labels)
        calinski = calinski_harabasz_score(self.embeddings, labels)
        
        summary = {
            'analysis_type': 'Fine-Grained K-Means Clustering',
            'k_range_tested': f"{df_metrics['k'].min()}-{df_metrics['k'].max()}",
            'total_configurations_tested': len(df_metrics),
            'best_configuration': {
                'k': best_k,
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'calinski_harabasz_score': float(calinski)
            },
            'all_metrics': df_metrics.to_dict('records')
        }
        
        with open(f'{self.output_dir}/analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved: {self.output_dir}/analysis_summary.json")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY - FINE-GRAINED CLUSTERING")
        logger.info("=" * 80)
        logger.info(f"Tested k values: {df_metrics['k'].min()} to {df_metrics['k'].max()}")
        logger.info(f"Best k: {best_k}")
        logger.info(f"\nFinal Model Performance:")
        logger.info(f"  ✅ Silhouette Score: {silhouette:.4f}")
        logger.info(f"  ✅ Davies-Bouldin Index: {davies_bouldin:.4f}")
        logger.info(f"  ✅ Calinski-Harabasz Score: {calinski:.2f}")
        logger.info(f"\nCluster Statistics:")
        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"  Total Clusters: {len(unique)}")
        logger.info(f"  Avg Cluster Size: {counts.mean():.1f}")
        logger.info(f"  Largest Cluster: {counts.max()} products")
        logger.info(f"  Smallest Cluster: {counts.min()} products")
        logger.info("=" * 80)
    
    def run_complete_analysis(self):
        """Run complete fine-grained clustering analysis"""
        logger.info("=" * 80)
        logger.info("FINE-GRAINED K-MEANS CLUSTERING ANALYSIS (k=150-200)")
        logger.info("=" * 80)
        logger.info("This analysis will take approximately 30-45 minutes")
        logger.info("=" * 80)
        
        # Load data
        self.load_data()
        
        # Find optimal k
        df_metrics, best_k = self.find_optimal_k(k_range=range(150, 201, 5))
        
        # Plot metrics trends
        self.plot_metrics_trends(df_metrics)
        
        # Create final model with best k
        labels = self.create_final_model(best_k)
        
        # Generate visualizations
        self.plot_pca_best_k(best_k)
        self.plot_cluster_size_distribution(best_k, labels)
        self.analyze_cluster_quality(best_k, labels)
        self.plot_tsne_best_k(best_k, n_samples=8000)
        
        # Create summary
        self.create_summary_report(df_metrics, best_k, labels)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ FINE-GRAINED CLUSTERING ANALYSIS COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📁 Results saved in: {self.output_dir}/")
        logger.info(f"🏆 Best k value: {best_k}")
        logger.info("=" * 80)
        
        # List all generated files
        files = sorted([f for f in os.listdir(self.output_dir)])
        logger.info("\nGenerated files:")
        for f in files:
            logger.info(f"  • {f}")


def main():
    """Main function"""
    
    analyzer = FineGrainedClustering(
        embeddings_path='embeddings',
        output_dir='fine_grained_results'
    )
    
    analyzer.run_complete_analysis()
    
    return 0


if __name__ == '__main__':
    exit(main())

