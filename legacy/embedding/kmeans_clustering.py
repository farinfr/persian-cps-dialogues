#!/usr/bin/env python3
"""
K-Means Clustering on Product Embeddings
Analyze products using embeddings from HuggingFace models
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
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingKMeansAnalysis:
    """K-Means clustering analysis on product embeddings"""
    
    def __init__(self, embeddings_path='embeddings', output_dir='clustering_plots'):
        self.embeddings_path = embeddings_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.embeddings = None
        self.metadata = None
        self.df = None
        self.results = {}
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def load_embeddings(self):
        """Load embeddings and metadata"""
        logger.info("Loading embeddings...")
        
        # Load embeddings array
        embeddings_file = f'{self.embeddings_path}/embeddings.npy'
        self.embeddings = np.load(embeddings_file)
        logger.info(f"✅ Loaded embeddings: {self.embeddings.shape}")
        
        # Load metadata
        with open(f'{self.embeddings_path}/metadata.json', 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load full product data
        with open(f'{self.embeddings_path}/embeddings_full.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.df = pd.DataFrame(data['products'])
        
        logger.info(f"Embedding dimension: {self.embeddings.shape[1]}")
        logger.info(f"Total products: {len(self.embeddings)}")
        
        return self.embeddings
    
    def plot_elbow_method(self, k_range=range(2, 31)):
        """Plot elbow curve for optimal k"""
        logger.info("Creating elbow curve...")
        
        inertias = []
        silhouettes = []
        
        for k in k_range:
            logger.info(f"  Computing k={k}")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.embeddings, labels))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Inertia plot
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=14, fontweight='bold')
        ax1.set_title('Elbow Method - K-Means on Embeddings\n(Look for the "elbow" point)', 
                     fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Mark some key points
        for k in [5, 10, 15, 20]:
            if k in k_range:
                idx = list(k_range).index(k)
                ax1.plot(k, inertias[idx], 'ro', markersize=12)
                ax1.annotate(f'k={k}', (k, inertias[idx]), 
                           textcoords="offset points", xytext=(0,10), ha='center')
        
        # Silhouette plot
        ax2.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=14, fontweight='bold')
        ax2.set_title('Silhouette Score vs Number of Clusters\n(Higher is Better)', 
                     fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Mark best silhouette
        best_idx = np.argmax(silhouettes)
        best_k = list(k_range)[best_idx]
        ax2.plot(best_k, silhouettes[best_idx], 'r*', markersize=20, 
                label=f'Best: k={best_k} (Score={silhouettes[best_idx]:.3f})')
        ax2.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_elbow_curve.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/01_elbow_curve.png")
        plt.close()
        
        return silhouettes
    
    def plot_pca_visualization(self, k_values=[5, 10, 15, 20]):
        """Visualize clusters using PCA"""
        logger.info("Creating PCA visualizations...")
        
        # Reduce to 2D
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        axes = axes.flatten()
        
        for idx, k in enumerate(k_values):
            logger.info(f"  PCA visualization for k={k}")
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            silhouette = silhouette_score(self.embeddings, labels)
            
            ax = axes[idx]
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='tab20',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
            
            ax.set_title(f'K-Means Clustering (k={k})\nSilhouette Score: {silhouette:.4f}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            
            plt.colorbar(scatter, ax=ax, label='Cluster', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_pca_clusters.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/02_pca_clusters.png")
        plt.close()
    
    def plot_tsne_visualization(self, k=20, n_samples=5000):
        """Visualize clusters using t-SNE"""
        logger.info(f"Creating t-SNE visualization (k={k})...")
        
        # Use subset for faster computation
        if len(self.embeddings) > n_samples:
            indices = np.random.choice(len(self.embeddings), n_samples, replace=False)
            embeddings_subset = self.embeddings[indices]
            df_subset = self.df.iloc[indices]
        else:
            embeddings_subset = self.embeddings
            df_subset = self.df
            indices = np.arange(len(self.embeddings))
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_subset)
        
        # t-SNE
        logger.info("  Computing t-SNE (this may take a few minutes)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        embeddings_tsne = tsne.fit_transform(embeddings_subset)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # By cluster
        scatter1 = ax1.scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.7,
            s=30,
            edgecolors='black',
            linewidth=0.5
        )
        ax1.set_title(f't-SNE Visualization - K-Means Clusters (k={k})\n{len(embeddings_subset)} products', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        # By category (if available)
        if 'category_name' in df_subset.columns:
            categories = df_subset['category_name'].fillna('Unknown')
            unique_cats = categories.unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
            
            for cat_idx, cat in enumerate(unique_cats):
                mask = categories == cat
                ax2.scatter(
                    embeddings_tsne[mask, 0],
                    embeddings_tsne[mask, 1],
                    c=[colors[cat_idx]],
                    label=cat[:30],
                    alpha=0.7,
                    s=30,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            ax2.set_title(f't-SNE Visualization - By Original Category\n{len(embeddings_subset)} products', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_tsne_visualization.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/03_tsne_visualization.png")
        plt.close()
    
    def plot_silhouette_analysis(self, k_values=[10, 15, 20, 25]):
        """Detailed silhouette analysis"""
        logger.info("Creating silhouette analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, k in enumerate(k_values):
            logger.info(f"  Silhouette analysis for k={k}")
            
            ax = axes[idx]
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            silhouette_avg = silhouette_score(self.embeddings, labels)
            sample_silhouette_values = silhouette_samples(self.embeddings, labels)
            
            y_lower = 10
            for i in range(k):
                ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
                ith_cluster_silhouette_values.sort()
                
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = plt.cm.nipy_spectral(float(i) / k)
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7
                )
                
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10
            
            ax.set_title(f'Silhouette Analysis (k={k})\nAverage Score: {silhouette_avg:.4f}', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Silhouette Coefficient', fontsize=11)
            ax.set_ylabel('Cluster', fontsize=11)
            ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2)
            ax.set_yticks([])
            ax.set_xlim([-0.2, 1])
        
        plt.suptitle('Silhouette Analysis - K-Means on Embeddings', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_silhouette_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/04_silhouette_analysis.png")
        plt.close()
    
    def plot_cluster_distribution(self, k_values=[10, 15, 20, 25]):
        """Plot cluster size distributions"""
        logger.info("Creating cluster distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, k in enumerate(k_values):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            unique, counts = np.unique(labels, return_counts=True)
            
            ax = axes[idx]
            bars = ax.bar(
                unique,
                counts,
                color=plt.cm.viridis(np.linspace(0, 1, len(unique))),
                edgecolor='black',
                alpha=0.8
            )
            
            ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
            ax.set_title(f'Cluster Distribution (k={k})\nTotal Products: {len(labels):,}', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
            
            # Add statistics
            mean_size = counts.mean()
            std_size = counts.std()
            ax.axhline(y=mean_size, color='r', linestyle='--', alpha=0.5, 
                      label=f'Mean: {mean_size:.0f}±{std_size:.0f}')
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_cluster_distribution.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/05_cluster_distribution.png")
        plt.close()
    
    def plot_metrics_comparison(self, k_range=range(2, 31)):
        """Compare clustering metrics across k values"""
        logger.info("Creating metrics comparison...")
        
        results = []
        
        for k in k_range:
            logger.info(f"  Evaluating k={k}")
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            results.append({
                'k': k,
                'silhouette': silhouette_score(self.embeddings, labels),
                'davies_bouldin': davies_bouldin_score(self.embeddings, labels),
                'calinski_harabasz': calinski_harabasz_score(self.embeddings, labels),
                'inertia': kmeans.inertia_
            })
        
        df_results = pd.DataFrame(results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Silhouette
        ax = axes[0, 0]
        ax.plot(df_results['k'], df_results['silhouette'], 'o-', linewidth=2, markersize=8, color='#2ecc71')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax.set_title('Silhouette Score (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        best_k = df_results.loc[df_results['silhouette'].idxmax(), 'k']
        ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.5, label=f'Best: k={int(best_k)}')
        ax.legend()
        
        # Davies-Bouldin
        ax = axes[0, 1]
        ax.plot(df_results['k'], df_results['davies_bouldin'], 'o-', linewidth=2, markersize=8, color='#e74c3c')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
        ax.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        best_k = df_results.loc[df_results['davies_bouldin'].idxmin(), 'k']
        ax.axvline(x=best_k, color='g', linestyle='--', alpha=0.5, label=f'Best: k={int(best_k)}')
        ax.legend()
        
        # Calinski-Harabasz
        ax = axes[1, 0]
        ax.plot(df_results['k'], df_results['calinski_harabasz'], 'o-', linewidth=2, markersize=8, color='#3498db')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Calinski-Harabasz Score', fontsize=12, fontweight='bold')
        ax.set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        best_k = df_results.loc[df_results['calinski_harabasz'].idxmax(), 'k']
        ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.5, label=f'Best: k={int(best_k)}')
        ax.legend()
        
        # Inertia
        ax = axes[1, 1]
        ax.plot(df_results['k'], df_results['inertia'], 'o-', linewidth=2, markersize=8, color='#9b59b6')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Inertia', fontsize=12, fontweight='bold')
        ax.set_title('Inertia (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Clustering Metrics Comparison - K-Means on Embeddings', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_metrics_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/06_metrics_comparison.png")
        plt.close()
        
        # Save results
        df_results.to_csv(f'{self.output_dir}/kmeans_metrics.csv', index=False)
        logger.info(f"✅ Saved: {self.output_dir}/kmeans_metrics.csv")
        
        return df_results
    
    def plot_cluster_heatmap(self, k=20):
        """Create cluster similarity heatmap"""
        logger.info(f"Creating cluster similarity heatmap (k={k})...")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        # Calculate cluster centers similarity
        centers = kmeans.cluster_centers_
        similarity_matrix = cosine_similarity(centers)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Cosine Similarity'},
            ax=ax
        )
        
        ax.set_title(f'Cluster Similarity Matrix (k={k})\nBased on Cluster Centroids', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Cluster ID', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_cluster_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/07_cluster_similarity_heatmap.png")
        plt.close()
    
    def plot_cluster_profiles(self, k=20):
        """Analyze cluster profiles by category"""
        logger.info(f"Creating cluster profiles (k={k})...")
        
        if 'category_name' not in self.df.columns:
            logger.warning("Category information not available, skipping cluster profiles")
            return
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        # Add cluster labels to dataframe
        df_with_clusters = self.df.copy()
        df_with_clusters['cluster'] = labels
        
        # Create cross-tabulation
        crosstab = pd.crosstab(
            df_with_clusters['category_name'],
            df_with_clusters['cluster'],
            normalize='columns'
        ) * 100
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        sns.heatmap(
            crosstab,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Percentage (%)'},
            ax=ax
        )
        
        ax.set_title(f'Cluster Composition by Category (k={k})\nPercentage of each category in each cluster', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Product Category', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_cluster_profiles.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/08_cluster_profiles.png")
        plt.close()
        
        # Save detailed profile
        cluster_stats = []
        for cluster_id in range(k):
            cluster_products = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            category_dist = cluster_products['category_name'].value_counts().to_dict()
            top_brands = cluster_products['brand'].value_counts().head(5).to_dict()
            
            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'size': len(cluster_products),
                'category_distribution': category_dist,
                'top_brands': top_brands
            })
        
        with open(f'{self.output_dir}/cluster_profiles.json', 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved: {self.output_dir}/cluster_profiles.json")
    
    def plot_3d_visualization(self, k=20):
        """Create 3D PCA visualization"""
        logger.info(f"Creating 3D visualization (k={k})...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # PCA to 3D
        pca = PCA(n_components=3, random_state=42)
        embeddings_3d = pca.fit_transform(self.embeddings)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            embeddings_3d[:, 0],
            embeddings_3d[:, 1],
            embeddings_3d[:, 2],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, fontweight='bold')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12, fontweight='bold')
        ax.set_title(f'3D PCA Visualization - K-Means (k={k})', fontsize=14, fontweight='bold', pad=20)
        
        plt.colorbar(scatter, ax=ax, label='Cluster', pad=0.1, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_3d_visualization.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/09_3d_visualization.png")
        plt.close()
    
    def analyze_best_k(self):
        """Find and analyze the best k value"""
        logger.info("Finding optimal k value...")
        
        k_range = range(2, 31)
        results = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            silhouette = silhouette_score(self.embeddings, labels)
            davies_bouldin = davies_bouldin_score(self.embeddings, labels)
            calinski = calinski_harabasz_score(self.embeddings, labels)
            
            results.append({
                'k': k,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski,
                'inertia': kmeans.inertia_
            })
        
        df_results = pd.DataFrame(results)
        
        # Find best k by different metrics
        best_silhouette = df_results.loc[df_results['silhouette'].idxmax()]
        best_db = df_results.loc[df_results['davies_bouldin'].idxmin()]
        best_ch = df_results.loc[df_results['calinski_harabasz'].idxmax()]
        
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMAL K VALUE ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Best by Silhouette Score: k={int(best_silhouette['k'])} (score={best_silhouette['silhouette']:.4f})")
        logger.info(f"Best by Davies-Bouldin: k={int(best_db['k'])} (score={best_db['davies_bouldin']:.4f})")
        logger.info(f"Best by Calinski-Harabasz: k={int(best_ch['k'])} (score={best_ch['calinski_harabasz']:.2f})")
        logger.info("=" * 80)
        
        return df_results, best_silhouette
    
    def create_final_clustering(self, k=None):
        """Create final clustering model and save results"""
        
        if k is None:
            # Auto-detect best k
            df_results, best = self.analyze_best_k()
            k = int(best['k'])
            logger.info(f"\nUsing auto-detected optimal k={k}")
        
        logger.info(f"Creating final clustering model with k={k}...")
        
        # Train final model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)
        
        # Add to dataframe
        self.df['cluster'] = labels
        
        # Calculate metrics
        silhouette = silhouette_score(self.embeddings, labels)
        davies_bouldin = davies_bouldin_score(self.embeddings, labels)
        calinski = calinski_harabasz_score(self.embeddings, labels)
        
        # Save clustered products
        output_data = []
        for idx, row in self.df.iterrows():
            output_data.append({
                'id': int(row['id']) if pd.notna(row['id']) else None,
                'digikala_id': int(row['digikala_id']) if pd.notna(row['digikala_id']) else None,
                'title': row['title'],
                'brand': row['brand'],
                'category_name': row.get('category_name'),
                'cluster': int(labels[idx])
            })
        
        with open(f'{self.output_dir}/clustered_products.json', 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'k': k,
                    'total_products': len(labels),
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'calinski_harabasz_score': calinski,
                    'model': 'K-Means on HuggingFace embeddings'
                },
                'products': output_data
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved: {self.output_dir}/clustered_products.json")
        logger.info("\nFinal Model Metrics:")
        logger.info(f"  Silhouette Score: {silhouette:.4f}")
        logger.info(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
        logger.info(f"  Calinski-Harabasz Score: {calinski:.2f}")
        
        return kmeans, labels
    
    def run_complete_analysis(self):
        """Run complete K-Means clustering analysis"""
        logger.info("=" * 80)
        logger.info("K-MEANS CLUSTERING ON EMBEDDINGS - COMPLETE ANALYSIS")
        logger.info("=" * 80)
        
        # Load embeddings
        self.load_embeddings()
        
        # Generate all plots
        logger.info("\nGenerating analysis plots...")
        logger.info("-" * 80)
        
        self.plot_elbow_method(k_range=range(2, 31))
        self.plot_metrics_comparison(k_range=range(2, 31))
        self.plot_pca_visualization(k_values=[5, 10, 15, 20])
        self.plot_silhouette_analysis(k_values=[10, 15, 20, 25])
        self.plot_cluster_distribution(k_values=[10, 15, 20, 25])
        self.plot_cluster_heatmap(k=20)
        self.plot_cluster_profiles(k=20)
        self.plot_3d_visualization(k=20)
        self.plot_tsne_visualization(k=20, n_samples=5000)
        
        # Find best k and create final model
        logger.info("\n" + "-" * 80)
        df_metrics, best = self.analyze_best_k()
        
        # Create final clustering
        kmeans, labels = self.create_final_clustering(k=int(best['k']))
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ANALYSIS COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"📁 All plots saved in: {self.output_dir}/")
        logger.info(f"🏆 Best k value: {int(best['k'])}")
        logger.info(f"📊 Silhouette Score: {best['silhouette']:.4f}")
        logger.info("=" * 80)


def main():
    """Main function"""
    
    analyzer = EmbeddingKMeansAnalysis(
        embeddings_path='embeddings',
        output_dir='clustering_plots'
    )
    
    analyzer.run_complete_analysis()
    
    return 0


if __name__ == '__main__':
    exit(main())

