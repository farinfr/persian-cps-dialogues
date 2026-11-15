#!/usr/bin/env python3
"""
Comprehensive visualization for clustering analysis
Generate all relevant plots for clustering evaluation
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusteringVisualizer:
    """Create comprehensive visualizations for clustering analysis"""
    
    def __init__(self, dataset_path, output_dir='plots'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        
        self.df = None
        self.features = None
        self.features_dense = None
    
    def load_and_prepare_data(self):
        """Load dataset and prepare features"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = data.get('products', data)
        self.df = pd.DataFrame(products)
        
        logger.info(f"Loaded {len(self.df)} products")
        
        # Prepare text
        self.df['combined_text'] = (
            self.df['title'].fillna('') + ' ' + 
            self.df['description'].fillna('')
        )
        
        # Vectorize
        logger.info("Creating TF-IDF features...")
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8, ngram_range=(1, 2))
        self.features = vectorizer.fit_transform(self.df['combined_text'])
        
        # Reduce dimensions
        logger.info("Reducing dimensions...")
        svd = TruncatedSVD(n_components=50, random_state=42)
        self.features_dense = svd.fit_transform(self.features)
        
        logger.info(f"Feature shape: {self.features_dense.shape}")
        
        return self.df, self.features_dense
    
    def plot_elbow_curve(self):
        """Plot elbow curve for K-Means"""
        logger.info("Creating Elbow Curve...")
        
        k_range = range(2, 21)
        inertias = []
        silhouettes = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.features_dense)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.features_dense, kmeans.labels_))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Inertia plot
        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        ax1.set_title('Elbow Method - K-Means Inertia', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette plot
        ax2.plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/elbow_curve.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/elbow_curve.png")
        plt.close()
    
    def plot_metrics_comparison(self):
        """Compare different clustering algorithms across metrics"""
        logger.info("Creating metrics comparison plot...")
        
        algorithms = [
            ('K-Means', [3, 5, 7, 10, 15, 20]),
            ('Hierarchical', [5, 10, 15, 20]),
            ('GMM', [5, 10, 15, 20])
        ]
        
        results_data = []
        
        for algo_name, k_values in algorithms:
            for k in k_values:
                logger.info(f"  Evaluating {algo_name} with k={k}")
                
                if algo_name == 'K-Means':
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                elif algo_name == 'Hierarchical':
                    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                elif algo_name == 'GMM':
                    model = GaussianMixture(n_components=k, random_state=42)
                
                labels = model.fit_predict(self.features_dense)
                
                silhouette = silhouette_score(self.features_dense, labels)
                davies_bouldin = davies_bouldin_score(self.features_dense, labels)
                calinski = calinski_harabasz_score(self.features_dense, labels)
                
                results_data.append({
                    'Algorithm': algo_name,
                    'k': k,
                    'Silhouette': silhouette,
                    'Davies-Bouldin': davies_bouldin,
                    'Calinski-Harabasz': calinski / 1000  # Scale for visibility
                })
        
        df_metrics = pd.DataFrame(results_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Silhouette score
        for algo in df_metrics['Algorithm'].unique():
            data = df_metrics[df_metrics['Algorithm'] == algo]
            axes[0].plot(data['k'], data['Silhouette'], 'o-', label=algo, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Silhouette Score', fontsize=12)
        axes[0].set_title('Silhouette Score Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Davies-Bouldin index
        for algo in df_metrics['Algorithm'].unique():
            data = df_metrics[df_metrics['Algorithm'] == algo]
            axes[1].plot(data['k'], data['Davies-Bouldin'], 'o-', label=algo, linewidth=2, markersize=8)
        
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Davies-Bouldin Index', fontsize=12)
        axes[1].set_title('Davies-Bouldin Index Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz score
        for algo in df_metrics['Algorithm'].unique():
            data = df_metrics[df_metrics['Algorithm'] == algo]
            axes[2].plot(data['k'], data['Calinski-Harabasz'], 'o-', label=algo, linewidth=2, markersize=8)
        
        axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[2].set_ylabel('Calinski-Harabasz Score (×1000)', fontsize=12)
        axes[2].set_title('Calinski-Harabasz Score Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/metrics_comparison.png")
        plt.close()
    
    def plot_pca_visualization(self):
        """Plot 2D and 3D PCA visualizations"""
        logger.info("Creating PCA visualizations...")
        
        # 2D PCA
        pca_2d = PCA(n_components=2, random_state=42)
        features_2d = pca_2d.fit_transform(self.features_dense)
        
        # 3D PCA
        pca_3d = PCA(n_components=3, random_state=42)
        features_3d = pca_3d.fit_transform(self.features_dense)
        
        # Apply K-Means with optimal k
        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.features_dense)
        
        # 2D visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # By category (if available)
        ax = axes[0]
        if 'category_name' in self.df.columns:
            categories = self.df['category_name'].fillna('Unknown')
            unique_cats = categories.unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cats)))
            
            for idx, cat in enumerate(unique_cats):
                mask = categories == cat
                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=[colors[idx]],
                    label=cat[:30],
                    alpha=0.6,
                    s=20
                )
            
            ax.set_title('PCA Visualization - By Original Category', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6, s=20)
            ax.set_title('PCA Visualization - All Products', fontsize=14, fontweight='bold')
        
        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        
        # By cluster
        ax = axes[1]
        scatter = ax.scatter(
            features_2d[:, 0],
            features_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        ax.set_title(f'PCA Visualization - K-Means Clusters (k=20)', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pca_visualization.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/pca_visualization.png")
        plt.close()
    
    def plot_silhouette_analysis(self, k_values=[5, 10, 15, 20]):
        """Create silhouette analysis plots"""
        logger.info("Creating silhouette analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, k in enumerate(k_values):
            ax = axes[idx]
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features_dense)
            
            # Compute silhouette scores
            silhouette_avg = silhouette_score(self.features_dense, labels)
            sample_silhouette_values = silhouette_samples(self.features_dense, labels)
            
            y_lower = 10
            for i in range(k):
                # Aggregate silhouette scores for samples belonging to cluster i
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
                
                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=10)
                
                y_lower = y_upper + 10
            
            ax.set_title(f'Silhouette Analysis (k={k})\nAvg Score: {silhouette_avg:.3f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Silhouette Coefficient', fontsize=11)
            ax.set_ylabel('Cluster', fontsize=11)
            
            # Vertical line for average silhouette score
            ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, label=f'Avg: {silhouette_avg:.3f}')
            ax.set_yticks([])
            ax.set_xlim([-0.1, 1])
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/silhouette_analysis.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/silhouette_analysis.png")
        plt.close()
    
    def plot_cluster_distribution(self):
        """Plot cluster size distributions for different algorithms"""
        logger.info("Creating cluster distribution plots...")
        
        algorithms = [
            ('K-Means k=10', KMeans(n_clusters=10, random_state=42, n_init=10)),
            ('K-Means k=15', KMeans(n_clusters=15, random_state=42, n_init=10)),
            ('K-Means k=20', KMeans(n_clusters=20, random_state=42, n_init=10)),
            ('Hierarchical k=15', AgglomerativeClustering(n_clusters=15, linkage='ward'))
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(algorithms):
            labels = model.fit_predict(self.features_dense)
            unique, counts = np.unique(labels, return_counts=True)
            
            ax = axes[idx]
            bars = ax.bar(unique, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique))), 
                         edgecolor='black', alpha=0.7)
            ax.set_xlabel('Cluster ID', fontsize=11)
            ax.set_ylabel('Number of Products', fontsize=11)
            ax.set_title(f'{name}\nCluster Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cluster_distribution.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/cluster_distribution.png")
        plt.close()
    
    def plot_algorithm_comparison_grid(self):
        """Create a grid comparing different algorithms visually"""
        logger.info("Creating algorithm comparison grid...")
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(self.features_dense)
        
        configs = [
            ('K-Means k=5', KMeans(n_clusters=5, random_state=42, n_init=10)),
            ('K-Means k=10', KMeans(n_clusters=10, random_state=42, n_init=10)),
            ('K-Means k=15', KMeans(n_clusters=15, random_state=42, n_init=10)),
            ('K-Means k=20', KMeans(n_clusters=20, random_state=42, n_init=10)),
            ('Hierarchical k=10', AgglomerativeClustering(n_clusters=10, linkage='ward')),
            ('Hierarchical k=20', AgglomerativeClustering(n_clusters=20, linkage='ward')),
            ('GMM k=10', GaussianMixture(n_components=10, random_state=42)),
            ('GMM k=20', GaussianMixture(n_components=20, random_state=42)),
            ('DBSCAN eps=0.5', DBSCAN(eps=0.5, min_samples=5))
        ]
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(configs):
            ax = axes[idx]
            labels = model.fit_predict(self.features_dense)
            
            scatter = ax.scatter(
                features_2d[:, 0],
                features_2d[:, 1],
                c=labels,
                cmap='tab20',
                alpha=0.6,
                s=15,
                edgecolors='none'
            )
            
            # Calculate metrics
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            valid_mask = labels != -1
            
            if n_clusters > 1 and valid_mask.sum() > 0:
                silhouette = silhouette_score(self.features_dense[valid_mask], labels[valid_mask])
                title = f'{name}\nClusters: {n_clusters}, Silhouette: {silhouette:.3f}'
            else:
                title = f'{name}\nClusters: {n_clusters}'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            
            if idx < len(configs):
                plt.colorbar(scatter, ax=ax, label='Cluster', fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/algorithm_comparison_grid.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/algorithm_comparison_grid.png")
        plt.close()
    
    def plot_tsne_visualization(self, perplexity=30):
        """Create t-SNE visualization"""
        logger.info(f"Creating t-SNE visualization (perplexity={perplexity})...")
        
        # Use a subset for faster t-SNE
        max_samples = min(5000, len(self.features_dense))
        indices = np.random.choice(len(self.features_dense), max_samples, replace=False)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        features_tsne = tsne.fit_transform(self.features_dense[indices])
        
        # Apply clustering on subset
        kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.features_dense[indices])
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        scatter = ax.scatter(
            features_tsne[:, 0],
            features_tsne[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=30,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_title(f't-SNE Visualization - K-Means (k=15)\n{max_samples} products', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/tsne_visualization.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/tsne_visualization.png")
        plt.close()
    
    def plot_variance_explained(self):
        """Plot explained variance by principal components"""
        logger.info("Creating variance explained plot...")
        
        pca = PCA(random_state=42)
        pca.fit(self.features_dense)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_,
               alpha=0.7,
               color='steelblue',
               edgecolor='black')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained Ratio', fontsize=12)
        ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 
                'o-', linewidth=2, markersize=6, color='orangered')
        ax2.axhline(y=0.90, color='green', linestyle='--', label='90% threshold', linewidth=2)
        ax2.axhline(y=0.95, color='blue', linestyle='--', label='95% threshold', linewidth=2)
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/variance_explained.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/variance_explained.png")
        plt.close()
    
    def plot_dbscan_sensitivity(self):
        """Analyze DBSCAN sensitivity to eps parameter"""
        logger.info("Creating DBSCAN sensitivity analysis...")
        
        eps_values = np.linspace(0.3, 2.0, 15)
        results = {
            'eps': [],
            'n_clusters': [],
            'n_noise': [],
            'silhouette': []
        }
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(self.features_dense)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            results['eps'].append(eps)
            results['n_clusters'].append(n_clusters)
            results['n_noise'].append(n_noise)
            
            if n_clusters > 1:
                valid_mask = labels != -1
                if valid_mask.sum() > 0:
                    silhouette = silhouette_score(self.features_dense[valid_mask], labels[valid_mask])
                else:
                    silhouette = None
            else:
                silhouette = None
            
            results['silhouette'].append(silhouette)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Number of clusters vs eps
        ax1.plot(results['eps'], results['n_clusters'], 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Epsilon (eps)', fontsize=12)
        ax1.set_ylabel('Number of Clusters', fontsize=12)
        ax1.set_title('DBSCAN: Number of Clusters vs Epsilon', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score vs eps
        valid_silhouettes = [s for s in results['silhouette'] if s is not None]
        valid_eps = [results['eps'][i] for i, s in enumerate(results['silhouette']) if s is not None]
        
        if valid_silhouettes:
            ax2.plot(valid_eps, valid_silhouettes, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Epsilon (eps)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('DBSCAN: Silhouette Score vs Epsilon', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/dbscan_sensitivity.png', dpi=300, bbox_inches='tight')
        logger.info(f"✅ Saved: {self.output_dir}/dbscan_sensitivity.png")
        plt.close()
    
    def plot_heatmap_comparison(self):
        """Create heatmap comparing all algorithms and configurations"""
        logger.info("Creating comparison heatmap...")
        
        # Load results if available
        results_file = 'results/clustering_results.csv'
        if os.path.exists(results_file):
            df_results = pd.read_csv(results_file)
            
            # Prepare data for heatmap
            pivot_data = df_results.pivot_table(
                index='algorithm',
                columns='n_clusters',
                values='silhouette_score',
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                       cbar_kws={'label': 'Silhouette Score'}, ax=ax,
                       linewidths=0.5, linecolor='gray')
            ax.set_title('Silhouette Score Heatmap - All Algorithms', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax.set_ylabel('Algorithm', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/heatmap_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved: {self.output_dir}/heatmap_comparison.png")
            plt.close()
    
    def plot_all(self):
        """Generate all plots"""
        logger.info("=" * 80)
        logger.info("GENERATING ALL CLUSTERING PLOTS")
        logger.info("=" * 80)
        
        # Load data
        self.load_and_prepare_data()
        
        # Generate all plots
        self.plot_elbow_curve()
        self.plot_metrics_comparison()
        self.plot_pca_visualization()
        self.plot_silhouette_analysis()
        self.plot_cluster_distribution()
        self.plot_variance_explained()
        self.plot_dbscan_sensitivity()
        self.plot_heatmap_comparison()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ ALL PLOTS GENERATED!")
        logger.info(f"📁 Saved in: {self.output_dir}/")
        logger.info("=" * 80)
        
        # List all generated files
        files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png')])
        logger.info("\nGenerated plots:")
        for f in files:
            logger.info(f"  • {f}")


def main():
    """Main function"""
    dataset_path = 'datasets/complete_dataset.json'
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return 1
    
    visualizer = ClusteringVisualizer(dataset_path, output_dir='plots')
    visualizer.plot_all()
    
    return 0


if __name__ == '__main__':
    exit(main())

