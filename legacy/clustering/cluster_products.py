#!/usr/bin/env python3
"""
Product Clustering Analysis
Cluster products based on title and description using multiple algorithms
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

# Evaluation metrics
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)

# Persian text processing
try:
    from hazm import Normalizer, word_tokenize, stopwords_list
    HAZM_AVAILABLE = True
except:
    HAZM_AVAILABLE = False
    print("Warning: hazm not available. Using basic text processing.")

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductClusteringAnalysis:
    """Comprehensive clustering analysis for products"""
    
    def __init__(self, dataset_path, output_dir='results'):
        """Initialize clustering analysis"""
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Persian text normalizer
        if HAZM_AVAILABLE:
            self.normalizer = Normalizer()
            self.stop_words = set(stopwords_list())
        else:
            self.normalizer = None
            self.stop_words = set()
        
        self.df = None
        self.features = None
        self.feature_names = None
        self.results = {}
    
    def load_data(self):
        """Load product dataset"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract products
        if 'products' in data:
            products = data['products']
        else:
            products = data
        
        self.df = pd.DataFrame(products)
        logger.info(f"Loaded {len(self.df)} products")
        
        # Filter products with title or description
        self.df = self.df[
            (self.df['title'].notna()) | (self.df['description'].notna())
        ].copy()
        
        logger.info(f"After filtering: {len(self.df)} products with text data")
        
        return self.df
    
    def preprocess_text(self, text):
        """Preprocess Persian/English text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Normalize Persian text
        if self.normalizer and HAZM_AVAILABLE:
            text = self.normalizer.normalize(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def prepare_features(self, method='tfidf', max_features=1000):
        """Extract features from text"""
        logger.info("Preparing text features...")
        
        # Combine title and description
        self.df['combined_text'] = (
            self.df['title'].fillna('') + ' ' + 
            self.df['description'].fillna('')
        ).apply(self.preprocess_text)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        self.features = vectorizer.fit_transform(self.df['combined_text'])
        self.feature_names = vectorizer.get_feature_names_out()
        
        logger.info(f"Feature matrix shape: {self.features.shape}")
        
        return self.features
    
    def reduce_dimensions(self, n_components=50):
        """Reduce dimensionality for better clustering"""
        logger.info(f"Reducing dimensions to {n_components}")
        
        svd = TruncatedSVD(n_components=min(n_components, self.features.shape[1] - 1))
        features_reduced = svd.fit_transform(self.features)
        
        logger.info(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
        
        return features_reduced
    
    def apply_kmeans(self, features, n_clusters_range=[3, 5, 7, 10, 15, 20]):
        """Apply K-Means clustering with different k values"""
        logger.info("Applying K-Means clustering...")
        
        results = []
        
        for n_clusters in n_clusters_range:
            logger.info(f"  K-Means with k={n_clusters}")
            
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            silhouette = silhouette_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            
            results.append({
                'algorithm': 'K-Means',
                'n_clusters': n_clusters,
                'labels': labels,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'inertia': kmeans.inertia_
            })
            
            logger.info(f"    Silhouette: {silhouette:.4f}")
            logger.info(f"    Davies-Bouldin: {davies_bouldin:.4f}")
            logger.info(f"    Calinski-Harabasz: {calinski_harabasz:.2f}")
        
        return results
    
    def apply_dbscan(self, features, eps_range=[0.3, 0.5, 0.7, 1.0], min_samples=5):
        """Apply DBSCAN clustering"""
        logger.info("Applying DBSCAN clustering...")
        
        results = []
        
        for eps in eps_range:
            logger.info(f"  DBSCAN with eps={eps}")
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = dbscan.fit_predict(features)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters > 1:
                # Only calculate if we have valid clusters
                valid_indices = labels != -1
                if valid_indices.sum() > 0:
                    silhouette = silhouette_score(features[valid_indices], labels[valid_indices])
                    davies_bouldin = davies_bouldin_score(features[valid_indices], labels[valid_indices])
                    calinski_harabasz = calinski_harabasz_score(features[valid_indices], labels[valid_indices])
                else:
                    silhouette = davies_bouldin = calinski_harabasz = None
            else:
                silhouette = davies_bouldin = calinski_harabasz = None
            
            results.append({
                'algorithm': 'DBSCAN',
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'labels': labels,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz
            })
            
            logger.info(f"    Clusters: {n_clusters}, Noise points: {n_noise}")
            if silhouette is not None:
                logger.info(f"    Silhouette: {silhouette:.4f}")
        
        return results
    
    def apply_hierarchical(self, features, n_clusters_range=[5, 10, 15, 20]):
        """Apply Hierarchical clustering"""
        logger.info("Applying Hierarchical clustering...")
        
        results = []
        
        for n_clusters in n_clusters_range:
            logger.info(f"  Hierarchical with k={n_clusters}")
            
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
            
            labels = hierarchical.fit_predict(features)
            
            silhouette = silhouette_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            
            results.append({
                'algorithm': 'Hierarchical',
                'n_clusters': n_clusters,
                'labels': labels,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz
            })
            
            logger.info(f"    Silhouette: {silhouette:.4f}")
        
        return results
    
    def apply_gmm(self, features, n_components_range=[5, 10, 15, 20]):
        """Apply Gaussian Mixture Model"""
        logger.info("Applying Gaussian Mixture Model...")
        
        results = []
        
        for n_components in n_components_range:
            logger.info(f"  GMM with k={n_components}")
            
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=42,
                covariance_type='tied'
            )
            
            labels = gmm.fit_predict(features)
            
            silhouette = silhouette_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            
            results.append({
                'algorithm': 'GMM',
                'n_clusters': n_components,
                'labels': labels,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz,
                'bic': gmm.bic(features),
                'aic': gmm.aic(features)
            })
            
            logger.info(f"    Silhouette: {silhouette:.4f}")
            logger.info(f"    BIC: {gmm.bic(features):.2f}")
        
        return results
    
    def visualize_results(self, features, results):
        """Create visualizations for clustering results"""
        logger.info("Creating visualizations...")
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features.toarray() if hasattr(features, 'toarray') else features)
        
        # Create subplots for different algorithms
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for result in results[:6]:  # Plot first 6 results
            ax = axes[plot_idx]
            
            labels = result['labels']
            scatter = ax.scatter(
                features_2d[:, 0],
                features_2d[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=10
            )
            
            title = f"{result['algorithm']}"
            if 'n_clusters' in result:
                title += f" (k={result['n_clusters']})"
            if 'eps' in result:
                title += f" (eps={result['eps']})"
            
            title += f"\nSilhouette: {result['silhouette_score']:.3f}" if result['silhouette_score'] else ""
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/clustering_visualization.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {self.output_dir}/clustering_visualization.png")
        plt.close()
    
    def save_results(self, all_results):
        """Save clustering results to file"""
        logger.info("Saving results...")
        
        # Prepare summary
        summary = []
        for result in all_results:
            summary_item = {
                'algorithm': result['algorithm'],
                'n_clusters': result.get('n_clusters', result.get('eps')),
                'silhouette_score': result.get('silhouette_score'),
                'davies_bouldin_score': result.get('davies_bouldin_score'),
                'calinski_harabasz_score': result.get('calinski_harabasz_score')
            }
            
            if 'inertia' in result:
                summary_item['inertia'] = result['inertia']
            if 'bic' in result:
                summary_item['bic'] = result['bic']
            if 'aic' in result:
                summary_item['aic'] = result['aic']
            if 'n_noise' in result:
                summary_item['n_noise'] = result['n_noise']
            
            summary.append(summary_item)
        
        # Save as JSON
        with open(f'{self.output_dir}/clustering_results.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Save as DataFrame
        df_results = pd.DataFrame(summary)
        df_results.to_csv(f'{self.output_dir}/clustering_results.csv', index=False)
        
        logger.info(f"Results saved to {self.output_dir}/")
        
        return df_results
    
    def find_best_clustering(self, all_results):
        """Find the best clustering configuration"""
        logger.info("Finding best clustering configuration...")
        
        valid_results = [r for r in all_results if r.get('silhouette_score') is not None]
        
        if not valid_results:
            logger.warning("No valid results to compare")
            return None
        
        # Sort by silhouette score (higher is better)
        best_by_silhouette = max(valid_results, key=lambda x: x['silhouette_score'])
        
        # Sort by Davies-Bouldin (lower is better)
        best_by_db = min(valid_results, key=lambda x: x['davies_bouldin_score'])
        
        # Sort by Calinski-Harabasz (higher is better)
        best_by_ch = max(valid_results, key=lambda x: x['calinski_harabasz_score'])
        
        logger.info("\n" + "=" * 80)
        logger.info("BEST CLUSTERING CONFIGURATIONS")
        logger.info("=" * 80)
        logger.info(f"Best by Silhouette Score ({best_by_silhouette['silhouette_score']:.4f}):")
        logger.info(f"  Algorithm: {best_by_silhouette['algorithm']}")
        logger.info(f"  Clusters: {best_by_silhouette.get('n_clusters', 'N/A')}")
        logger.info("")
        logger.info(f"Best by Davies-Bouldin Score ({best_by_db['davies_bouldin_score']:.4f}):")
        logger.info(f"  Algorithm: {best_by_db['algorithm']}")
        logger.info(f"  Clusters: {best_by_db.get('n_clusters', 'N/A')}")
        logger.info("")
        logger.info(f"Best by Calinski-Harabasz Score ({best_by_ch['calinski_harabasz_score']:.2f}):")
        logger.info(f"  Algorithm: {best_by_ch['algorithm']}")
        logger.info(f"  Clusters: {best_by_ch.get('n_clusters', 'N/A')}")
        logger.info("=" * 80)
        
        return {
            'best_silhouette': best_by_silhouette,
            'best_davies_bouldin': best_by_db,
            'best_calinski_harabasz': best_by_ch
        }
    
    def run_full_analysis(self):
        """Run complete clustering analysis"""
        logger.info("=" * 80)
        logger.info("PRODUCT CLUSTERING ANALYSIS")
        logger.info("=" * 80)
        
        # Load data
        self.load_data()
        
        # Prepare features
        features_sparse = self.prepare_features(max_features=1000)
        
        # Reduce dimensions
        features_dense = self.reduce_dimensions(n_components=50)
        
        # Apply different clustering algorithms
        all_results = []
        
        # K-Means
        all_results.extend(self.apply_kmeans(features_dense))
        
        # Hierarchical
        all_results.extend(self.apply_hierarchical(features_dense))
        
        # GMM
        all_results.extend(self.apply_gmm(features_dense))
        
        # DBSCAN (on original features)
        all_results.extend(self.apply_dbscan(features_dense, eps_range=[0.5, 1.0, 1.5]))
        
        # Visualize results
        self.visualize_results(features_dense, all_results)
        
        # Save results
        df_results = self.save_results(all_results)
        
        # Find best configuration
        best_configs = self.find_best_clustering(all_results)
        
        logger.info("\n✅ Analysis complete!")
        logger.info(f"📁 Results saved in: {self.output_dir}/")
        
        return all_results, best_configs


def main():
    """Main function"""
    import sys
    
    # Check if dataset path provided
    dataset_path = 'datasets/complete_dataset.json'
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please provide the correct path to your dataset")
        return 1
    
    # Run analysis
    analyzer = ProductClusteringAnalysis(
        dataset_path=dataset_path,
        output_dir='results'
    )
    
    results, best_configs = analyzer.run_full_analysis()
    
    return 0


if __name__ == '__main__':
    exit(main())

