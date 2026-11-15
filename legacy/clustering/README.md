# Product Clustering Analysis

Cluster products based on title and description using multiple machine learning algorithms.

## Features

### Clustering Algorithms
- **K-Means**: Partitional clustering with different k values
- **Hierarchical Clustering**: Agglomerative clustering with Ward linkage
- **DBSCAN**: Density-based clustering for arbitrary shapes
- **Gaussian Mixture Model (GMM)**: Probabilistic clustering

### Evaluation Metrics
- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances (lower is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster variance (higher is better)
- **Inertia**: Within-cluster sum of squares (for K-Means)
- **BIC/AIC**: Information criteria (for GMM)

## Installation

```bash
cd clustering_analysis
pip install -r requirements.txt
```

## Usage

```bash
python cluster_products.py
```

## Output

The analysis creates the following files in the `results/` directory:

- `clustering_results.json` - Detailed results for all algorithms
- `clustering_results.csv` - Summary table of all results
- `clustering_visualization.png` - Visual comparison of clustering methods

## Results Interpretation

### Silhouette Score
- **1.0**: Perfect clustering
- **0.5-1.0**: Strong clustering
- **0.25-0.5**: Weak clustering
- **< 0.25**: No substantial structure

### Davies-Bouldin Index
- **Lower is better**
- Values close to 0 indicate better clustering

### Calinski-Harabasz Score
- **Higher is better**
- Larger values indicate denser, well-separated clusters

