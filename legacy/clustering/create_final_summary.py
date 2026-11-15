#!/usr/bin/env python3
"""
Create final summary visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load results
df = pd.read_csv('results/clustering_results.csv')

# Create comprehensive summary visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Product Clustering Analysis - Model Comparison Summary', 
             fontsize=18, fontweight='bold', y=0.98)

# 1. Silhouette Score Comparison
ax1 = fig.add_subplot(gs[0, 0])
top_5_sil = df.nlargest(5, 'silhouette_score')
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_5_sil))]
bars = ax1.barh(range(len(top_5_sil)), top_5_sil['silhouette_score'], color=colors)
ax1.set_yticks(range(len(top_5_sil)))
labels = [f"{row['algorithm']} (k={int(row['n_clusters'])})" for _, row in top_5_sil.iterrows()]
ax1.set_yticklabels(labels)
ax1.set_xlabel('Silhouette Score', fontweight='bold')
ax1.set_title('🏆 Top 5 Models - Silhouette Score (Higher is Better)', fontweight='bold')
ax1.invert_yaxis()
for i, (_, row) in enumerate(top_5_sil.iterrows()):
    ax1.text(row['silhouette_score'] + 0.01, i, f"{row['silhouette_score']:.4f}", 
            va='center', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. Davies-Bouldin Comparison
ax2 = fig.add_subplot(gs[0, 1])
top_5_db = df.nsmallest(5, 'davies_bouldin_score')
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_5_db))]
bars = ax2.barh(range(len(top_5_db)), top_5_db['davies_bouldin_score'], color=colors)
ax2.set_yticks(range(len(top_5_db)))
labels = [f"{row['algorithm']} (k={int(row['n_clusters'])})" for _, row in top_5_db.iterrows()]
ax2.set_yticklabels(labels)
ax2.set_xlabel('Davies-Bouldin Index', fontweight='bold')
ax2.set_title('🏆 Top 5 Models - Davies-Bouldin (Lower is Better)', fontweight='bold')
ax2.invert_yaxis()
for i, (_, row) in enumerate(top_5_db.iterrows()):
    ax2.text(row['davies_bouldin_score'] + 0.05, i, f"{row['davies_bouldin_score']:.4f}", 
            va='center', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Algorithm Metrics Scatter
ax3 = fig.add_subplot(gs[1, :])
for algo in df['algorithm'].unique():
    if algo != 'DBSCAN':
        data = df[df['algorithm'] == algo]
        ax3.scatter(data['silhouette_score'], data['davies_bouldin_score'], 
                   s=data['n_clusters']*20, alpha=0.6, label=algo)

# Highlight best model
best = df.nlargest(1, 'silhouette_score').iloc[0]
ax3.scatter([best['silhouette_score']], [best['davies_bouldin_score']], 
           s=500, color='red', marker='*', edgecolors='black', linewidth=2,
           label=f"WINNER: {best['algorithm']} k={int(best['n_clusters'])}", zorder=10)

ax3.set_xlabel('Silhouette Score (Higher is Better) →', fontsize=12, fontweight='bold')
ax3.set_ylabel('Davies-Bouldin Index (Lower is Better) ←', fontsize=12, fontweight='bold')
ax3.set_title('Model Performance Map\n(Best models = Top-Left corner)', 
             fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Model Performance Summary Table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('tight')
ax4.axis('off')

summary_data = [
    ['Rank', 'Algorithm', 'k', 'Silhouette ↑', 'Davies-Bouldin ↓', 'Verdict'],
    ['🥇', 'K-Means', '20', '0.3030', '1.5971', '✅ BEST - Fast & Accurate'],
    ['🥈', 'Hierarchical', '20', '0.2937', '1.4646', '✅ For Taxonomy'],
    ['🥉', 'GMM', '20', '0.2934', '1.5252', '✅ For Probabilities'],
    ['4', 'K-Means', '15', '0.2702', '1.5830', '✅ Good Alternative'],
    ['5', 'Hierarchical', '15', '0.2784', '1.6150', '✅ Balanced'],
]

table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.08, 0.18, 0.08, 0.18, 0.22, 0.26])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Style winner row
for i in range(6):
    cell = table[(1, i)]
    cell.set_facecolor('#2ecc71')
    cell.set_text_props(weight='bold')

ax4.set_title('Model Performance Summary Table', fontsize=14, fontweight='bold', pad=20)

plt.savefig('plots/final_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
print('✅ Created final summary visualization: plots/final_summary.png')
plt.close()

print()
print('=' * 80)
print('📊 FINAL CONCLUSION')
print('=' * 80)
print()
print('🏆 BEST MODEL: K-Means with k=20 clusters')
print()
print('Key Metrics:')
print('  ✅ Silhouette Score: 0.3030 (BEST)')
print('  ✅ Davies-Bouldin: 1.5971 (Good)')
print('  ✅ Calinski-Harabasz: 1133.24 (Strong)')
print()
print('Why K-Means k=20?')
print('  1. Highest silhouette score = Best cluster quality')
print('  2. Fast computation (~3 seconds)')
print('  3. Reproducible and deterministic')
print('  4. Easy to implement and maintain')
print('  5. Provides detailed product segmentation (20 groups)')
print()
print('Alternative Options:')
print('  - K-Means k=15: Good balance (Silhouette: 0.2702)')
print('  - Hierarchical k=20: Best for taxonomy (DB: 1.4646)')
print('  - GMM k=20: For probability assignments (Silhouette: 0.2934)')
print()
print('Not Recommended:')
print('  - DBSCAN: Not suitable for this dataset (only 3 clusters)')
print('  - K < 10: Too broad, loses product detail')
print()
print('=' * 80)
