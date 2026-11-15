#!/usr/bin/env python3
"""
K=7 Clustering vs Category Evaluation (MiniLM Embeddings)
========================================================

This script:
  - Loads MiniLM-based product embeddings from `embedding_generator/embeddings/`.
  - Runs K-Means clustering with k=7.
  - Uses `category_id` as ground-truth labels and cluster labels (0..6) as predictions.
  - Builds a category × cluster confusion (coherence) matrix.
  - Computes per-category F1 scores (best cluster per category) and macro/micro F1.
  - Saves:
      - `cluster_k7_category_evaluation/f1_results_k7.json`
      - `cluster_k7_category_evaluation/confusion_matrix_k7.json`
      - `cluster_k7_category_evaluation/confusion_matrix_k7_heatmap.png`
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

EMB_DIR = "embedding_generator/embeddings"
EMB_FULL_PATH = os.path.join(EMB_DIR, "embeddings_full.json")
EMB_NPY_PATH = os.path.join(EMB_DIR, "embeddings.npy")

# Enhanced dataset with category labels
ENHANCED_PATH = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"

OUTPUT_DIR = "cluster_k7_category_evaluation"


def load_embeddings_and_products() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load embeddings (MiniLM) and associated product metadata from embeddings_full.json."""
    embeddings = np.load(EMB_NPY_PATH)
    with open(EMB_FULL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    products = data.get("products", data)
    if embeddings.shape[0] != len(products):
        raise ValueError(f"Embeddings count ({embeddings.shape[0]}) "
                         f"!= products count ({len(products)})")
    return embeddings, products


def load_category_map() -> Dict[int, Dict[str, Any]]:
    """
    Load enhanced products and build mapping:
      product_id -> {"category_id": int, "category_name": str}
    """
    with open(ENHANCED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    enhanced_products = data.get("products", data)

    id_to_cat: Dict[int, Dict[str, Any]] = {}
    for p in enhanced_products:
        pid = p.get("id")
        if not isinstance(pid, int):
            continue
        cid = p.get("category_id")
        cname = p.get("category_name", "")
        if cid is None:
            continue
        id_to_cat[pid] = {"category_id": int(cid), "category_name": cname}
    return id_to_cat


def run_kmeans_k7(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Run K-Means clustering with k=7 and return cluster labels."""
    kmeans = KMeans(n_clusters=7, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels


def build_confusion(
    products: List[Dict[str, Any]],
    labels: np.ndarray,
    id_to_category: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, str], List[int], List[int]]:
    """
    Build confusion matrix counts:
      counts[category_id][cluster_id] = number of products

    Returns:
      counts, category_id_to_name, sorted_category_ids, sorted_cluster_ids
    """
    counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    category_id_to_name: Dict[int, str] = {}
    cluster_ids_set = set()

    if len(products) != len(labels):
        raise ValueError("Products and labels length mismatch")

    for p, clus in zip(products, labels):
        pid = p.get("id")
        if not isinstance(pid, int):
            continue

        cat_info = id_to_category.get(pid)
        if not cat_info:
            continue

        cid_int = cat_info["category_id"]
        cname = cat_info.get("category_name", "")
        clus_int = int(clus)

        category_id_to_name[cid_int] = cname
        counts[cid_int][clus_int] += 1
        cluster_ids_set.add(clus_int)

    sorted_category_ids = sorted(category_id_to_name.keys())
    sorted_cluster_ids = sorted(cluster_ids_set)
    return counts, category_id_to_name, sorted_category_ids, sorted_cluster_ids


def compute_f1_scores(
    counts: Dict[int, Dict[int, int]],
    sorted_category_ids: List[int],
    sorted_cluster_ids: List[int],
) -> Dict[str, Any]:
    """
    Compute per-category F1 scores, best cluster per category, and aggregate metrics.

    For each category c and cluster k:
      tp = n(c,k)
      fp = n(*,k) - tp
      fn = n(c,*) - tp
    """
    total_per_category: Dict[int, int] = {}
    total_per_cluster: Dict[int, int] = {k: 0 for k in sorted_cluster_ids}
    total_all = 0

    for c in sorted_category_ids:
        total_c = sum(counts[c].get(k, 0) for k in sorted_cluster_ids)
        total_per_category[c] = total_c
        total_all += total_c
        for k in sorted_cluster_ids:
            total_per_cluster[k] += counts[c].get(k, 0)

    per_category_results: Dict[int, Dict[str, Any]] = {}
    macro_f1_sum = 0.0
    macro_f1_count = 0
    weighted_f1_sum = 0.0

    for c in sorted_category_ids:
        best_f1 = 0.0
        best_cluster = None
        details: List[Dict[str, Any]] = []

        for k in sorted_cluster_ids:
            tp = counts[c].get(k, 0)
            if tp == 0:
                continue
            fp = total_per_cluster[k] - tp
            fn = total_per_category[c] - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision == 0.0 and recall == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            details.append(
                {
                    "cluster_id": k,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            if f1 > best_f1:
                best_f1 = f1
                best_cluster = k

        per_category_results[c] = {
            "total": total_per_category[c],
            "best_cluster_id": best_cluster,
            "best_f1": best_f1,
            "details": sorted(details, key=lambda d: d["f1"], reverse=True),
        }

        macro_f1_sum += best_f1
        macro_f1_count += 1
        weighted_f1_sum += best_f1 * total_per_category[c]

    macro_f1 = macro_f1_sum / macro_f1_count if macro_f1_count > 0 else 0.0
    micro_f1 = weighted_f1_sum / total_all if total_all > 0 else 0.0

    return {
        "per_category": per_category_results,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "total_products": total_all,
    }


def save_confusion_matrix(
    counts: Dict[int, Dict[int, int]],
    sorted_category_ids: List[int],
    sorted_cluster_ids: List[int],
    category_id_to_name: Dict[int, str],
) -> None:
    """Save confusion matrix as JSON and heatmap (normalized by category)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    matrix = np.zeros((len(sorted_category_ids), len(sorted_cluster_ids)), dtype=int)
    for i, c in enumerate(sorted_category_ids):
        for j, k in enumerate(sorted_cluster_ids):
            matrix[i, j] = counts[c].get(k, 0)

    # JSON export
    matrix_json = {
        "categories": [
            {"id": c, "name": category_id_to_name.get(c, "")}
            for c in sorted_category_ids
        ],
        "clusters": sorted_cluster_ids,
        "counts": matrix.tolist(),
    }
    with open(
        os.path.join(OUTPUT_DIR, "confusion_matrix_k7.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(matrix_json, f, ensure_ascii=False, indent=2)

    # Row-normalized heatmap
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_matrix = np.divide(matrix, row_sums, where=row_sums != 0)

    plt.figure(figsize=(10, 6))
    plt.imshow(norm_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="نسبت محصولات در خوشه (نرمال‌شده بر اساس دسته)")
    plt.xlabel("شناسه خوشه (0 تا 6)")
    plt.ylabel("دسته محصول (category_id)")

    y_ticks = range(len(sorted_category_ids))
    y_labels = [str(c) for c in sorted_category_ids]
    plt.yticks(y_ticks, y_labels)

    plt.xticks(
        ticks=range(len(sorted_cluster_ids)),
        labels=[str(k) for k in sorted_cluster_ids],
        rotation=0,
    )

    plt.title("ماتریس هم‌خوانی دسته / خوشه برای K=7 (MiniLM Embeddings)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "confusion_matrix_k7_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    print("K=7 clustering vs category evaluation (MiniLM embeddings)")

    if not os.path.exists(EMB_NPY_PATH) or not os.path.exists(EMB_FULL_PATH):
        raise FileNotFoundError("Embeddings files not found in embedding_generator/embeddings/")
    if not os.path.exists(ENHANCED_PATH):
        raise FileNotFoundError(f"Enhanced dataset not found at {ENHANCED_PATH}")

    embeddings, products = load_embeddings_and_products()
    print(f"Loaded embeddings: {embeddings.shape}, products from embeddings_full: {len(products)}")

    id_to_category = load_category_map()
    print(f"Loaded category map for {len(id_to_category)} products from enhanced dataset.")

    labels = run_kmeans_k7(embeddings)
    print("K-Means with k=7 finished.")

    counts, category_id_to_name, sorted_category_ids, sorted_cluster_ids = build_confusion(
        products, labels, id_to_category
    )
    print(f"Found {len(sorted_category_ids)} categories and {len(sorted_cluster_ids)} clusters (0..6).")

    f1_results = compute_f1_scores(counts, sorted_category_ids, sorted_cluster_ids)
    macro_f1 = f1_results["macro_f1"]
    micro_f1 = f1_results["micro_f1"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(
        os.path.join(OUTPUT_DIR, "f1_results_k7.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(f1_results, f, ensure_ascii=False, indent=2)

    save_confusion_matrix(counts, sorted_category_ids, sorted_cluster_ids, category_id_to_name)

    print("\nSummary for K=7:")
    print(f"  Total products evaluated: {f1_results['total_products']}")
    print(f"  Macro F1 (average over categories): {macro_f1:.4f}")
    print(f"  Micro F1 (weighted by category size): {micro_f1:.4f}")
    print(f"  Detailed per-category F1 saved to: {os.path.join(OUTPUT_DIR, 'f1_results_k7.json')}")
    print(f"  Confusion matrix JSON + heatmap saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()


