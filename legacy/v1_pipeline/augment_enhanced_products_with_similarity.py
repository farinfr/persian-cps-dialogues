#!/usr/bin/env python3
"""
Augment Enhanced Product Dataset with Similarity Insights
========================================================

This script enriches `enhanced_products_with_cluster_comparison.json` with:

* Top-10 similar products per product (based on cluster/category peers)
* Aggregated similarity statistics (price/rating/rating_count)
* Comparison labels such as "high_rated", "good_price", etc., including price range context

The additional metadata helps downstream dialogue generators surface comparative
insights (e.g., value-for-money, premium positioning) when referencing products.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

DATASET_PATH = "cluster_analysis_results/enhanced_products_with_cluster_comparison.json"
BACKUP_PATH = "cluster_analysis_results/enhanced_products_with_cluster_comparison_backup.json"
SIMILAR_PRODUCTS_TARGET = 10
MAX_CATEGORY_CANDIDATES = 60


def coerce_float(value: Any) -> Optional[float]:
    """Convert a value to float when possible, else return None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Compute the mean while ignoring None values."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def extract_spec_keys(product: Dict[str, Any]) -> Set[str]:
    """Collect specification keys (section + field names) for Jaccard similarity."""
    keys: Set[str] = set()
    specs = product.get("specifications")
    if not isinstance(specs, dict):
        return keys

    for section, content in specs.items():
        section_key = str(section).strip()
        if section_key:
            keys.add(section_key)

        if isinstance(content, dict):
            for field in content.keys():
                field_key = str(field).strip()
                if field_key:
                    keys.add(field_key)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    for field in item.keys():
                        field_key = str(field).strip()
                        if field_key:
                            keys.add(field_key)
    return keys


def compute_feature_ranges(products: Iterable[Dict[str, Any]]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Compute min/max for key numeric fields across a collection."""
    fields = ("price", "rating_average", "rating_count", "discount_percent", "selling_price")
    ranges: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    items = list(products)
    for field in fields:
        values = [coerce_float(prod.get(field)) for prod in items]
        numeric_values = [v for v in values if v is not None]
        if numeric_values:
            ranges[field] = (min(numeric_values), max(numeric_values))
        else:
            ranges[field] = (None, None)
    return ranges


def normalized_similarity(a: Optional[float], b: Optional[float], value_range: Tuple[Optional[float], Optional[float]]) -> Optional[float]:
    """Return a normalized similarity score (0..1) given two values and their range."""
    if a is None or b is None:
        return None
    minimum, maximum = value_range
    if minimum is None or maximum is None:
        return None
    if math.isclose(maximum, minimum):
        return 1.0
    diff = abs(a - b) / (maximum - minimum)
    return max(0.0, 1.0 - diff)


def jaccard_similarity(keys_a: Set[str], keys_b: Set[str]) -> Optional[float]:
    """Compute Jaccard similarity for two key sets."""
    if not keys_a or not keys_b:
        return None
    intersection = len(keys_a & keys_b)
    if intersection == 0:
        return 0.0
    union = len(keys_a | keys_b)
    return intersection / union


def calculate_similarity(
    product: Dict[str, Any],
    candidate: Dict[str, Any],
    feature_ranges: Dict[str, Tuple[Optional[float], Optional[float]]],
    spec_keys_map: Dict[int, Set[str]],
) -> float:
    """Aggregate similarity score combining numeric features and specification overlap."""
    weights = {
        "price": 0.4,
        "rating_average": 0.25,
        "rating_count": 0.15,
        "discount_percent": 0.05,
        "selling_price": 0.05,
    }

    total = 0.0
    total_weight = 0.0

    for field, weight in weights.items():
        similarity = normalized_similarity(
            coerce_float(product.get(field)),
            coerce_float(candidate.get(field)),
            feature_ranges.get(field, (None, None)),
        )
        if similarity is not None:
            total += similarity * weight
            total_weight += weight

    # Specification overlap
    product_keys = spec_keys_map.get(product["id"], set())
    candidate_keys = spec_keys_map.get(candidate["id"], set())
    spec_similarity = jaccard_similarity(product_keys, candidate_keys)
    if spec_similarity is not None:
        weight = 0.1
        total += spec_similarity * weight
        total_weight += weight

    if total_weight == 0:
        base_score = 0.0
    else:
        base_score = total / total_weight

    # Brand and category bonus
    if product.get("brand") and product.get("brand") == candidate.get("brand"):
        base_score += 0.05
    if product.get("category_id") == candidate.get("category_id"):
        base_score += 0.02

    return max(0.0, min(base_score, 1.0))


def select_candidates(
    product: Dict[str, Any],
    cluster_map: Dict[int, List[Dict[str, Any]]],
    category_map: Dict[int, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Gather candidate products from the same cluster (and category fallback)."""
    product_id = product["id"]
    cluster_id = product.get("cluster_id")
    category_id = product.get("category_id")

    candidates: List[Dict[str, Any]] = []
    seen_ids = {product_id}

    if cluster_id in cluster_map:
        for item in cluster_map[cluster_id]:
            if item["id"] != product_id:
                candidates.append(item)
                seen_ids.add(item["id"])

    if len(candidates) >= SIMILAR_PRODUCTS_TARGET:
        return candidates

    if category_id in category_map:
        for item in category_map[category_id]:
            if item["id"] in seen_ids:
                continue
            candidates.append(item)
            seen_ids.add(item["id"])
            if len(candidates) >= SIMILAR_PRODUCTS_TARGET + MAX_CATEGORY_CANDIDATES:
                break

    return candidates


def score_candidates(
    product: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    spec_keys_map: Dict[int, Set[str]],
) -> List[Dict[str, Any]]:
    """Compute similarity scores and auxiliary deltas for each candidate."""
    if not candidates:
        return []

    feature_ranges = compute_feature_ranges([product] + candidates)
    product_price = coerce_float(product.get("price"))
    product_rating = coerce_float(product.get("rating_average"))
    product_rating_count = coerce_float(product.get("rating_count"))

    scored: List[Dict[str, Any]] = []
    for candidate in candidates:
        score = calculate_similarity(product, candidate, feature_ranges, spec_keys_map)
        candidate_price = coerce_float(candidate.get("price"))
        candidate_rating = coerce_float(candidate.get("rating_average"))
        candidate_rating_count = coerce_float(candidate.get("rating_count"))

        price_diff_percent: Optional[float] = None
        if product_price not in (None, 0.0) and candidate_price is not None:
            price_diff_percent = ((candidate_price - product_price) / product_price) * 100.0

        price_relation = "similar"
        if price_diff_percent is not None:
            if price_diff_percent >= 8:
                price_relation = "higher"
            elif price_diff_percent <= -8:
                price_relation = "lower"

        rating_average_diff: Optional[float] = None
        if product_rating is not None and candidate_rating is not None:
            rating_average_diff = candidate_rating - product_rating

        rating_count_diff: Optional[float] = None
        if product_rating_count is not None and candidate_rating_count is not None:
            rating_count_diff = candidate_rating_count - product_rating_count

        scored.append(
            {
                "candidate": candidate,
                "similarity_score": score,
                "price": candidate_price,
                "price_diff_percent": price_diff_percent,
                "price_relation": price_relation,
                "rating_average": candidate_rating,
                "rating_average_diff": rating_average_diff,
                "rating_count": candidate_rating_count,
                "rating_count_diff": rating_count_diff,
            }
        )

    scored.sort(key=lambda item: item["similarity_score"], reverse=True)
    return scored[:SIMILAR_PRODUCTS_TARGET]


def compute_price_percentile(product_price: Optional[float], peer_prices: List[float]) -> Optional[float]:
    """Estimate percentile rank of product price among peers (0..1)."""
    if product_price is None or not peer_prices:
        return None
    combined = sorted(peer_prices + [product_price])
    lower = sum(1 for price in combined if price < product_price)
    equal = sum(1 for price in combined if math.isclose(price, product_price))
    percentile = (lower + 0.5 * equal) / len(combined)
    return min(max(percentile, 0.0), 1.0)


def determine_price_bucket(percentile: Optional[float]) -> str:
    """Categorize product pricing position."""
    if percentile is None:
        return "unknown"
    if percentile <= 0.33:
        return "budget_range"
    if percentile >= 0.67:
        return "premium_range"
    return "mid_range"


def build_summary(
    product: Dict[str, Any],
    top_candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize numeric comparisons used for labelling."""
    product_price = coerce_float(product.get("price"))
    product_rating = coerce_float(product.get("rating_average"))
    product_rating_count = coerce_float(product.get("rating_count"))

    similarity_scores = [entry["similarity_score"] for entry in top_candidates]
    price_values = [entry["price"] for entry in top_candidates if entry["price"] is not None]
    rating_values = [entry["rating_average"] for entry in top_candidates if entry["rating_average"] is not None]
    rating_count_values = [entry["rating_count"] for entry in top_candidates if entry["rating_count"] is not None]

    price_percentile = compute_price_percentile(product_price, price_values)
    price_bucket = determine_price_bucket(price_percentile)
    price_avg = safe_mean(price_values)
    price_diff_percent_vs_avg: Optional[float] = None
    if price_avg not in (None, 0.0) and product_price is not None:
        price_diff_percent_vs_avg = ((product_price - price_avg) / price_avg) * 100.0

    return {
        "price": {
            "product_price": product_price,
            "similar_min": min(price_values) if price_values else None,
            "similar_max": max(price_values) if price_values else None,
            "similar_avg": price_avg,
            "price_percentile": price_percentile,
            "price_position_label": price_bucket,
            "price_diff_percent_vs_similar_avg": price_diff_percent_vs_avg,
            "sample_size": len(price_values),
        },
        "rating": {
            "product_rating": product_rating,
            "similar_min": min(rating_values) if rating_values else None,
            "similar_max": max(rating_values) if rating_values else None,
            "similar_avg": safe_mean(rating_values),
            "sample_size": len(rating_values),
        },
        "rating_count": {
            "product_rating_count": product_rating_count,
            "similar_min": min(rating_count_values) if rating_count_values else None,
            "similar_max": max(rating_count_values) if rating_count_values else None,
            "similar_avg": safe_mean(rating_count_values),
            "sample_size": len(rating_count_values),
        },
        "average_similarity_score": safe_mean(similarity_scores),
        "similar_products_considered": len(top_candidates),
    }


def make_label(
    label: str,
    sentiment: str,
    reason: str,
    price_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a label dictionary with price context."""
    return {
        "label": label,
        "sentiment": sentiment,
        "reason": reason,
        "price_position": price_summary.get("price_position_label"),
        "price_range": {
            "product_price": price_summary.get("product_price"),
            "similar_min": price_summary.get("similar_min"),
            "similar_max": price_summary.get("similar_max"),
            "similar_avg": price_summary.get("similar_avg"),
            "price_percentile": price_summary.get("price_percentile"),
            "relative_price_to_average_percent": price_summary.get("price_diff_percent_vs_similar_avg"),
            "sample_size": price_summary.get("sample_size"),
        },
    }


def generate_labels(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Derive comparison-based labels for a product."""
    labels: List[Dict[str, Any]] = []
    price_summary = summary["price"]
    rating_summary = summary["rating"]
    rating_count_summary = summary["rating_count"]

    product_rating = rating_summary.get("product_rating")
    rating_avg = rating_summary.get("similar_avg")
    product_price = price_summary.get("product_price")
    price_percent = price_summary.get("price_diff_percent_vs_similar_avg")
    product_rating_count = rating_count_summary.get("product_rating_count")
    rating_count_avg = rating_count_summary.get("similar_avg")

    if product_rating is not None and rating_avg is not None:
        if product_rating >= rating_avg + 5 or product_rating >= 85:
            reason = f"Rating {product_rating:.1f} vs similar avg {rating_avg:.1f}"
            labels.append(make_label("high_rated", "positive", reason, price_summary))
        elif product_rating <= rating_avg - 10 and rating_summary.get("sample_size", 0) >= 3:
            reason = f"Rating {product_rating:.1f} trails similar avg {rating_avg:.1f}"
            labels.append(make_label("needs_rating_improvement", "attention", reason, price_summary))

    if product_rating_count is not None and rating_count_avg is not None:
        if rating_count_avg > 0 and product_rating_count >= rating_count_avg * 1.35:
            reason = f"Rating volume {product_rating_count:.0f} exceeds peer avg {rating_count_avg:.0f}"
            labels.append(make_label("popular_choice", "positive", reason, price_summary))
        elif rating_count_avg > 0 and product_rating_count <= rating_count_avg * 0.5:
            reason = f"Rating volume {product_rating_count:.0f} below peer avg {rating_count_avg:.0f}"
            labels.append(make_label("niche_pick", "neutral", reason, price_summary))

    if product_price is not None and price_percent is not None:
        if price_percent <= -10:
            reason = f"Priced {abs(price_percent):.1f}% below similar average"
            labels.append(make_label("good_price", "positive", reason, price_summary))
        elif price_percent >= 12:
            reason = f"Priced {price_percent:.1f}% above similar average"
            labels.append(make_label("premium_price", "neutral", reason, price_summary))

    if (
        product_rating is not None
        and rating_avg is not None
        and price_percent is not None
        and product_rating >= rating_avg + 4
        and price_percent <= -5
    ):
        reason = "Higher rated while priced below similar products"
        labels.append(make_label("high_value_pick", "positive", reason, price_summary))

    if not labels:
        reason = "Comparable to similar products in price and ratings"
        labels.append(make_label("well_aligned", "neutral", reason, price_summary))

    return labels


def transform_product(
    product: Dict[str, Any],
    cluster_map: Dict[int, List[Dict[str, Any]]],
    category_map: Dict[int, List[Dict[str, Any]]],
    spec_keys_map: Dict[int, Set[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """Compute similar products, summary, and labels for a single product."""
    candidates = select_candidates(product, cluster_map, category_map)
    scored = score_candidates(product, candidates, spec_keys_map)

    similar_products: List[Dict[str, Any]] = []
    for rank, entry in enumerate(scored, start=1):
        candidate = entry["candidate"]
        similar_products.append(
            {
                "rank": rank,
                "product_id": candidate["id"],
                "title": candidate.get("title"),
                "brand": candidate.get("brand"),
                "category_id": candidate.get("category_id"),
                "price": entry["price"],
                "rating_average": entry["rating_average"],
                "rating_count": entry["rating_count"],
                "similarity_score": round(entry["similarity_score"], 4),
                "price_diff_percent": round(entry["price_diff_percent"], 2) if entry["price_diff_percent"] is not None else None,
                "rating_average_diff": round(entry["rating_average_diff"], 2) if entry["rating_average_diff"] is not None else None,
                "rating_count_diff": round(entry["rating_count_diff"], 2) if entry["rating_count_diff"] is not None else None,
                "price_relation": entry["price_relation"],
            }
        )

    summary = build_summary(product, scored)
    labels = generate_labels(summary)
    return similar_products, summary, labels


def augment_dataset(data: Dict[str, Any]) -> Dict[str, int]:
    """Augment all products with similarity metadata, returning stats."""
    products: List[Dict[str, Any]] = data.get("products", [])

    cluster_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    category_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    spec_keys_map: Dict[int, Set[str]] = {}

    for product in products:
        cluster_map[product.get("cluster_id")].append(product)
        category_map[product.get("category_id")].append(product)
        spec_keys_map[product["id"]] = extract_spec_keys(product)

    label_counts: Dict[str, int] = defaultdict(int)

    for idx, product in enumerate(products, start=1):
        similar_products, summary, labels = transform_product(product, cluster_map, category_map, spec_keys_map)
        product["similar_products"] = similar_products
        product["similarity_summary"] = summary
        product["comparison_labels"] = labels
        product["price_position_label"] = summary["price"].get("price_position_label")

        for label in labels:
            label_counts[label["label"]] += 1

        if idx % 500 == 0 or idx == len(products):
            print(f"Processed {idx}/{len(products)} products")

    return label_counts


def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_dataset(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def ensure_backup(original_path: str, backup_path: str, data: Dict[str, Any]) -> None:
    if os.path.exists(backup_path):
        return
    print(f"Creating backup at {backup_path}")
    save_dataset(backup_path, data)


def main() -> None:
    print("Augmenting enhanced product dataset with similarity metadata")
    print("=" * 70)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    data = load_dataset(DATASET_PATH)
    ensure_backup(DATASET_PATH, BACKUP_PATH, data)

    label_counts = augment_dataset(data)
    save_dataset(DATASET_PATH, data)

    print("\nAugmentation complete.")
    print("Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {label}: {count}")


if __name__ == "__main__":
    main()


