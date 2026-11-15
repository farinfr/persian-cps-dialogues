"""Contextual feature augmentation (Section 3.4.2).

Each product is compared with its ten nearest neighbours inside its cluster
(cosine similarity of the MiniLM title+description embeddings) on price,
average rating and number of ratings, which yields descriptive labels such as
"good_price" or "popular_choice" that the dialogue generator can verbalise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .catalog import ROOT

NEIGHBOURS = 10

LABEL_TEXT = {
    "high_rated": "امتیاز کاربران نسبت به محصولات مشابه بالاتر است",
    "needs_rating_improvement": "امتیاز کاربران از محصولات مشابه پایین‌تر است",
    "popular_choice": "تعداد نظرات کاربران نشان می‌دهد محصول پرطرفدار است",
    "niche_pick": "محصول خاص‌تر است و نظرات کمتری دارد",
    "good_price": "قیمت نسبت به محصولات مشابه مقرون‌به‌صرفه است",
    "premium_price": "قیمت نسبت به محصولات مشابه بالاتر است (محصول پریمیوم)",
    "high_value_pick": "ترکیب قیمت و امتیاز، ارزش خرید بالایی ایجاد کرده است",
    "well_aligned": "از نظر قیمت و امتیاز با محصولات مشابه هم‌سطح است",
}
PRICE_POSITION_TEXT = {"budget": "اقتصادی", "mid": "متوسط", "premium": "گران‌تر از مشابه‌ها"}


def load_embeddings(path: Path = ROOT / "data" / "embeddings.npz") -> Dict[int, np.ndarray]:
    z = np.load(path)
    emb = z["emb"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return {int(i): emb[k] for k, i in enumerate(z["ids"])}


def _mean(xs: List[float]):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def augment_cluster(products: List[Dict[str, Any]], emb: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
    """Return {product id: augmentation} for all products of one cluster."""
    ids = [p["id"] for p in products]
    mat = np.stack([emb[i] for i in ids])
    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)
    out: Dict[int, Dict[str, Any]] = {}
    for row, p in enumerate(products):
        k = min(NEIGHBOURS, len(products) - 1)
        nb = np.argsort(-sim[row])[:k]
        peers = [products[j] for j in nb]
        price = p.get("price")
        peer_prices = [q["price"] for q in peers if q.get("price")]
        summary: Dict[str, Any] = {}
        labels: List[str] = []
        if price and peer_prices:
            pct = sum(1 for x in peer_prices if x <= price) / len(peer_prices)
            avg = sum(peer_prices) / len(peer_prices)
            diff = 100.0 * (price - avg) / avg
            summary.update(price_percentile=pct, price_diff_percent=diff,
                           price_position="budget" if pct <= 0.33 else "premium" if pct >= 0.67 else "mid")
            if diff <= -10:
                labels.append("good_price")
            elif diff >= 12:
                labels.append("premium_price")
        r = p.get("rating_average") if p.get("rating_count") else None
        r_avg = _mean([q.get("rating_average") for q in peers if q.get("rating_count")])
        if r is not None and r_avg is not None:
            summary.update(rating=r, peer_rating=r_avg)
            if r >= r_avg + 5 or r >= 85:
                labels.append("high_rated")
            elif r <= r_avg - 10:
                labels.append("needs_rating_improvement")
        c = p.get("rating_count") or 0
        c_avg = _mean([q.get("rating_count") or 0 for q in peers])
        if c_avg:
            summary.update(rating_count=c, peer_rating_count=c_avg)
            if c >= 1.35 * c_avg and c >= 5:
                labels.append("popular_choice")
            elif c <= 0.5 * c_avg:
                labels.append("niche_pick")
        if r is not None and r_avg is not None and summary.get("price_diff_percent", 0) <= -5 and r >= r_avg + 4:
            labels.append("high_value_pick")
        if not labels:
            labels.append("well_aligned")
        out[p["id"]] = {
            "neighbours": [products[j]["id"] for j in nb],
            "neighbour_similarity": [float(sim[row, j]) for j in nb],
            "summary": summary,
            "labels": labels,
        }
    return out
