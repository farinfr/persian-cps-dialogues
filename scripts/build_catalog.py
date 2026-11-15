#!/usr/bin/env python3
"""Build data/catalog.jsonl.gz and data/embeddings.npz from the raw crawl.

Inputs (produced by legacy/crawler and legacy/clustering):
  --products     products.json exported from the PostgreSQL crawl database
  --clusters     final_clustered_products.json (K-Means, k=170, MiniLM embeddings)
  --embeddings   embeddings.npy + metadata.json from legacy/embedding_generator
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpsdata.catalog import flatten_specs, normalize_text  # noqa: E402


def to_float(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def parse_specs(s):
    if isinstance(s, dict):
        return s
    if not s or s == "None":
        return {}
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--embeddings", required=True, help="directory with embeddings.npy and metadata.json")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    products = json.load(open(args.products, encoding="utf-8"))
    products = products["products"] if isinstance(products, dict) else products
    clusters = {int(p["id"]): int(p["cluster"]) for p in json.load(open(args.clusters, encoding="utf-8"))["products"]}

    n = 0
    with gzip.open(out / "catalog.jsonl.gz", "wt", encoding="utf-8") as fh:
        for p in products:
            pid = int(p["id"])
            if pid not in clusters:
                continue
            price = to_float(p.get("selling_price")) or to_float(p.get("price"))
            rec = {
                "id": pid,
                "digikala_id": int(p["digikala_id"]),
                "cluster": clusters[pid],
                "category": normalize_text(p.get("category_name")),
                "title": normalize_text(p.get("title")),
                "title_en": normalize_text(p.get("title_en")),
                "brand": normalize_text(p.get("brand")),
                "price": price,
                "rating_average": to_float(p.get("rating_average")),
                "rating_count": int(to_float(p.get("rating_count")) or 0),
                "description": normalize_text(p.get("description")),
                "url": "https://www.digikala.com" + (p.get("url") or ""),
                "specs": {k: list(v) for k, v in flatten_specs(parse_specs(p.get("specifications"))).items()},
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"catalog: {n} products -> {out / 'catalog.jsonl.gz'}")

    emb_dir = Path(args.embeddings)
    emb = np.load(emb_dir / "embeddings.npy")
    meta = json.load(open(emb_dir / "metadata.json", encoding="utf-8"))
    ids = np.array([int(x) for x in meta["product_ids"]])
    np.savez_compressed(out / "embeddings.npz", ids=ids, emb=emb.astype(np.float16))
    print(f"embeddings: {emb.shape} ({meta['model_name']}) -> {out / 'embeddings.npz'}")


if __name__ == "__main__":
    main()
