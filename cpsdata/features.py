"""Per-cluster feature tables used by the RPB score and the simulator.

For every cluster we build an integer matrix: one column per feature, one row
per product, holding the index of the product's value or -1 when the product
lacks the feature (the *default group*).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .catalog import leading_number, normalize_text

BRAND = "برند"
PRICE = "قیمت"
MISC_BRAND = "متفرقه"  # "miscellaneous": not a value a user can name

MULTI_VALUE_LIMIT = 0.10   # exclude a feature if >10% of its products carry several values
NUMERIC_MIN_VALUES = 12    # numerical features with more values are binned
NUMERIC_BINS = 6
NUMERIC_PARSE_SHARE = 0.90
FREE_TEXT_MEAN_CHARS = 40     # values longer than this on average are free text
FREE_TEXT_NAMES = ("توضیحات",)  # e.g. "سایر توضیحات" (other descriptions)


@dataclass
class Feature:
    name: str
    codes: np.ndarray            # (n,) value index or -1
    labels: List[str]            # value labels
    numeric: bool = False
    free_text: bool = False      # long descriptive values a user cannot give as an answer

    @property
    def m(self) -> int:
        return len(self.labels)

    def value_of(self, row: int) -> Optional[str]:
        c = int(self.codes[row])
        return None if c < 0 else self.labels[c]


@dataclass
class ClusterTable:
    cluster: int
    products: List[Dict[str, Any]]
    popularity: np.ndarray
    features: List[Feature]
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.products)

    def codes(self) -> np.ndarray:
        """(n, F) matrix of value codes."""
        if not self.features:
            return np.zeros((self.n, 0), dtype=np.int32)
        return np.stack([f.codes for f in self.features], axis=1)


def _bin_numeric(nums: Dict[int, float], n: int, unit: str) -> Optional[Feature]:
    vals = np.array(list(nums.values()))
    edges = np.unique(np.quantile(vals, np.linspace(0, 1, NUMERIC_BINS + 1)))
    if len(edges) < 3:
        return None
    codes = np.full(n, -1, dtype=np.int32)
    for row, v in nums.items():
        codes[row] = min(int(np.searchsorted(edges, v, side="right")) - 1, len(edges) - 2)
    def fmt(x: float) -> str:
        return f"{x:,.0f}" if abs(x) >= 100 or float(x).is_integer() else f"{x:g}"
    labels = [f"{fmt(edges[i])} تا {fmt(edges[i + 1])}{(' ' + unit) if unit else ''}" for i in range(len(edges) - 1)]
    used = sorted(set(codes[codes >= 0].tolist()))
    remap = {old: new for new, old in enumerate(used)}
    codes = np.array([remap.get(int(c), -1) if c >= 0 else -1 for c in codes], dtype=np.int32)
    return Feature("", codes, [labels[u] for u in used], numeric=True)


def _unit_of(values: Sequence[str]) -> str:
    units = Counter(v.split(" ", 1)[1] if " " in v else "" for v in values)
    unit, _ = units.most_common(1)[0]
    return unit if len(unit) <= 20 else ""


def build_cluster_table(cluster: int, products: List[Dict[str, Any]]) -> ClusterTable:
    n = len(products)
    raw: Dict[str, Dict[int, tuple]] = {}
    for row, p in enumerate(products):
        for name, values in (p.get("specs") or {}).items():
            raw.setdefault(name, {})[row] = tuple(values)
        brand = normalize_text(p.get("brand"))
        if brand and brand != MISC_BRAND:
            raw.setdefault(BRAND, {})[row] = (brand,)
    stats = Counter()
    features: List[Feature] = []

    # Price is always numerical and binned.
    prices = {row: float(p["price"]) for row, p in enumerate(products) if p.get("price")}
    if len(set(prices.values())) > 1:
        f = _bin_numeric(prices, n, "ریال")
        if f is not None and f.m >= 2:
            f.name = PRICE
            features.append(f)

    for name, carriers in raw.items():
        multi = sum(1 for v in carriers.values() if len(v) > 1)
        if multi > MULTI_VALUE_LIMIT * len(carriers):
            stats["excluded_multi_valued"] += 1
            continue
        single = {row: v[0] for row, v in carriers.items() if len(v) == 1}
        distinct = sorted(set(single.values()))
        if len(distinct) < 2:
            stats["single_valued"] += 1
            continue
        nums = {row: leading_number(v) for row, v in single.items()}
        parsed = {row: x for row, x in nums.items() if x is not None}
        if len(distinct) > NUMERIC_MIN_VALUES and len(parsed) >= NUMERIC_PARSE_SHARE * len(single):
            f = _bin_numeric(parsed, n, _unit_of(list(single.values())))
            if f is None or f.m < 2:
                stats["single_valued"] += 1
                continue
            f.name = name
            features.append(f)
            continue
        index = {v: i for i, v in enumerate(distinct)}
        codes = np.full(n, -1, dtype=np.int32)
        for row, v in single.items():
            codes[row] = index[v]
        free = (any(t in name for t in FREE_TEXT_NAMES)
                or sum(len(v) for v in single.values()) / len(single) > FREE_TEXT_MEAN_CHARS)
        features.append(Feature(name, codes, distinct, free_text=free))

    stats["features"] = len(features)
    popularity = np.array([float(p.get("rating_count") or 0) + 1.0 for p in products])
    return ClusterTable(cluster, products, popularity, features, dict(stats))
