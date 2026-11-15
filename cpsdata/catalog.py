"""Loading and normalising the Digikala catalog.

The catalog is stored as ``data/catalog.jsonl.gz``: one product per line with
its id, title, brand, category, price, rating, description, flattened
specifications and K-Means cluster (k=170).  ``scripts/build_catalog.py``
creates it from the raw crawl.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalog.jsonl.gz"

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_CHARS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", " ": " ", "‏": "", "‎": ""})
_WS = re.compile(r"\s+")
_ZWNJ_SPACES = re.compile(r"\s*‌\s*")


def normalize_text(value: Any) -> str:
    """Normalise Persian text: digits, Arabic ye/kaf, ZWNJ and whitespace."""
    if value is None:
        return ""
    text = str(value).translate(_DIGITS).translate(_CHARS)
    text = _ZWNJ_SPACES.sub("‌", text)
    text = _WS.sub(" ", text).strip(" /\r\n\t")
    return text


_NUM = re.compile(r"^\s*(\d+(?:[.,/٫]\d+)?)")


def leading_number(text: str) -> Optional[float]:
    """Return the number at the start of a value such as '128 گیگابایت'."""
    m = _NUM.match(text)
    if not m:
        return None
    raw = m.group(1).replace(",", ".").replace("٫", ".").replace("/", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def flatten_specs(specs: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[str, ...]]:
    """Flatten ``{group: {key: value}}`` into ``{"group / key": (values...)}``.

    List values are kept as tuples so that multi-valued features can be
    detected; empty values are dropped.
    """
    flat: Dict[str, Tuple[str, ...]] = {}
    for group, kv in (specs or {}).items():
        if not isinstance(kv, dict):
            continue
        for key, value in kv.items():
            values = value if isinstance(value, list) else [value]
            norm = tuple(v for v in (normalize_text(x) for x in values) if v)
            if norm:
                name = f"{normalize_text(group)} / {normalize_text(key)}"
                flat[name] = tuple(sorted(set(norm)))
    return flat


def load_catalog(path: Path = CATALOG_PATH) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def by_cluster(products: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for p in products:
        clusters.setdefault(int(p["cluster"]), []).append(p)
    return clusters
