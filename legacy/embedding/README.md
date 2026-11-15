# Product Embedding Generator

Generate high-quality embeddings for products using state-of-the-art HuggingFace models.

## Features

- ✅ **Multilingual Support**: Works with Persian, English, and 50+ languages
- ✅ **State-of-the-art Models**: Uses sentence-transformers from HuggingFace
- ✅ **Multiple Output Formats**: NumPy, JSON, CSV
- ✅ **Similarity Index**: Pre-computed similar products for each item
- ✅ **Batch Processing**: Efficient GPU/CPU utilization

## Supported Models

1. **paraphrase-multilingual-MiniLM-L12-v2** (Default)
   - Dimension: 384
   - Speed: Fast ⚡⚡⚡
   - Quality: High ⭐⭐⭐⭐
   - Best for: Production use

2. **intfloat/multilingual-e5-base**
   - Dimension: 768
   - Speed: Medium ⚡⚡
   - Quality: Very High ⭐⭐⭐⭐⭐
   - Best for: High accuracy needs

3. **intfloat/multilingual-e5-large**
   - Dimension: 1024
   - Speed: Slow ⚡
   - Quality: Excellent ⭐⭐⭐⭐⭐
   - Best for: Maximum quality

4. **sentence-transformers/LaBSE**
   - Dimension: 768
   - Languages: 109
   - Best for: Multi-language datasets

## Installation

```bash
cd embedding_generator
pip install -r requirements.txt
```

## Usage

```bash
python generate_embeddings.py
```

## Output Files

The script generates the following files in `embeddings/`:

1. **embeddings.npy** - NumPy array of embeddings (fast loading)
2. **metadata.json** - Model and dataset metadata
3. **embeddings.csv** - CSV with product info + embeddings
4. **embeddings_full.json** - Complete product data with embeddings
5. **embeddings_compact.json** - Embeddings + IDs only (smaller file)
6. **similarity_index.json** - Pre-computed similar products

## Output Format

### embeddings.npy
```python
# Shape: (n_products, embedding_dim)
# Example: (12926, 384)
```

### embeddings_full.json
```json
{
  "metadata": {
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
    "embedding_dimension": 384,
    "total_products": 12926
  },
  "products": [
    {
      "id": 1,
      "digikala_id": 17580036,
      "title": "گوشی موبایل...",
      "brand": "شیائومی",
      "embedding": [0.123, -0.456, ...]
    }
  ]
}
```

### similarity_index.json
```json
{
  "index": [
    {
      "product_id": 1,
      "title": "Product A",
      "similar_products": [
        {
          "product_id": 42,
          "title": "Product B",
          "similarity_score": 0.95
        }
      ]
    }
  ]
}
```

## Use Cases

1. **Product Recommendations**: Find similar products using cosine similarity
2. **Semantic Search**: Search products by meaning, not just keywords
3. **Duplicate Detection**: Identify duplicate or very similar products
4. **Advanced Clustering**: Use embeddings for better clustering
5. **Product Matching**: Match products across different sources

## Loading Embeddings

### Python
```python
import numpy as np
import json

# Load embeddings
embeddings = np.load('embeddings/embeddings.npy')

# Load metadata
with open('embeddings/metadata.json', 'r') as f:
    metadata = json.load(f)

# Load similarity index
with open('embeddings/similarity_index.json', 'r') as f:
    similarity = json.load(f)
```

## Performance

- **~12,926 products**: ~2-5 minutes on CPU
- **With GPU**: ~30-60 seconds
- **Memory usage**: ~2-4 GB for large models

## Model Comparison

| Model | Dimension | Speed | Quality | File Size |
|-------|-----------|-------|---------|-----------|
| MiniLM | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ | ~20 MB |
| E5-Base | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ | ~40 MB |
| E5-Large | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | ~53 MB |

**Recommendation**: Use **MiniLM** for production (best speed/quality balance)

