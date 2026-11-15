#!/usr/bin/env python3
"""
Generate product embeddings using HuggingFace models
Creates high-quality embeddings for title and description
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available")


class ProductEmbeddingGenerator:
    """Generate embeddings for products using HuggingFace models"""
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', output_dir='embeddings'):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model name. Recommended models:
                - 'paraphrase-multilingual-MiniLM-L12-v2' (384 dim, fast, multilingual)
                - 'intfloat/multilingual-e5-base' (768 dim, high quality)
                - 'sentence-transformers/LaBSE' (768 dim, 109 languages)
                - 'intfloat/multilingual-e5-large' (1024 dim, best quality)
        """
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model = None
        self.df = None
        self.embeddings = None
    
    def load_model(self):
        """Load the embedding model"""
        logger.info(f"Loading model: {self.model_name}")
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            return False
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_products(self, dataset_path):
        """Load product dataset"""
        logger.info(f"Loading products from {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = data.get('products', data)
        self.df = pd.DataFrame(products)
        
        logger.info(f"Loaded {len(self.df)} products")
        
        # Prepare text fields
        self.df['title'] = self.df['title'].fillna('')
        self.df['description'] = self.df['description'].fillna('')
        
        # Count products with description
        with_desc = (self.df['description'] != '').sum()
        logger.info(f"Products with description: {with_desc} ({with_desc/len(self.df)*100:.1f}%)")
        
        return self.df
    
    def generate_embeddings(self, combine_fields=True, batch_size=32):
        """
        Generate embeddings for products
        
        Args:
            combine_fields: If True, combine title + description. If False, embed separately
            batch_size: Batch size for encoding (adjust based on GPU memory)
        """
        logger.info("Generating embeddings...")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Combine title + description: {combine_fields}")
        
        if combine_fields:
            # Combine title and description
            texts = []
            for _, row in self.df.iterrows():
                title = row['title'] if row['title'] else ''
                desc = row['description'] if row['description'] else ''
                
                # Combine with separator
                if title and desc:
                    combined = f"{title}. {desc}"
                elif title:
                    combined = title
                elif desc:
                    combined = desc
                else:
                    combined = "No information"
                
                texts.append(combined)
            
            logger.info(f"Encoding {len(texts)} combined texts...")
            self.embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"✅ Generated embeddings shape: {self.embeddings.shape}")
            
        else:
            # Generate separate embeddings for title and description
            logger.info("Encoding titles...")
            title_embeddings = self.model.encode(
                self.df['title'].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info("Encoding descriptions...")
            desc_embeddings = self.model.encode(
                self.df['description'].tolist(),
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Concatenate or average
            self.embeddings = np.concatenate([title_embeddings, desc_embeddings], axis=1)
            logger.info(f"✅ Generated embeddings shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def save_embeddings(self, format='all'):
        """
        Save embeddings in various formats
        
        Args:
            format: 'numpy', 'json', 'csv', or 'all'
        """
        logger.info("Saving embeddings...")
        
        # Prepare metadata
        metadata = {
            'model_name': self.model_name,
            'embedding_dimension': self.embeddings.shape[1],
            'total_products': self.embeddings.shape[0],
            'generation_date': datetime.now().isoformat(),
            'product_ids': self.df['id'].tolist(),
            'digikala_ids': self.df['digikala_id'].tolist()
        }
        
        if format in ['numpy', 'all']:
            # Save as numpy array
            np.save(f'{self.output_dir}/embeddings.npy', self.embeddings)
            logger.info(f"✅ Saved: {self.output_dir}/embeddings.npy")
            
            # Save metadata
            with open(f'{self.output_dir}/metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Saved: {self.output_dir}/metadata.json")
        
        if format in ['csv', 'all']:
            # Save as CSV with embeddings
            df_export = self.df[['id', 'digikala_id', 'title', 'brand', 'category_name']].copy()
            
            # Add embedding columns
            for i in range(self.embeddings.shape[1]):
                df_export[f'emb_{i}'] = self.embeddings[:, i]
            
            df_export.to_csv(f'{self.output_dir}/embeddings.csv', index=False, encoding='utf-8')
            logger.info(f"✅ Saved: {self.output_dir}/embeddings.csv")
        
        if format in ['json', 'all']:
            # Save as JSON with full product info + embeddings
            products_with_embeddings = []
            
            for idx, row in self.df.iterrows():
                product_data = {
                    'id': int(row['id']) if pd.notna(row['id']) else None,
                    'digikala_id': int(row['digikala_id']) if pd.notna(row['digikala_id']) else None,
                    'title': row['title'],
                    'brand': row['brand'],
                    'category_name': row.get('category_name'),
                    'embedding': self.embeddings[idx].tolist()
                }
                products_with_embeddings.append(product_data)
            
            output_data = {
                'metadata': metadata,
                'products': products_with_embeddings
            }
            
            with open(f'{self.output_dir}/embeddings_full.json', 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Saved: {self.output_dir}/embeddings_full.json")
        
        # Save compact version (embeddings + IDs only)
        compact_data = {
            'metadata': metadata,
            'embeddings': self.embeddings.tolist(),
            'product_ids': self.df['id'].tolist(),
            'digikala_ids': self.df['digikala_id'].tolist()
        }
        
        with open(f'{self.output_dir}/embeddings_compact.json', 'w', encoding='utf-8') as f:
            json.dump(compact_data, f, ensure_ascii=False)
        logger.info(f"✅ Saved: {self.output_dir}/embeddings_compact.json")
    
    def create_similarity_index(self, top_k=10):
        """Create product similarity index using embeddings"""
        logger.info(f"Creating similarity index (top {top_k} similar products)...")
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate similarity matrix (in batches to save memory)
        similarities = cosine_similarity(self.embeddings)
        
        # For each product, find top k similar products
        similarity_index = []
        
        for idx in tqdm(range(len(self.df)), desc="Building similarity index"):
            # Get similarities for this product
            sim_scores = similarities[idx]
            
            # Get top k (excluding itself)
            top_indices = np.argsort(sim_scores)[::-1][1:top_k+1]
            
            similar_products = []
            for sim_idx in top_indices:
                similar_products.append({
                    'product_id': int(self.df.iloc[sim_idx]['id']),
                    'digikala_id': int(self.df.iloc[sim_idx]['digikala_id']),
                    'title': self.df.iloc[sim_idx]['title'],
                    'similarity_score': float(sim_scores[sim_idx])
                })
            
            similarity_index.append({
                'product_id': int(self.df.iloc[idx]['id']),
                'digikala_id': int(self.df.iloc[idx]['digikala_id']),
                'title': self.df.iloc[idx]['title'],
                'similar_products': similar_products
            })
        
        # Save similarity index
        with open(f'{self.output_dir}/similarity_index.json', 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_products': len(similarity_index),
                    'top_k': top_k,
                    'model': self.model_name,
                    'generation_date': datetime.now().isoformat()
                },
                'index': similarity_index
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved: {self.output_dir}/similarity_index.json")
        
        return similarity_index
    
    def run_full_pipeline(self, dataset_path, create_similarity=True):
        """Run complete embedding generation pipeline"""
        logger.info("=" * 80)
        logger.info("PRODUCT EMBEDDING GENERATION PIPELINE")
        logger.info("=" * 80)
        
        # Load model
        if not self.load_model():
            return False
        
        # Load products
        self.load_products(dataset_path)
        
        # Generate embeddings
        self.generate_embeddings(combine_fields=True, batch_size=32)
        
        # Save in all formats
        self.save_embeddings(format='all')
        
        # Create similarity index
        if create_similarity:
            self.create_similarity_index(top_k=10)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ EMBEDDING GENERATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Total Products: {len(self.df)}")
        logger.info(f"Embedding Dimension: {self.embeddings.shape[1]}")
        logger.info(f"Output Directory: {self.output_dir}/")
        logger.info("=" * 80)
        
        return True


def main():
    """Main function"""
    
    # Dataset path
    dataset_path = 'datasets/complete_dataset.json'
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return 1
    
    # Available models (choose one):
    models = {
        'multilingual-mini': 'paraphrase-multilingual-MiniLM-L12-v2',  # 384 dim, fast, good quality
        'multilingual-e5-base': 'intfloat/multilingual-e5-base',  # 768 dim, high quality
        'multilingual-e5-large': 'intfloat/multilingual-e5-large',  # 1024 dim, best quality
        'labse': 'sentence-transformers/LaBSE',  # 768 dim, 109 languages
    }
    
    # Use multilingual-mini for balance of speed and quality
    selected_model = models['multilingual-mini']
    
    logger.info("=" * 80)
    logger.info("Selected Model Information")
    logger.info("=" * 80)
    logger.info(f"Model: {selected_model}")
    logger.info("Features:")
    logger.info("  ✅ Supports 50+ languages (including Persian)")
    logger.info("  ✅ 384-dimensional embeddings")
    logger.info("  ✅ Optimized for semantic similarity")
    logger.info("  ✅ Fast inference")
    logger.info("=" * 80)
    logger.info("")
    
    # Create generator
    generator = ProductEmbeddingGenerator(
        model_name=selected_model,
        output_dir='embeddings'
    )
    
    # Run pipeline
    success = generator.run_full_pipeline(
        dataset_path=dataset_path,
        create_similarity=True
    )
    
    if success:
        # Show sample embeddings
        logger.info("\nSample Embedding Info:")
        logger.info(f"First product: {generator.df.iloc[0]['title'][:60]}...")
        logger.info(f"Embedding shape: {generator.embeddings[0].shape}")
        logger.info(f"Embedding range: [{generator.embeddings[0].min():.4f}, {generator.embeddings[0].max():.4f}]")
        
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())

