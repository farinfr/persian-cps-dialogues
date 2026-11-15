#!/usr/bin/env python3
"""
Export products and categories data to dataset files
"""

import os
import json
import csv
import logging
from datetime import datetime
from database import db
from models import Product, Category

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_to_json(output_dir='datasets'):
    """Export data to JSON format"""
    logger.info("Exporting to JSON format...")
    
    os.makedirs(output_dir, exist_ok=True)
    session = db.get_session()
    
    try:
        # Export categories
        categories = session.query(Category).all()
        categories_data = []
        
        for cat in categories:
            categories_data.append({
                'id': cat.id,
                'digikala_id': cat.digikala_id,
                'title': cat.title,
                'title_en': cat.title_en,
                'code': cat.code,
                'url': cat.url,
                'created_at': cat.created_at.isoformat() if cat.created_at else None,
                'updated_at': cat.updated_at.isoformat() if cat.updated_at else None
            })
        
        # Save categories JSON
        with open(f'{output_dir}/categories.json', 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Exported {len(categories_data)} categories to {output_dir}/categories.json")
        
        # Export products
        products = session.query(Product).all()
        products_data = []
        
        for prod in products:
            products_data.append({
                'id': prod.id,
                'digikala_id': prod.digikala_id,
                'category_id': prod.category_id,
                'category_name': prod.category.title if prod.category else None,
                'title': prod.title,
                'title_en': prod.title_en,
                'url': prod.url,
                'brand': prod.brand,
                'description': prod.description,
                'price': prod.price,
                'selling_price': prod.selling_price,
                'discount_percent': prod.discount_percent,
                'status': prod.status,
                'is_incredible': prod.is_incredible,
                'is_fresh': prod.is_fresh,
                'main_image_url': prod.main_image_url,
                'images': prod.images,
                'rating_average': prod.rating_average,
                'rating_count': prod.rating_count,
                'is_available': prod.is_available,
                'specifications': prod.specifications,
                'created_at': prod.created_at.isoformat() if prod.created_at else None,
                'updated_at': prod.updated_at.isoformat() if prod.updated_at else None
            })
        
        # Save products JSON
        with open(f'{output_dir}/products.json', 'w', encoding='utf-8') as f:
            json.dump(products_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Exported {len(products_data)} products to {output_dir}/products.json")
        
        # Export combined dataset
        combined_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_categories': len(categories_data),
                'total_products': len(products_data)
            },
            'categories': categories_data,
            'products': products_data
        }
        
        with open(f'{output_dir}/complete_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Exported complete dataset to {output_dir}/complete_dataset.json")
        
    finally:
        session.close()


def export_to_csv(output_dir='datasets'):
    """Export data to CSV format"""
    logger.info("Exporting to CSV format...")
    
    os.makedirs(output_dir, exist_ok=True)
    session = db.get_session()
    
    try:
        # Export categories CSV
        categories = session.query(Category).all()
        
        with open(f'{output_dir}/categories.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'digikala_id', 'title', 'title_en', 'code', 'url', 'created_at', 'updated_at'])
            
            for cat in categories:
                writer.writerow([
                    cat.id,
                    cat.digikala_id,
                    cat.title,
                    cat.title_en,
                    cat.code,
                    cat.url,
                    cat.created_at.isoformat() if cat.created_at else '',
                    cat.updated_at.isoformat() if cat.updated_at else ''
                ])
        
        logger.info(f"✅ Exported {len(categories)} categories to {output_dir}/categories.csv")
        
        # Export products CSV (with basic fields, specs in separate column)
        products = session.query(Product).all()
        
        with open(f'{output_dir}/products.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'digikala_id', 'category_id', 'category_name', 'title', 'title_en',
                'url', 'brand', 'description', 'price', 'selling_price', 'discount_percent',
                'status', 'is_incredible', 'is_fresh', 'main_image_url', 'rating_average',
                'rating_count', 'is_available', 'specifications_json', 'images_json',
                'created_at', 'updated_at'
            ])
            
            for prod in products:
                writer.writerow([
                    prod.id,
                    prod.digikala_id,
                    prod.category_id,
                    prod.category.title if prod.category else '',
                    prod.title,
                    prod.title_en,
                    prod.url,
                    prod.brand,
                    prod.description,
                    prod.price,
                    prod.selling_price,
                    prod.discount_percent,
                    prod.status,
                    prod.is_incredible,
                    prod.is_fresh,
                    prod.main_image_url,
                    prod.rating_average,
                    prod.rating_count,
                    prod.is_available,
                    json.dumps(prod.specifications, ensure_ascii=False) if prod.specifications else '',
                    json.dumps(prod.images, ensure_ascii=False) if prod.images else '',
                    prod.created_at.isoformat() if prod.created_at else '',
                    prod.updated_at.isoformat() if prod.updated_at else ''
                ])
        
        logger.info(f"✅ Exported {len(products)} products to {output_dir}/products.csv")
        
    finally:
        session.close()


def export_products_by_category(output_dir='datasets/by_category'):
    """Export products grouped by category"""
    logger.info("Exporting products by category...")
    
    os.makedirs(output_dir, exist_ok=True)
    session = db.get_session()
    
    try:
        categories = session.query(Category).all()
        
        for cat in categories:
            products = session.query(Product).filter_by(category_id=cat.id).all()
            
            if products:
                products_data = []
                for prod in products:
                    products_data.append({
                        'id': prod.id,
                        'digikala_id': prod.digikala_id,
                        'title': prod.title,
                        'brand': prod.brand,
                        'description': prod.description,
                        'price': prod.price,
                        'selling_price': prod.selling_price,
                        'discount_percent': prod.discount_percent,
                        'main_image_url': prod.main_image_url,
                        'images': prod.images,
                        'rating_average': prod.rating_average,
                        'rating_count': prod.rating_count,
                        'specifications': prod.specifications
                    })
                
                # Sanitize filename
                filename = cat.code or f"category_{cat.digikala_id}"
                filename = filename.replace('/', '_').replace('\\', '_')
                
                with open(f'{output_dir}/{filename}.json', 'w', encoding='utf-8') as f:
                    json.dump({
                        'category': {
                            'id': cat.id,
                            'title': cat.title,
                            'code': cat.code
                        },
                        'total_products': len(products_data),
                        'products': products_data
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"✅ Exported {len(products_data)} products for category: {cat.title}")
        
    finally:
        session.close()


def create_summary_report(output_dir='datasets'):
    """Create a summary report of the dataset"""
    logger.info("Creating summary report...")
    
    session = db.get_session()
    
    try:
        total_categories = session.query(Category).count()
        total_products = session.query(Product).count()
        products_with_desc = session.query(Product).filter(Product.description.isnot(None)).count()
        products_with_specs = session.query(Product).filter(Product.specifications.isnot(None)).count()
        products_with_images = session.query(Product).filter(Product.images.isnot(None)).count()
        
        # Get category breakdown
        categories = session.query(Category).all()
        category_breakdown = []
        
        for cat in categories:
            count = session.query(Product).filter_by(category_id=cat.id).count()
            category_breakdown.append({
                'category': cat.title,
                'code': cat.code,
                'product_count': count
            })
        
        # Get top brands
        from sqlalchemy import func
        top_brands = session.query(
            Product.brand,
            func.count(Product.id).label('count')
        ).filter(
            Product.brand.isnot(None)
        ).group_by(
            Product.brand
        ).order_by(
            func.count(Product.id).desc()
        ).limit(10).all()
        
        summary = {
            'export_date': datetime.now().isoformat(),
            'statistics': {
                'total_categories': total_categories,
                'total_products': total_products,
                'products_with_description': products_with_desc,
                'products_with_description_percent': round((products_with_desc / total_products * 100), 2) if total_products > 0 else 0,
                'products_with_specifications': products_with_specs,
                'products_with_images': products_with_images
            },
            'category_breakdown': category_breakdown,
            'top_10_brands': [{'brand': brand, 'count': count} for brand, count in top_brands]
        }
        
        with open(f'{output_dir}/dataset_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Created summary report at {output_dir}/dataset_summary.json")
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Categories: {total_categories}")
        logger.info(f"Total Products: {total_products}")
        logger.info(f"Products with Description: {products_with_desc} ({round((products_with_desc / total_products * 100), 2)}%)")
        logger.info(f"Products with Specifications: {products_with_specs}")
        logger.info(f"Products with Images: {products_with_images}")
        logger.info("")
        logger.info("Category Breakdown:")
        for cat in category_breakdown:
            logger.info(f"  • {cat['category']}: {cat['product_count']} products")
        logger.info("=" * 80)
        
    finally:
        session.close()


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Dataset Export Tool")
    logger.info("=" * 80)
    
    # Connect to database
    if not db.connect():
        logger.error("Failed to connect to database")
        return 1
    
    # Create output directory
    output_dir = 'datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Exporting datasets to: {output_dir}/")
    logger.info("")
    
    # Export in different formats
    export_to_json(output_dir)
    logger.info("")
    
    export_to_csv(output_dir)
    logger.info("")
    
    export_products_by_category()
    logger.info("")
    
    create_summary_report(output_dir)
    logger.info("")
    
    db.close()
    
    logger.info("=" * 80)
    logger.info("✅ Dataset export completed!")
    logger.info(f"📁 Files saved in: {output_dir}/")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

