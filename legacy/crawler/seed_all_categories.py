#!/usr/bin/env python3
"""
Seed multiple categories from Digikala API
"""

import logging
import time
from database import db
from seeder import ProductSeeder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Seed all specified categories"""
    
    # List of categories to seed
    categories = [
        'stationery',
        'notebook-netbook-ultrabook',
        'headphone',
        'household-furniture',
        'carpet',
        'wearable-gadget',
        'women-accessories',
        'men-accessories',
        'book'
    ]
    
    logger.info("=" * 80)
    logger.info("Multi-Category Seeder - Digikala Products")
    logger.info("=" * 80)
    logger.info(f"Categories to seed: {len(categories)}")
    for cat in categories:
        logger.info(f"  - {cat}")
    logger.info("=" * 80)
    
    # Connect to database
    logger.info("Connecting to database...")
    if not db.connect():
        logger.error("Failed to connect to database")
        return 1
    
    # Create tables if needed
    db.create_tables()
    
    # Track statistics
    total_start_time = time.time()
    success_count = 0
    failed_categories = []
    
    # Seed each category
    for idx, category_slug in enumerate(categories, 1):
        logger.info("")
        logger.info("*" * 80)
        logger.info(f"Processing Category {idx}/{len(categories)}: {category_slug}")
        logger.info("*" * 80)
        
        seeder = ProductSeeder()
        
        try:
            start_time = time.time()
            
            # Seed the category (fetch all pages, with details, 0.5s delay)
            seeder.seed_category(
                category_slug=category_slug,
                max_pages=None,  # Get all pages
                fetch_details=True,
                delay_between_requests=0.5
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Completed {category_slug} in {elapsed_time/60:.1f} minutes")
            success_count += 1
            
        except KeyboardInterrupt:
            logger.info("\n❌ Seeding interrupted by user")
            seeder.close()
            break
            
        except Exception as e:
            logger.error(f"❌ Failed to seed {category_slug}: {e}", exc_info=True)
            failed_categories.append(category_slug)
            
        finally:
            seeder.close()
        
        # Small pause between categories
        if idx < len(categories):
            logger.info(f"Waiting 2 seconds before next category...")
            time.sleep(2)
    
    # Summary
    total_elapsed = time.time() - total_start_time
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    logger.info(f"Successfully seeded: {success_count}/{len(categories)} categories")
    
    if failed_categories:
        logger.info(f"Failed categories: {', '.join(failed_categories)}")
    
    # Get final statistics from database
    try:
        from models import Category, Product
        session = db.get_session()
        
        total_products = session.query(Product).count()
        total_categories = session.query(Category).count()
        
        logger.info("")
        logger.info(f"📁 Total Categories in DB: {total_categories}")
        logger.info(f"📦 Total Products in DB: {total_products}")
        logger.info("")
        
        # Show products per category
        categories_in_db = session.query(Category).all()
        for cat in categories_in_db:
            count = session.query(Product).filter_by(category_id=cat.id).count()
            logger.info(f"  {cat.title}: {count} products")
        
        session.close()
        
    except Exception as e:
        logger.error(f"Failed to get final statistics: {e}")
    
    db.close()
    logger.info("=" * 80)
    logger.info("✅ All categories processed!")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

