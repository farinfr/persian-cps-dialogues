#!/usr/bin/env python3
"""
Continue seeding remaining products from specific categories
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
    """Seed remaining products from specific categories"""
    
    # List of categories to continue seeding
    categories = [
        ('notebook-netbook-ultrabook', 'Laptops', 17249),
        ('household-furniture', 'Household Furniture', 91543),
        ('carpet', 'Carpets', 102436),
        ('wearable-gadget', 'Wearable Gadgets', 9341),
        ('book', 'Books', 436579),
    ]
    
    logger.info("=" * 80)
    logger.info("Continue Seeding - Remaining Products")
    logger.info("=" * 80)
    
    total_to_fetch = sum(total for _, _, total in categories)
    logger.info(f"Total products to fetch: {total_to_fetch:,}")
    logger.info(f"Estimated time: ~{(total_to_fetch * 0.5 / 3600):.1f} hours")
    logger.info("=" * 80)
    
    # Connect to database
    if not db.connect():
        logger.error("Failed to connect to database")
        return 1
    
    db.create_tables()
    
    # Seed each category
    for idx, (category_slug, name, total_products) in enumerate(categories, 1):
        logger.info("")
        logger.info("*" * 80)
        logger.info(f"Category {idx}/{len(categories)}: {name} ({category_slug})")
        logger.info(f"Target: {total_products:,} products")
        logger.info("*" * 80)
        
        seeder = ProductSeeder()
        
        try:
            start_time = time.time()
            
            # Seed the category - fetch ALL pages
            seeder.seed_category(
                category_slug=category_slug,
                max_pages=None,  # Get all pages
                fetch_details=True,
                delay_between_requests=0.5
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Completed {name} in {elapsed_time/60:.1f} minutes")
            
        except KeyboardInterrupt:
            logger.info("\n❌ Seeding interrupted by user")
            seeder.close()
            break
            
        except Exception as e:
            logger.error(f"❌ Error seeding {name}: {e}", exc_info=True)
            
        finally:
            seeder.close()
        
        # Small pause between categories
        if idx < len(categories):
            logger.info(f"Waiting 5 seconds before next category...")
            time.sleep(5)
    
    # Final statistics
    try:
        from models import Category, Product
        session = db.get_session()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 80)
        
        categories_in_db = session.query(Category).all()
        for cat in categories_in_db:
            count = session.query(Product).filter_by(category_id=cat.id).count()
            logger.info(f"  {cat.title}: {count:,} products")
        
        total_products = session.query(Product).count()
        logger.info(f"\n📦 Total Products in Database: {total_products:,}")
        
        session.close()
        
    except Exception as e:
        logger.error(f"Failed to get final statistics: {e}")
    
    db.close()
    logger.info("=" * 80)
    logger.info("✅ Seeding completed!")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

