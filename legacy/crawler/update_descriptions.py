#!/usr/bin/env python3
"""
Update all existing products with descriptions from Digikala API
"""

import logging
import time
from database import db
from models import Product
from api_client import DigikalaAPIClient
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_description_column():
    """Add description column to products table if it doesn't exist"""
    try:
        with db.engine.connect() as connection:
            # Check if column exists
            result = connection.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='products' AND column_name='description'"
            ))
            
            if result.fetchone() is None:
                # Column doesn't exist, add it
                logger.info("Adding description column to products table...")
                connection.execute(text(
                    "ALTER TABLE products ADD COLUMN description TEXT"
                ))
                connection.commit()
                logger.info("✅ Description column added successfully")
            else:
                logger.info("Description column already exists")
    except Exception as e:
        logger.error(f"Error adding description column: {e}")
        raise


def update_product_descriptions(delay_between_requests=0.5):
    """Update all products with descriptions from API"""
    
    api_client = DigikalaAPIClient()
    session = db.get_session()
    
    try:
        # Get all products
        products = session.query(Product).all()
        total_products = len(products)
        
        logger.info(f"Found {total_products} products to update")
        logger.info(f"Estimated time: ~{(total_products * delay_between_requests / 60):.1f} minutes")
        logger.info("=" * 80)
        
        updated_count = 0
        skipped_count = 0
        error_count = 0
        
        for idx, product in enumerate(products, 1):
            try:
                # Skip if already has description
                if product.description:
                    logger.info(f"[{idx}/{total_products}] Skipping product {product.digikala_id} - already has description")
                    skipped_count += 1
                    continue
                
                logger.info(f"[{idx}/{total_products}] Fetching description for product {product.digikala_id}")
                
                # Fetch product detail
                detail = api_client.get_product_detail(product.digikala_id)
                
                if detail and 'data' in detail:
                    detail_data = detail['data'].get('product', {})
                    review_data = detail_data.get('review', {})
                    
                    if review_data and isinstance(review_data, dict):
                        description = review_data.get('description')
                        
                        if description:
                            product.description = description
                            session.commit()
                            updated_count += 1
                            logger.info(f"✅ Updated product {product.digikala_id} with description")
                        else:
                            logger.info(f"⚠️ No description found for product {product.digikala_id}")
                            skipped_count += 1
                    else:
                        logger.info(f"⚠️ No review data for product {product.digikala_id}")
                        skipped_count += 1
                else:
                    logger.warning(f"❌ Failed to fetch detail for product {product.digikala_id}")
                    error_count += 1
                
                # Rate limiting
                time.sleep(delay_between_requests)
                
                # Progress update every 50 products
                if idx % 50 == 0:
                    logger.info("")
                    logger.info(f"Progress: {idx}/{total_products} ({(idx/total_products)*100:.1f}%)")
                    logger.info(f"Updated: {updated_count} | Skipped: {skipped_count} | Errors: {error_count}")
                    logger.info("")
                
            except KeyboardInterrupt:
                logger.info("\n❌ Update interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error updating product {product.digikala_id}: {e}")
                error_count += 1
                continue
        
        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("UPDATE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total products: {total_products}")
        logger.info(f"✅ Successfully updated: {updated_count}")
        logger.info(f"⏭️  Skipped (already had description): {skipped_count}")
        logger.info(f"❌ Errors: {error_count}")
        logger.info("=" * 80)
        
    finally:
        api_client.close()
        session.close()


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Update Product Descriptions")
    logger.info("=" * 80)
    
    # Connect to database
    if not db.connect():
        logger.error("Failed to connect to database")
        return 1
    
    # Add description column if it doesn't exist
    add_description_column()
    
    # Update all products
    logger.info("")
    logger.info("Starting to update product descriptions...")
    logger.info("")
    
    update_product_descriptions(delay_between_requests=0.5)
    
    db.close()
    logger.info("✅ Update completed!")
    
    return 0


if __name__ == '__main__':
    exit(main())

