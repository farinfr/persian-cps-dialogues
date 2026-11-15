#!/usr/bin/env python3
"""
Retry fetching descriptions for products that don't have them
"""

import logging
import time
from database import db
from models import Product
from api_client import DigikalaAPIClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def retry_missing_descriptions(delay_between_requests=0.5):
    """Retry fetching descriptions for products without them"""
    
    api_client = DigikalaAPIClient()
    session = db.get_session()
    
    try:
        # Get all products WITHOUT descriptions
        products = session.query(Product).filter(Product.description.is_(None)).all()
        total_products = len(products)
        
        logger.info("=" * 80)
        logger.info(f"Found {total_products:,} products without descriptions")
        logger.info(f"Estimated time: ~{(total_products * delay_between_requests / 60):.1f} minutes")
        logger.info("=" * 80)
        
        if total_products == 0:
            logger.info("✅ All products already have descriptions!")
            return
        
        updated_count = 0
        not_found_count = 0
        error_count = 0
        
        for idx, product in enumerate(products, 1):
            try:
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
                            logger.info(f"✅ Updated product {product.digikala_id}")
                        else:
                            not_found_count += 1
                            logger.info(f"⚠️ No description in review data for {product.digikala_id}")
                    else:
                        not_found_count += 1
                        logger.info(f"⚠️ No review data for product {product.digikala_id}")
                else:
                    error_count += 1
                    logger.warning(f"❌ Failed to fetch detail for {product.digikala_id}")
                
                # Rate limiting
                time.sleep(delay_between_requests)
                
                # Progress update every 100 products
                if idx % 100 == 0:
                    logger.info("")
                    logger.info(f"Progress: {idx}/{total_products} ({(idx/total_products)*100:.1f}%)")
                    logger.info(f"Updated: {updated_count} | Not Found: {not_found_count} | Errors: {error_count}")
                    logger.info("")
                
            except KeyboardInterrupt:
                logger.info("\n❌ Process interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error updating product {product.digikala_id}: {e}")
                error_count += 1
                continue
        
        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("RETRY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total processed: {total_products:,}")
        logger.info(f"✅ Successfully updated: {updated_count:,}")
        logger.info(f"⚠️ No description available: {not_found_count:,}")
        logger.info(f"❌ Errors: {error_count:,}")
        logger.info("")
        
        # Get current stats
        total = session.query(Product).count()
        with_desc = session.query(Product).filter(Product.description.isnot(None)).count()
        logger.info(f"📊 Current Status:")
        logger.info(f"   Total Products: {total:,}")
        logger.info(f"   With Description: {with_desc:,} ({(with_desc/total)*100:.1f}%)")
        logger.info(f"   Without Description: {total-with_desc:,} ({((total-with_desc)/total)*100:.1f}%)")
        logger.info("=" * 80)
        
    finally:
        api_client.close()
        session.close()


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Retry Missing Descriptions")
    logger.info("=" * 80)
    
    # Connect to database
    if not db.connect():
        logger.error("Failed to connect to database")
        return 1
    
    # Retry fetching descriptions
    retry_missing_descriptions(delay_between_requests=0.5)
    
    db.close()
    logger.info("✅ Retry completed!")
    
    return 0


if __name__ == '__main__':
    exit(main())

