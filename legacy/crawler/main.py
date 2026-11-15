#!/usr/bin/env python3
"""
Digikala Product Seeder
Main script to seed product data from Digikala API to PostgreSQL database
"""

import argparse
import logging
from database import db
from seeder import ProductSeeder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the seeder"""
    parser = argparse.ArgumentParser(
        description='Seed product data from Digikala API to PostgreSQL database'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='mobile-phone',
        help='Category slug to seed (default: mobile-phone)'
    )
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to fetch (default: all pages)'
    )
    parser.add_argument(
        '--no-details',
        action='store_true',
        help='Skip fetching detailed product information'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API requests in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database (create tables)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Digikala Product Seeder")
    logger.info("=" * 60)
    
    # Connect to database
    logger.info("Connecting to database...")
    if not db.connect():
        logger.error("Failed to connect to database. Please check your configuration.")
        return 1
    
    # Initialize database if requested
    if args.init_db:
        logger.info("Initializing database tables...")
        if not db.create_tables():
            logger.error("Failed to create database tables")
            return 1
        logger.info("Database initialized successfully")
    
    # Create seeder instance
    seeder = ProductSeeder()
    
    try:
        # Run seeding
        logger.info(f"Starting to seed category: {args.category}")
        logger.info(f"Max pages: {args.max_pages or 'All'}")
        logger.info(f"Fetch details: {not args.no_details}")
        logger.info(f"Request delay: {args.delay}s")
        logger.info("-" * 60)
        
        seeder.seed_category(
            category_slug=args.category,
            max_pages=args.max_pages,
            fetch_details=not args.no_details,
            delay_between_requests=args.delay
        )
        
        logger.info("-" * 60)
        logger.info("Seeding completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nSeeding interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1
    finally:
        seeder.close()
        db.close()
    
    return 0


if __name__ == '__main__':
    exit(main())

