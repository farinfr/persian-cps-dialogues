import logging
from typing import Optional, Dict, List
from sqlalchemy.exc import SQLAlchemyError
from api_client import DigikalaAPIClient
from database import db
from models import Category, Product
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductSeeder:
    """Service to seed products from Digikala API to database"""
    
    def __init__(self):
        self.api_client = DigikalaAPIClient()
        self.session = None
    
    def filter_specifications(self, specifications: List[Dict]) -> Dict:
        """
        Extract ALL product specifications
        
        Args:
            specifications: Raw specifications from API
        
        Returns:
            Dictionary of all specifications organized by groups
        """
        all_specs = {}
        
        if not specifications:
            return all_specs
        
        for spec_group in specifications:
            if not isinstance(spec_group, dict):
                continue
            
            group_title = spec_group.get('title', 'Other')
            attributes = spec_group.get('attributes', [])
            
            # Create a sub-dictionary for this specification group
            if attributes:
                group_specs = {}
                for attr in attributes:
                    if not isinstance(attr, dict):
                        continue
                    
                    attr_title = attr.get('title', '')
                    values = attr.get('values', [])
                    
                    if attr_title and values:
                        # Extract the value(s)
                        if isinstance(values, list):
                            if len(values) == 1:
                                # Single value - store as string
                                value = values[0] if isinstance(values[0], str) else str(values[0])
                                group_specs[attr_title] = value
                            else:
                                # Multiple values - store as list
                                group_specs[attr_title] = [
                                    v if isinstance(v, str) else str(v) for v in values
                                ]
                        else:
                            group_specs[attr_title] = str(values)
                
                if group_specs:
                    all_specs[group_title] = group_specs
        
        return all_specs
    
    def extract_category_data(self, category_data: Dict) -> Optional[Dict]:
        """Extract category information from API response"""
        if not category_data:
            return None
        
        return {
            'digikala_id': category_data.get('id'),
            'title': category_data.get('title_fa') or category_data.get('title'),
            'title_en': category_data.get('title_en'),
            'code': category_data.get('code'),
            'url': category_data.get('url_path'),
        }
    
    def save_or_update_category(self, category_data: Dict) -> Optional[Category]:
        """Save or update category in database"""
        if not category_data or not category_data.get('digikala_id'):
            return None
        
        try:
            # Check if category exists
            category = self.session.query(Category).filter_by(
                digikala_id=category_data['digikala_id']
            ).first()
            
            if category:
                # Update existing category
                for key, value in category_data.items():
                    if hasattr(category, key):
                        setattr(category, key, value)
                logger.info(f"Updated category: {category_data.get('title')}")
            else:
                # Create new category
                category = Category(**category_data)
                self.session.add(category)
                logger.info(f"Created new category: {category_data.get('title')}")
            
            self.session.commit()
            return category
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to save category: {e}")
            self.session.rollback()
            return None
    
    def extract_product_data(self, product_list_item: Dict, product_detail: Optional[Dict] = None) -> Dict:
        """Extract product information from API response"""
        product_data = {}
        
        # Extract from list data
        product_data['digikala_id'] = product_list_item.get('id')
        product_data['title'] = product_list_item.get('title_fa') or product_list_item.get('title')
        product_data['title_en'] = product_list_item.get('title_en')
        product_data['url'] = product_list_item.get('url', {}).get('uri') if isinstance(
            product_list_item.get('url'), dict
        ) else product_list_item.get('url')
        
        # Brand
        brand_data = product_list_item.get('brand', {})
        if isinstance(brand_data, dict):
            product_data['brand'] = brand_data.get('title_fa') or brand_data.get('title')
        else:
            product_data['brand'] = brand_data
        
        # Default price
        default_variant = product_list_item.get('default_variant', {})
        if default_variant:
            price_data = default_variant.get('price', {})
            product_data['price'] = price_data.get('rrp_price')
            product_data['selling_price'] = price_data.get('selling_price')
            product_data['discount_percent'] = price_data.get('discount_percent')
        
        # Status and flags
        product_data['status'] = product_list_item.get('status')
        product_data['is_incredible'] = product_list_item.get('is_incredible', False)
        product_data['is_fresh'] = product_list_item.get('is_fresh', False)
        
        # Images
        images_data = product_list_item.get('images', {})
        if isinstance(images_data, dict):
            main_image = images_data.get('main', {})
            product_data['main_image_url'] = main_image.get('url', [None])[0] if isinstance(
                main_image.get('url'), list
            ) else main_image.get('url')
            
            # Store all image URLs
            all_images = []
            for img in images_data.get('list', []):
                if isinstance(img, dict):
                    img_url = img.get('url', [None])[0] if isinstance(
                        img.get('url'), list
                    ) else img.get('url')
                    if img_url:
                        all_images.append(img_url)
            product_data['images'] = all_images if all_images else None
        
        # Rating
        rating_data = product_list_item.get('rating', {})
        if rating_data:
            product_data['rating_average'] = rating_data.get('rate')
            product_data['rating_count'] = rating_data.get('count')
        
        # Availability
        product_data['is_available'] = product_list_item.get('is_available', True)
        
        # If we have detailed data, extract specifications and description
        if product_detail and 'data' in product_detail:
            detail_data = product_detail['data'].get('product', {})
            specifications_raw = detail_data.get('specifications', [])
            product_data['specifications'] = self.filter_specifications(specifications_raw)
            
            # Extract description from review
            review_data = detail_data.get('review', {})
            if review_data and isinstance(review_data, dict):
                description = review_data.get('description')
                if description:
                    product_data['description'] = description
            
            # Update other fields from detail if available
            if not product_data.get('brand') and detail_data.get('brand'):
                brand = detail_data['brand']
                product_data['brand'] = brand.get('title_fa') or brand.get('title')
        
        return product_data
    
    def save_or_update_product(self, product_data: Dict, category: Optional[Category] = None) -> Optional[Product]:
        """Save or update product in database"""
        if not product_data or not product_data.get('digikala_id'):
            return None
        
        try:
            # Check if product exists
            product = self.session.query(Product).filter_by(
                digikala_id=product_data['digikala_id']
            ).first()
            
            if product:
                # Update existing product
                for key, value in product_data.items():
                    if hasattr(product, key):
                        setattr(product, key, value)
                if category:
                    product.category = category
                logger.info(f"Updated product: {product_data.get('title', '')[:50]}")
            else:
                # Create new product
                if category:
                    product_data['category_id'] = category.id
                product = Product(**product_data)
                self.session.add(product)
                logger.info(f"Created new product: {product_data.get('title', '')[:50]}")
            
            self.session.commit()
            return product
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to save product: {e}")
            self.session.rollback()
            return None
    
    def seed_category(self, category_slug: str, max_pages: Optional[int] = None, 
                     fetch_details: bool = True, delay_between_requests: float = 0.5):
        """
        Seed products from a specific category
        
        Args:
            category_slug: Category slug (e.g., 'mobile-phone')
            max_pages: Maximum number of pages to process (None for all)
            fetch_details: Whether to fetch detailed product information
            delay_between_requests: Delay in seconds between API requests
        """
        logger.info(f"Starting to seed category: {category_slug}")
        
        # Get database session
        self.session = db.get_session()
        
        try:
            # Fetch first page to get category info
            first_page = self.api_client.get_product_list(category_slug, page=1)
            if not first_page or 'data' not in first_page:
                logger.error("Failed to fetch first page")
                return
            
            # Extract and save category
            category_data_raw = first_page.get('data', {}).get('category', {})
            category_data = self.extract_category_data(category_data_raw)
            category = self.save_or_update_category(category_data)
            
            if not category:
                logger.error("Failed to save category")
                return
            
            # Fetch all products from all pages
            all_products = self.api_client.get_all_pages(category_slug, max_pages)
            logger.info(f"Found {len(all_products)} products to process")
            
            # Process each product
            for idx, product_item in enumerate(all_products, 1):
                product_id = product_item.get('id')
                logger.info(f"Processing product {idx}/{len(all_products)}: {product_id}")
                
                product_detail = None
                if fetch_details and product_id:
                    time.sleep(delay_between_requests)  # Rate limiting
                    product_detail = self.api_client.get_product_detail(product_id)
                
                # Extract and save product
                product_data = self.extract_product_data(product_item, product_detail)
                self.save_or_update_product(product_data, category)
            
            logger.info(f"Successfully seeded {len(all_products)} products from category {category_slug}")
            
        except Exception as e:
            logger.error(f"Error during seeding: {e}")
        finally:
            if self.session:
                self.session.close()
    
    def close(self):
        """Close API client and clean up"""
        self.api_client.close()

