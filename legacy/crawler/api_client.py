import requests
import logging
from typing import Optional, Dict, List
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigikalaAPIClient:
    """Client for interacting with Digikala API"""
    
    def __init__(self):
        self.base_url = Config.DIGIKALA_BASE_URL
        self.timeout = Config.API_REQUEST_TIMEOUT
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
    
    def get_product_list(self, category: str, page: int = 1, rch: str = 'db340a7f7c4f') -> Optional[Dict]:
        """
        Fetch list of products from a category
        
        Args:
            category: Category slug (e.g., 'mobile-phone')
            page: Page number
            rch: Request channel identifier
        
        Returns:
            Dictionary containing product list data or None if failed
        """
        url = f"{self.base_url}/v1/categories/{category}/search/"
        params = {
            'page': page,
            '_rch': rch
        }
        
        try:
            logger.info(f"Fetching product list for category '{category}', page {page}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 200 or 'data' in data:
                logger.info(f"Successfully fetched product list (page {page})")
                return data
            else:
                logger.warning(f"Unexpected response format: {data}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch product list: {e}")
            return None
    
    def get_product_detail(self, product_id: int, variant_id: Optional[int] = None, 
                          rch: str = 'db340a7f7c4f') -> Optional[Dict]:
        """
        Fetch detailed information for a specific product
        
        Args:
            product_id: Product ID
            variant_id: Variant ID (optional)
            rch: Request channel identifier
        
        Returns:
            Dictionary containing product detail data or None if failed
        """
        url = f"{self.base_url}/v2/product/{product_id}/"
        params = {'_rch': rch}
        
        if variant_id:
            params['variant_id'] = variant_id
        
        try:
            logger.info(f"Fetching product detail for product_id {product_id}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 200 or 'data' in data:
                logger.info(f"Successfully fetched product detail for product_id {product_id}")
                return data
            else:
                logger.warning(f"Unexpected response format for product {product_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch product detail for {product_id}: {e}")
            return None
    
    def get_all_pages(self, category: str, max_pages: Optional[int] = None) -> List[Dict]:
        """
        Fetch all pages of products from a category
        
        Args:
            category: Category slug
            max_pages: Maximum number of pages to fetch (None for all)
        
        Returns:
            List of product data dictionaries
        """
        all_products = []
        page = 1
        
        while True:
            if max_pages and page > max_pages:
                break
            
            result = self.get_product_list(category, page)
            if not result or 'data' not in result:
                break
            
            products = result.get('data', {}).get('products', [])
            if not products:
                logger.info(f"No more products found at page {page}")
                break
            
            all_products.extend(products)
            logger.info(f"Collected {len(products)} products from page {page}")
            
            # Check if there are more pages
            pagination = result.get('data', {}).get('pager', {})
            current_page = pagination.get('current_page', page)
            total_pages = pagination.get('total_pages', 0)
            
            logger.info(f"Page {current_page}/{total_pages}")
            
            if current_page >= total_pages:
                logger.info(f"Reached last page ({current_page}/{total_pages})")
                break
            
            page += 1
        
        logger.info(f"Total products collected: {len(all_products)}")
        return all_products
    
    def close(self):
        """Close the session"""
        self.session.close()

