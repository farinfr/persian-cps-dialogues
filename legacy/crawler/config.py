import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for the application"""
    
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'digikala_products')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # API Configuration
    DIGIKALA_BASE_URL = os.getenv('DIGIKALA_BASE_URL', 'https://api.digikala.com')
    API_REQUEST_TIMEOUT = int(os.getenv('API_REQUEST_TIMEOUT', '30'))
    
    @classmethod
    def get_database_url(cls):
        """Returns the database connection URL"""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"

