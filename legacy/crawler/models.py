from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Category(Base):
    """Category model for storing product categories"""
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    digikala_id = Column(Integer, unique=True, nullable=False, index=True)
    title = Column(String(255), nullable=False)
    title_en = Column(String(255))
    code = Column(String(100), unique=True, index=True)
    url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    products = relationship("Product", back_populates="category")
    
    def __repr__(self):
        return f"<Category(id={self.id}, title='{self.title}')>"


class Product(Base):
    """Product model for storing product details"""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    digikala_id = Column(Integer, unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    title_en = Column(String(500))
    url = Column(String(500))
    brand = Column(String(255))
    
    # Category relationship
    category_id = Column(Integer, ForeignKey('categories.id'))
    category = relationship("Category", back_populates="products")
    
    # Pricing
    price = Column(Float)
    selling_price = Column(Float)
    discount_percent = Column(Integer)
    
    # Product details
    status = Column(String(50))
    description = Column(Text)  # Product description
    is_incredible = Column(Boolean, default=False)
    is_fresh = Column(Boolean, default=False)
    
    # Images
    main_image_url = Column(String(500))
    images = Column(JSON)  # Store multiple images as JSON array
    
    # Ratings
    rating_average = Column(Float)
    rating_count = Column(Integer)
    
    # Stock
    is_available = Column(Boolean, default=True)
    
    # Specifications - store filtered meaningful specs as JSON
    specifications = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Product(id={self.id}, title='{self.title}', brand='{self.brand}')>"

