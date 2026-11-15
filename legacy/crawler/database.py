from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from models import Base
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Database:
    """Database connection and session management"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
    
    def connect(self):
        """Create database engine and session"""
        try:
            database_url = Config.get_database_url()
            self.engine = create_engine(
                database_url,
                pool_pre_ping=True,
                echo=False
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            logger.info("Database connection established successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to create tables: {e}")
            return False
    
    def get_session(self):
        """Get a new database session"""
        if self.SessionLocal is None:
            raise Exception("Database not connected. Call connect() first.")
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global database instance
db = Database()

