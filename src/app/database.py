from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os
from dotenv import load_dotenv
import urllib.parse
import logging
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_database_url():
    """Construct database URL from environment variables with proper escaping"""
    try:
        db_user = os.getenv("DB_USER", "weatherwise_user")
        db_password = urllib.parse.quote_plus(os.getenv("DB_PASSWORD", ""))
        db_host = os.getenv("DB_HOST", "dpg-cvm3v5re5dus73aevum0-a.oregon-postgres.render.com")
        db_name = os.getenv("DB_NAME", "weatherwise")
        
        return f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}?sslmode=require"
    except Exception as e:
        logger.error(f"Error constructing database URL: {str(e)}")
        raise

DATABASE_URL = get_database_url()

# Configure engine with robust connection pooling and timeout settings
engine = create_engine(
    DATABASE_URL,
    pool_size=10,                     # Increased from 5
    max_overflow=20,                  # Increased from 10
    pool_pre_ping=True,               # Check connection health before use
    pool_recycle=3600,                # Recycle connections every hour
    pool_timeout=30,                  # Wait 30 seconds for a connection
    connect_args={
        "connect_timeout": 10,        # Connection timeout
        "keepalives": 1,              # Enable TCP keepalive
        "keepalives_idle": 30,        # Start sending keepalives after 30s idle
        "keepalives_interval": 10,    # Send keepalives every 10s
        "keepalives_count": 5         # Max failed keepalives before closing
    }
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Helps with long-running sessions
)

def get_db():
    """Dependency to get DB session with proper error handling"""
    db = SessionLocal()
    try:
        yield db
        db.commit()  # Commit if no exceptions occur
    except SQLAlchemyError as e:
        db.rollback()  # Rollback on error
        logger.error(f"Database error occurred: {str(e)}")
        raise
    except Exception as e:
        db.rollback()  # Rollback on non-SQL errors too
        logger.error(f"Unexpected error in database session: {str(e)}")
        raise
    finally:
        try:
            db.close()  # Always try to close the session
        except Exception as e:
            logger.warning(f"Error closing database session: {str(e)}")
            # Connection might already be closed, which is okay

def init_db():
    """Initialize database tables with error handling"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database tables: {str(e)}")
        # Don't raise here to allow app to start even if tables exist

# Initialize database on module import
init_db()