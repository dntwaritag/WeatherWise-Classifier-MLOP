# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from urllib.parse import quote_plus
from contextlib import contextmanager
from typing import Generator
import time

DB_USER = "seattle_weather_user"
DB_PASSWORD = "hS1yirAgqJqyY6Kd6KMgwuKkTEhZInQn"
DB_HOST = "dpg-cvmr6sp5pdvs73flesc0-a.oregon-postgres.render.com"
DB_NAME = "seattle_weather"

encoded_password = quote_plus(DB_PASSWORD)
DATABASE_URL = f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}?sslmode=require"

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        # Test the connection first
        db.execute("SELECT 1")
        yield db
    except Exception as e:
        db.close()
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )
    finally:
        db.close()

def initialize_database():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)