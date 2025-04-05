from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os
from dotenv import load_dotenv
import urllib.parse

# Load environment variables
load_dotenv()

def get_database_url():
    # Get credentials from environment
    db_user = os.getenv("DB_USER", "weatherwise_user")
    db_password = urllib.parse.quote_plus(os.getenv("DB_PASSWORD", ""))
    db_host = os.getenv("DB_HOST", "dpg-cvm3v5re5dus73aevum0-a.oregon-postgres.render.com")
    db_name = os.getenv("DB_NAME", "weatherwise")
    
    return f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}?sslmode=require"

DATABASE_URL = get_database_url()

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30
    }
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables initialized")
    except Exception as e:
        print(f"Warning: Could not create tables on startup: {e}")

init_db()