from sqlalchemy import Column, Integer, Float, String, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class WeatherData(Base):
    """
    SQLAlchemy model for storing raw weather observations with MLOP optimizations
    """
    __tablename__ = 'weather_data'
    
    # Changed to String UUID primary key for distributed systems
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    date = Column(DateTime, nullable=False, index=True)
    precipitation = Column(Float, nullable=False)
    temp_max = Column(Float, nullable=False)
    temp_min = Column(Float, nullable=False)
    wind = Column(Float, nullable=False)
    weather = Column(String(20), nullable=False)
    
    # Only include recorded_at if it exists in your database
    # recorded_at = Column(DateTime, default=datetime.utcnow)  # Commented out if column doesn't exist
    
    __table_args__ = (
        Index('idx_weather_date', 'weather', 'date'),
        Index('idx_temp_range', 'temp_min', 'temp_max'),
    )

class ModelMetadata(Base):
    """
    MLOP model versioning and metadata tracking
    """
    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50), unique=True, nullable=False)
    model_type = Column(String(20), nullable=False)
    training_date = Column(DateTime, default=datetime.utcnow)
    metrics = Column(String(1000))
    feature_set_version = Column(String(20), nullable=False)
    trained_on_records = Column(Integer, nullable=False)
    artifact_path = Column(String(200), nullable=False)

class PredictionLog(Base):
    """
    MLOP prediction auditing and monitoring
    """
    __tablename__ = 'prediction_logs'
    
    id = Column(Integer, primary_key=True)
    input_data = Column(String(500), nullable=False)
    prediction = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50), nullable=False)
    execution_time = Column(Float)