import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import logging
from typing import Tuple
import numpy as np
from fastapi import UploadFile
from io import StringIO
from sqlalchemy.orm import Session
from src.app.models import WeatherData
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Define expected columns with type information
EXPECTED_COLUMNS = [
    "precipitation",
    "temp_max",
    "temp_min",
    "wind",
    "lag_wind_1",
    "lag_precipitation_1",
    "lag_temp_max_1",
    "lag_temp_min_1"
]

# Define numeric columns and their expected ranges
NUMERIC_RANGES = {
    "precipitation": (0, 100),
    "temp_max": (-50, 50),
    "temp_min": (-50, 50),
    "wind": (0, 100)
}

def load_data(file_path: str) -> pd.DataFrame:
    """Loads and validates weather dataset with type checking"""
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation
        required_cols = ['precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Convert numeric columns to float32 with range checking
        for col, (min_val, max_val) in NUMERIC_RANGES.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
                if (df[col] < min_val).any() or (df[col] > max_val).any():
                    raise ValueError(f"Column {col} contains values outside expected range ({min_val}-{max_val})")
        
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses weather data with robust type handling and validation"""
    try:
        # Create lag features with type safety
        lag_features = {
            'lag_wind_1': 'wind',
            'lag_precipitation_1': 'precipitation',
            'lag_temp_max_1': 'temp_max',
            'lag_temp_min_1': 'temp_min'
        }
        
        for new_col, src_col in lag_features.items():
            df[new_col] = df[src_col].shift(1).astype(np.float32)
        
        # Clean data and validate
        df_clean = df.dropna().copy()
        if len(df_clean) == 0:
            raise ValueError("No valid data remaining after cleaning")
        
        # Encode target variable
        le = LabelEncoder()
        y = le.fit_transform(df_clean['weather'])
        
        # Prepare features with type validation
        X = df_clean[EXPECTED_COLUMNS].astype(np.float32)
        
        # Validate numerical stability
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            raise ValueError("Data contains invalid values (inf or nan)")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        # Scale features with numerical stability checks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check for numerical issues after scaling
        if np.any(np.abs(X_train_scaled) > 1e6):
            logging.warning("Extremely large values detected after scaling")
        
        # Handle class imbalance safely
        class_counts = Counter(y_train)
        if len(class_counts) > 1:
            imbalance_ratio = min(class_counts.values()) / max(class_counts.values())
            if imbalance_ratio < 0.3:
                logging.info(f"Applying Random Over-Sampling (imbalance ratio: {imbalance_ratio:.2f})")
                ros = RandomOverSampler(random_state=42)
                X_train_scaled, y_train = ros.fit_resample(X_train_scaled, y_train)
        
        return (
            X_train_scaled.astype(np.float32), 
            X_test_scaled.astype(np.float32), 
            y_train, 
            y_test
        )
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise

def load_prediction_data(upload_file: UploadFile) -> pd.DataFrame:
    """Loads and strictly validates prediction data"""
    try:
        content = upload_file.file.read().decode('utf-8')
        upload_file.file.seek(0)
        
        df = pd.read_csv(StringIO(content))
        
        # Validate columns
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Type conversion and validation
        for col in EXPECTED_COLUMNS:
            if col in NUMERIC_RANGES:
                min_val, max_val = NUMERIC_RANGES[col]
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
                if df[col].isna().any():
                    raise ValueError(f"Column {col} contains invalid numeric values")
                if (df[col] < min_val).any() or (df[col] > max_val).any():
                    raise ValueError(f"Column {col} contains values outside range ({min_val}-{max_val})")
        
        return df[EXPECTED_COLUMNS]
    except Exception as e:
        raise ValueError(f"Failed to process file: {str(e)}")

def fetch_training_data(db: Session) -> pd.DataFrame:
    """Fetches and validates training data from database"""
    try:
        records = db.query(WeatherData).all()
        
        if not records:
            raise ValueError("No training data found in database")
            
        data = [{
            'precipitation': float(r.precipitation),
            'temp_max': float(r.temp_max),
            'temp_min': float(r.temp_min),
            'wind': float(r.wind),
            'weather': str(r.weather)
        } for r in records]
        
        df = pd.DataFrame(data)
        
        # Validate numeric ranges
        for col, (min_val, max_val) in NUMERIC_RANGES.items():
            if (df[col] < min_val).any() or (df[col] > max_val).any():
                raise ValueError(f"Database contains invalid {col} values outside range ({min_val}-{max_val})")
        
        return df
    except Exception as e:
        logging.error(f"Error fetching training data: {str(e)}", exc_info=True)
        raise