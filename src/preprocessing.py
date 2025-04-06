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

# Define valid weather types and their expected encoding
VALID_WEATHER_TYPES = {
    'rain': 0,
    'sun': 1,
    'fog': 2,
    'drizzle': 3,
    'snow': 4
}

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
        
        # Validate and encode target variable
        if not all(weather in VALID_WEATHER_TYPES for weather in df_clean['weather']):
            invalid_types = set(df_clean['weather']) - set(VALID_WEATHER_TYPES.keys())
            raise ValueError(f"Invalid weather types found: {invalid_types}")
        
        # Manual encoding to ensure consistent label values
        y = np.array([VALID_WEATHER_TYPES[weather] for weather in df_clean['weather']], dtype=np.int32)
        
        # Validate label range (should be 0-4 based on VALID_WEATHER_TYPES)
        if y.min() < 0 or y.max() >= len(VALID_WEATHER_TYPES):
            raise ValueError(f"Encoded labels out of range (0-{len(VALID_WEATHER_TYPES)-1}): min={y.min()}, max={y.max()}")
        
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
        
        # Clip values to prevent extreme numbers
        X_train_scaled = np.clip(X_train_scaled, -5, 5).astype(np.float32)  # Tighter clipping range
        X_test_scaled = np.clip(X_test_scaled, -5, 5).astype(np.float32)
        
        # Handle class imbalance safely
        class_counts = Counter(y_train)
        if len(class_counts) > 1:
            imbalance_ratio = min(class_counts.values()) / max(class_counts.values())
            if imbalance_ratio < 0.3:
                logging.info(f"Applying Random Over-Sampling (imbalance ratio: {imbalance_ratio:.2f})")
                ros = RandomOverSampler(random_state=42)
                X_train_scaled, y_train = ros.fit_resample(X_train_scaled, y_train)
                y_train = y_train.astype(np.int32)
        
        # Final validation before returning
        if X_train_scaled.dtype != np.float32 or X_test_scaled.dtype != np.float32:
            raise ValueError("Feature arrays must be float32")
        if y_train.dtype != np.int32 or y_test.dtype != np.int32:
            raise ValueError("Label arrays must be int32")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
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
            
        data = []
        invalid_records = 0
        
        for r in records:
            try:
                # Validate weather type before processing
                if r.weather not in VALID_WEATHER_TYPES:
                    invalid_records += 1
                    continue
                    
                data.append({
                    'precipitation': float(r.precipitation),
                    'temp_max': float(r.temp_max),
                    'temp_min': float(r.temp_min),
                    'wind': float(r.wind),
                    'weather': str(r.weather)
                })
            except (ValueError, TypeError) as e:
                invalid_records += 1
                logging.warning(f"Skipping invalid record: {e}")
                continue
        
        if invalid_records > 0:
            logging.warning(f"Skipped {invalid_records} invalid records")
            
        if len(data) == 0:
            raise ValueError("No valid weather records found after filtering")
        
        df = pd.DataFrame(data)
        
        # Validate numeric ranges
        for col, (min_val, max_val) in NUMERIC_RANGES.items():
            if (df[col] < min_val).any() or (df[col] > max_val).any():
                raise ValueError(f"Database contains invalid {col} values outside range ({min_val}-{max_val})")
        
        return df
    except Exception as e:
        logging.error(f"Error fetching training data: {str(e)}", exc_info=True)
        raise