import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = "models"
DATA_DIR = "data"
REQUIRED_COLUMNS = ['precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
LAG_FEATURES = ['lag_wind_1', 'lag_precipitation_1', 'lag_temp_max_1', 'lag_temp_min_1']

def load_data(file_path: str) -> pd.DataFrame:
    """Load and sort weather data with date validation"""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        if 'date' not in df.columns:
            raise ValueError("Dataset must contain 'date' column")
            
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date')
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataset structure and content"""
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df['date'].isnull().any():
        raise ValueError("Date column contains null values")
        
    return df[REQUIRED_COLUMNS]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal lag features"""
    logger.info("Creating lag features")
    try:
        df = df.copy()
        df['lag_wind_1'] = df['wind'].shift(1)
        df['lag_precipitation_1'] = df['precipitation'].shift(1)
        df['lag_temp_max_1'] = df['temp_max'].shift(1)
        df['lag_temp_min_1'] = df['temp_min'].shift(1)
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode weather labels and save encoder"""
    logger.info("Encoding target variable")
    try:
        le = LabelEncoder()
        df['weather_encoded'] = le.fit_transform(df['weather'])
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        return df
        
    except Exception as e:
        logger.error(f"Target encoding failed: {str(e)}")
        raise

def split_and_save_data(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> None:
    """Split data into train/test sets and save to files"""
    logger.info("Splitting and saving data")
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['weather_encoded']
        )
        
        os.makedirs(DATA_DIR, exist_ok=True)
        train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
        
    except Exception as e:
        logger.error(f"Data splitting failed: {str(e)}")
        raise

def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Handle missing values and scale features"""
    logger.info("Preprocessing features")
    try:
        # Imputation
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)
        
        # Save preprocessing artifacts
        joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        
        return X_train_scaled, X_test_scaled
        
    except Exception as e:
        logger.error(f"Feature preprocessing failed: {str(e)}")
        raise

def generate_lag_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Generate lag features for time series data"""
    logger.info("Generating lag features")
    try:
        df = df.sort_values('date').copy()
        
        for i in range(1, window+1):
            df[f'lag_precipitation_{i}'] = df['precipitation'].shift(i)
            df[f'lag_temp_max_{i}'] = df['temp_max'].shift(i)
            df[f'lag_temp_min_{i}'] = df['temp_min'].shift(i)
            df[f'lag_wind_{i}'] = df['wind'].shift(i)
            
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Lag feature generation failed: {str(e)}")
        raise   

def full_preprocessing_pipeline(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Complete preprocessing pipeline"""
    try:
        # Load and validate raw data
        df = load_data(file_path)
        df = validate_data(df)
        
        # Feature engineering
        df = create_features(df)
        
        # Target encoding
        df = encode_target(df)
        
        # Split data
        split_and_save_data(df)
        
        # Load split data
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
        
        # Prepare features
        X_train = train_df.drop(['weather', 'weather_encoded'], axis=1)
        X_test = test_df.drop(['weather', 'weather_encoded'], axis=1)
        
        # Preprocess features
        return preprocess_features(X_train, X_test)
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        X_train_scaled, X_test_scaled = full_preprocessing_pipeline("/content/seattle-weather.csv")
        logger.info("Preprocessing completed successfully")
        logger.info(f"Training data shape: {X_train_scaled.shape}")
        logger.info(f"Test data shape: {X_test_scaled.shape}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")