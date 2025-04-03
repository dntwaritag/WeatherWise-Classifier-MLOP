import os
import joblib
import numpy as np
import pandas as pd
import logging
import glob
from datetime import datetime
from typing import Dict, Union
from tensorflow.keras.models import load_model as load_keras_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherClassifier:
    def __init__(self, model_dir: str = "models"):
        """Initialize the weather classification predictor"""
        self.model_dir = model_dir
        self.loaded = False
        self.required_features = [
            'precipitation', 'temp_max', 'temp_min', 'wind',
            'lag_wind_1', 'lag_precipitation_1', 
            'lag_temp_max_1', 'lag_temp_min_1'
        ]
        
        # Load artifacts on initialization
        self.load_artifacts()

    def load_artifacts(self):
        """Load all required preprocessing artifacts and model"""
        try:
            # Load preprocessing components
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
            self.imputer = joblib.load(os.path.join(self.model_dir, 'imputer.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            
            # Load model (prioritize Random Forest)
            model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.model_type = 'sklearn'
            else:
                model_path = os.path.join(self.model_dir, 'neural_network_model.h5')
                self.model = load_keras_model(model_path)
                self.model_type = 'keras'
            
            self.classes = self.label_encoder.classes_
            self.loaded = True
            logger.info("Successfully loaded all artifacts")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise

    # [Keep all other class methods unchanged] 

def predict_weather(input_data: Union[pd.DataFrame, Dict], model_dir: str = "models") -> Dict:
    """Standalone function for weather prediction"""
    classifier = WeatherClassifier(model_dir)
    return classifier.predict(input_data)

def load_latest_model(model_dir: str = "models") -> Union[LogisticRegression, RandomForestClassifier, Sequential]:
    """Load the most recent model based on filename timestamp"""
    try:
        # Get all model files with timestamp patterns
        model_files = []
        for ext in ['*.pkl', '*.h5']:
            model_files.extend(glob.glob(os.path.join(model_dir, ext)))

        if not model_files:
            raise FileNotFoundError(f"No models found in {model_dir}")

        # Extract timestamps from valid filenames
        valid_models = []
        for f in model_files:
            try:
                # Expected format: {model_type}_YYYYMMDDHHMM.{ext}
                filename = os.path.basename(f)
                parts = filename.split('_')
                timestamp_str = parts[-1].split('.')[0]
                datetime.strptime(timestamp_str, "%Y%m%d%H%M")  # Validate format
                valid_models.append(f)
            except (ValueError, IndexError):
                continue

        if not valid_models:
            raise FileNotFoundError("No models with valid timestamp format")

        # Sort by timestamp descending
        valid_models.sort(reverse=True, key=lambda x: x.split('_')[-1])
        latest_model = valid_models[0]

        # Load the model
        if latest_model.endswith('.pkl'):
            return joblib.load(latest_model)
        else:
            return load_keras_model(latest_model)

    except Exception as e:
        logger.error(f"Failed to load latest model: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    classifier = WeatherClassifier()
    
    sample_input = {
        'precipitation': 0.1,
        'temp_max': 12.5,
        'temp_min': 8.2,
        'wind': 4.3,
        'lag_wind_1': 3.9,
        'lag_precipitation_1': 0.0,
        'lag_temp_max_1': 11.8,
        'lag_temp_min_1': 7.5
    }
    
    try:
        result = classifier.predict(sample_input)
        print("Prediction Result:")
        print(f"Weather: {result['predictions'][0]}")
        print(f"Confidence: {result['confidence'][0]:.2%}")
        print("Class Probabilities:")
        for cls, prob in zip(result['class_labels'], result['probabilities'][0]):
            print(f"- {cls}: {prob:.2%}")
    except Exception as e:
        print(f"Prediction error: {str(e)}")