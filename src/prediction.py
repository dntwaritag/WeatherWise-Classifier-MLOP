import os
import pandas as pd
import numpy as np
import logging
from tensorflow.keras.models import load_model as load_keras_model

logging.basicConfig(level=logging.INFO)

def load_model(model_path):
    """Loads the trained Keras model from the specified file path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = load_keras_model(model_path)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def predict(data, model):
    """
    Makes predictions with proper feature validation and returns both 
    class predictions and probabilities.
    """
    # Convert input to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    
    # Get both predictions and probabilities
    predictions = np.argmax(model.predict(data), axis=1)
    probabilities = model.predict(data)[:, 1]  # Probability of class 1 (rain)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities
    }