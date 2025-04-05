import pickle
import logging
import os
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional
from tensorflow.keras.models import Sequential, save_model as save_keras_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score,
    recall_score,
    confusion_matrix
)

logging.basicConfig(level=logging.INFO)

def evaluate_model(
    model: Sequential, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates model performance and returns comprehensive metrics.
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'confusion_matrix': {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
    }
    
    logging.info("\nModel Evaluation:")
    logging.info(f"Accuracy: {metrics['accuracy']}")
    logging.info(f"Precision: {metrics['precision']}")
    logging.info(f"Recall: {metrics['recall']}")
    logging.info(f"F1 Score: {metrics['f1']}")
    logging.info(f"Confusion Matrix:\n{cm}")
    
    return metrics

def save_model(
    model: Sequential, 
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Saves trained Keras model to disk.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        logging.info(f"Saving model to {filepath}")
        model.save(filepath)
        logging.info("Model saved successfully")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise