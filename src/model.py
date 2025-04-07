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

# def evaluate_model(
#     model: Sequential, 
#     X_test: np.ndarray, 
#     y_test: np.ndarray
# ) -> Dict[str, Any]:
#     """
#     Evaluates model performance with strict type checking
#     """
#     # Convert and validate inputs
#     X_test = np.asarray(X_test, dtype=np.float32)
#     y_test = np.asarray(y_test, dtype=np.int32)
    
#     # Validate shapes and types
#     if len(X_test.shape) != 2:
#         raise ValueError(f"X_test must be 2D array, got shape {X_test.shape}")
#     if len(y_test.shape) != 1:
#         raise ValueError(f"y_test must be 1D array, got shape {y_test.shape}")
    
#     # Get predictions
#     y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
#     # Calculate metrics
#     cm = confusion_matrix(y_test, y_pred)
    
#     metrics = {
#         'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
#         'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
#         'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
#         'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
#         'confusion_matrix': {
#             'true_negative': int(cm[0, 0]),
#             'false_positive': int(cm[0, 1]),
#             'false_negative': int(cm[1, 0]),
#             'true_positive': int(cm[1, 1])
#         }
#     }
    
#     logging.info("\nModel Evaluation:")
#     logging.info(f"Accuracy: {metrics['accuracy']}")
#     logging.info(f"Precision: {metrics['precision']}")
#     logging.info(f"Recall: {metrics['recall']}")
#     logging.info(f"F1 Score: {metrics['f1']}")
#     logging.info(f"Confusion Matrix:\n{cm}")
    
#     return metrics

def evaluate_model(
    model: Sequential, 
    X_test: np.ndarray, 
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates model performance with strict type checking
    """
    try:
        # Convert and validate inputs
        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.int32)
        
        # Validate shapes and types
        if len(X_test.shape) != 2:
            raise ValueError(f"X_test must be 2D array, got shape {X_test.shape}")
        if len(y_test.shape) != 1:
            raise ValueError(f"y_test must be 1D array, got shape {y_test.shape}")
        
        # Get predictions
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        
        # Handle multi-class metrics
        num_classes = len(np.unique(y_test))
        if num_classes == 2:  # Binary classification
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:  # Multi-class classification
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4),
            'confusion_matrix': cm.tolist()
        }
        
        logging.info("\nModel Evaluation:")
        logging.info(f"Accuracy: {metrics['accuracy']}")
        logging.info(f"Precision: {metrics['precision']}")
        logging.info(f"Recall: {metrics['recall']}")
        logging.info(f"F1 Score: {metrics['f1']}")
        logging.info(f"Confusion Matrix:\n{cm}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise ValueError(f"Model evaluation failed: {str(e)}")
# End

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