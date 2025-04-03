import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, Any, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_DIR = "models"

class ModelTrainer:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """Train Logistic Regression with GridSearchCV"""
        logger.info("Training Logistic Regression model")
        
        param_grid = [
            {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
             'solver': ['lbfgs'], 
             'penalty': ['l2'], 
             'max_iter': [100, 200, 300]},
            {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
             'solver': ['liblinear', 'saga'], 
             'penalty': ['l1', 'l2'], 
             'max_iter': [100, 200, 300]}
        ]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            LogisticRegression(random_state=42, multi_class='ovr'),
            param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        return self.model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest with RandomizedSearchCV"""
        logger.info("Training Random Forest model")
        
        param_grid = {
            'n_estimators': [50, 150, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'criterion': ['gini', 'entropy']
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions=param_grid,
            n_iter=10,
            cv=cv,
            scoring='accuracy',
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        logger.info(f"Best parameters: {random_search.best_params_}")
        return self.model
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray) -> Sequential:
        """Train Neural Network with Early Stopping"""
        logger.info("Training Neural Network model")
        
        n_classes = len(np.unique(y_train))
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.model = model
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = self.model.predict(X_test)
        
        if self.model_type == 'neural_network':
            y_pred = np.argmax(y_pred, axis=1)
            
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Handle ROC AUC for multiclass
        try:
            if len(np.unique(y_test)) > 2:
                y_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else y_pred
                metrics['roc_auc_ovo'] = roc_auc_score(y_test, y_proba, multi_class='ovo')
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        logger.info("\nModel Evaluation:")
        for k, v in metrics.items():
            if k != 'confusion_matrix' and k != 'classification_report':
                logger.info(f"{k}: {v:.4f}")
                
        return metrics

    def save_model(self, model_name: str) -> None:
        """Save model to appropriate format"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if self.model_type == 'neural_network':
            save_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
            save_model(self.model, save_path)
        else:
            save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
            joblib.dump(self.model, save_path)
            
        logger.info(f"Model saved to {save_path}")

# Add these top-level functions here
def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[LogisticRegression, dict]:
    """Top-level wrapper for logistic regression training"""
    trainer = ModelTrainer('logistic_regression')
    model = trainer.train_logistic_regression(X_train, y_train)
    metrics = trainer.evaluate_model(X_train, y_train)
    return model, metrics

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[RandomForestClassifier, dict]:
    """Top-level wrapper for random forest training"""
    trainer = ModelTrainer('random_forest')
    model = trainer.train_random_forest(X_train, y_train)
    metrics = trainer.evaluate_model(X_train, y_train)
    return model, metrics

def train_neural_network(X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Sequential, dict]:
    """Top-level wrapper for neural network training"""
    trainer = ModelTrainer('neural_network')
    model = trainer.train_neural_network(X_train, y_train, X_val, y_val)
    metrics = trainer.evaluate_model(X_val, y_val)
    return model, metrics

def save_model(model: Union[LogisticRegression, RandomForestClassifier, Sequential], 
              model_name: str) -> None:
    """Top-level model saving function"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if isinstance(model, Sequential):
        save_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
        save_model(model, save_path)
    else:
        save_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, save_path)
    
    logger.info(f"Model saved to {save_path}")

def load_saved_model(model_name: str) -> Union[LogisticRegression, RandomForestClassifier, Sequential]:
    """Load a saved model from disk"""
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if model_name.endswith('.h5'):
        return load_model(model_path)
    elif model_name.endswith('.pkl'):
        return joblib.load(model_path)
    else:
        raise ValueError("Unsupported model format")

if __name__ == "__main__":
    # Example usage
    from preprocessing import full_preprocessing_pipeline
    
    try:
        logger.info("Starting training pipeline")
        
        # Preprocess data
        X_train_scaled, X_test_scaled = full_preprocessing_pipeline("/content/seattle-weather.csv")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_scaled[0], X_train_scaled[1],  # Adjust based on your data structure
            test_size=0.2,
            random_state=42,
            stratify=X_train_scaled[1]
        )
        
        # Train Random Forest
        trainer = ModelTrainer('random_forest')
        trainer.train_random_forest(X_train, y_train)
        metrics = trainer.evaluate_model(X_test_scaled, y_test)
        trainer.save_model("random_forest_model.pkl")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise