from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os
import pickle
import sys
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import io
import numpy as np
from tensorflow.keras.models import Sequential, load_model as load_keras_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Local imports
from .database import get_db
from .models import WeatherData
from src.preprocessing import preprocess_data, load_prediction_data, fetch_training_data
from src.model import save_model, evaluate_model
from src.prediction import predict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
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

MODEL_PATH = Path("models/neural_network_model.h5").absolute()
os.makedirs(MODEL_PATH.parent, exist_ok=True)
trained_models_cache = {}

class ConfusionMatrix(BaseModel):
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

class MetricDetail(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: ConfusionMatrix

class WeatherInput(BaseModel):
    precipitation: float
    temp_max: float
    temp_min: float
    wind: float
    lag_wind_1: float
    lag_precipitation_1: float
    lag_temp_max_1: float
    lag_temp_min_1: float

class RetrainResponse(BaseModel):
    metrics: MetricDetail
    model_id: str
    message: str
    training_samples: int
    test_samples: int

class SaveResponse(BaseModel):
    success: bool
    message: str

class BulkUploadResponse(BaseModel):
    success: bool
    message: str
    records_added: int
    invalid_records: int

# Helper Functions
def cleanup_expired_models():
    try:
        now = datetime.now()
        expired_ids = [
            model_id for model_id, data in trained_models_cache.items()
            if data['expires'] < now
        ]
        for model_id in expired_ids:
            del trained_models_cache[model_id]
            logger.info(f"Cleaned up expired model: {model_id}")
    except Exception as e:
        logger.error(f"Error cleaning up expired models: {str(e)}", exc_info=True)

# API Endpoints
@app.post("/upload-training-data/", response_model=BulkUploadResponse)
async def upload_training_data(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload training data to database"""
    logger.info("Starting training data upload")
    try:
        # Read file
        file_content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
            logger.debug("Successfully read CSV file")
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_content))
            logger.debug("Successfully read Excel file")
        else:
            logger.error(f"Unsupported file format: {file.filename}")
            raise HTTPException(400, "Unsupported file format")

        # Validate required columns
        required_columns = ['precipitation', 'temp_max', 'temp_min', 'wind', 'weather']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise HTTPException(400, f"Missing required columns: {missing_cols}")

        # Insert records
        records_added = 0
        invalid_records = 0
        
        for _, row in df.iterrows():
            try:
                # Handle date if it exists in the data
                date_value = pd.to_datetime(row['date']).date() if 'date' in df.columns else None
                
                record = WeatherData(
                    date=date_value,
                    precipitation=float(row['precipitation']),
                    temp_max=float(row['temp_max']),
                    temp_min=float(row['temp_min']),
                    wind=float(row['wind']),
                    weather=str(row['weather'])
                )
                db.add(record)
                records_added += 1
            except (ValueError, TypeError) as e:
                invalid_records += 1
                logger.warning(f"Invalid record: {e}")
                continue
            except Exception as e:
                invalid_records += 1
                logger.error(f"Unexpected error processing record: {e}", exc_info=True)
                continue

        db.commit()
        logger.info(f"Successfully added {records_added} records, {invalid_records} invalid records")
        return BulkUploadResponse(
            success=True,
            message=f"Added {records_added} records",
            records_added=records_added,
            invalid_records=invalid_records
        )

    except HTTPException as he:
        logger.error(f"HTTP Exception during upload: {he.detail}")
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Database operation failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Database operation failed: {str(e)}")

@app.post("/predict-single/")
async def predict_single(input_data: WeatherInput):
    logger.info("Starting single prediction")
    try:
        input_df = pd.DataFrame([input_data.dict()], columns=EXPECTED_COLUMNS)
        model = load_keras_model(MODEL_PATH)
        results = predict(input_df, model)
        
        logger.info("Successfully completed single prediction")
        return {
            "prediction": int(results['predictions'][0]),
            "probability": float(results['probabilities'][0])
        }
    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-bulk/")
async def predict_bulk(file: UploadFile = File(...)):
    logger.info("Starting bulk prediction")
    try:
        df = load_prediction_data(file)
        model = load_keras_model(MODEL_PATH)
        results = predict(df, model)
        
        combined_results = [
            {"prediction": int(pred), "probability": float(prob)}
            for pred, prob in zip(results['predictions'], results['probabilities'])
        ]
        
        logger.info(f"Successfully completed bulk prediction for {len(combined_results)} records")
        return {
            "results": combined_results,
            "count": len(combined_results)
        }
    except Exception as e:
        logger.error(f"Error during bulk prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain/", response_model=RetrainResponse)
async def retrain_model(db: Session = Depends(get_db)):
    """
    Endpoint to retrain the model with fresh data from the database.
    Steps:
    1. Fetch data from database
    2. Preprocess the data
    3. Validate data types and ranges
    4. Train a new neural network model
    5. Evaluate the model
    6. Save the new model
    7. Return training metrics
    """
    logger.info("Starting model retraining process")
    
    try:
        # 1. Fetch data from database
        logger.info("Fetching training data from database")
        df = fetch_training_data(db)
        
        if len(df) < 100:
            logger.error(f"Insufficient data for retraining. Only {len(df)} records available.")
            raise HTTPException(
                status_code=400,
                detail=f"At least 100 records required for retraining. Only {len(df)} available."
            )
        
        logger.info(f"Successfully fetched {len(df)} records from database")

        # 2. Preprocess the data
        logger.info("Preprocessing data")
        try:
            X_train, X_test, y_train, y_test = preprocess_data(df)
            logger.info(f"Data preprocessed. Shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Data preprocessing failed: {str(e)}"
            )

        # 3. Validate data types and ranges
        logger.info("Validating data types and ranges")
        try:
            # Ensure proper data types
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_train = y_train.astype(np.int32)
            y_test = y_test.astype(np.int32)

            # Check for extreme values in features
            if (np.abs(X_train) > 1e6).any():
                logger.error("Extreme feature values detected in training data")
                raise ValueError("Feature values too large - check preprocessing")

            # Check label values
            unique_labels = np.unique(y_train)
            if len(unique_labels) > 100:  # Assuming reasonable number of classes
                logger.error(f"Too many unique labels detected: {len(unique_labels)}")
                raise ValueError("Too many unique labels - check label encoding")

            if y_train.max() > len(unique_labels) - 1:
                logger.error(f"Label values out of range. Max: {y_train.max()}, Expected max: {len(unique_labels)-1}")
                raise ValueError("Label values out of expected range")
                
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Data validation failed: {str(e)}"
            )

        # 4. Train new model
        logger.info("Initializing new neural network model")
        try:
            num_classes = len(np.unique(y_train))
            logger.info(f"Number of classes detected: {num_classes}")

            model = Sequential([
                Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            logger.info("Starting model training")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                verbose=1
            )
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Model training failed: {str(e)}"
            )

        # 5. Evaluate model
        logger.info("Evaluating model performance")
        try:
            metrics = evaluate_model(model, X_test, y_test)
            logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']}")
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Model evaluation failed: {str(e)}"
            )

        # 6. Save the new model
        logger.info("Saving retrained model")
        try:
            model.save(MODEL_PATH)
            logger.info(f"Model saved successfully to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Model save failed: {str(e)}"
            )

        # 7. Prepare response
        response = RetrainResponse(
            metrics=MetricDetail(**metrics),
            model_id=str(uuid.uuid4()),
            message="Model retrained successfully",
            training_samples=len(X_train),
            test_samples=len(X_test)
        )
        
        logger.info("Retraining process completed successfully")
        return response

    except HTTPException:
        raise  # Re-raise HTTPExceptions we created
    except Exception as e:
        logger.error(f"Unexpected error during retraining: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during retraining: {str(e)}"
        )

@app.post("/save-model/", response_model=SaveResponse)
async def save_model_endpoint(
    model_data: dict,
    db: Session = Depends(get_db)
):
    """Save a trained model from cache to persistent storage"""
    logger.info("Starting model save operation")
    try:
        model_id = model_data.get('model_id')
        if not model_id:
            logger.error("Missing model_id in save request")
            raise HTTPException(400, "Missing model_id")
        
        model_data = trained_models_cache.get(model_id)
        if not model_data:
            logger.error(f"Model not found in cache: {model_id}")
            raise HTTPException(404, "Model not found or expired")
        
        model_data['model'].save(MODEL_PATH)
        logger.info(f"Successfully saved model to {MODEL_PATH}")
        
        del trained_models_cache[model_id]
        logger.info(f"Removed model {model_id} from cache")
        
        return SaveResponse(
            success=True,
            message="Model saved successfully"
        )
        
    except HTTPException as he:
        logger.error(f"HTTP Exception during model save: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}", exc_info=True)
        raise HTTPException(500, str(e))

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the WeatherWise Prediction API!"}

if __name__ == "__main__":
    logger.info("Starting FastAPI application")
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)