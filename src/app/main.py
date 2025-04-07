from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
import uvicorn
import os
import pickle
from fastapi.responses import FileResponse
import sys
from sqlalchemy.orm import Session
from pathlib import Path
import uuid
from typing import List
from pydantic import BaseModel
# start
#from main import ConfusionMatrix, MetricDetail
# End
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
logging.getLogger("uvicorn.access").addFilter(lambda record: "/favicon.ico" not in record.getMessage())

# Local imports
from .database import get_db
from .models import WeatherData
from src.preprocessing import preprocess_data, load_prediction_data, fetch_training_data
from src.model import save_model, evaluate_model
from src.prediction import predict
from src.preprocessing import VALID_WEATHER_TYPES

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
    matrix: List[List[int]]  # 2D array for multi-class
    labels: List[str]  # Class names

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
    model_path: str

class BulkUploadResponse(BaseModel):
    success: bool
    message: str
    records_added: int
    invalid_records: int
class SaveRequest(BaseModel):
    model_id: str

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

def cleanup_old_models():
    now = datetime.now()
    for model_id, data in list(trained_models_cache.items()):
        if data['expires'] < now:
            del trained_models_cache[model_id]
            try:
                os.remove(f"models/model_{model_id}.h5")
            except FileNotFoundError:
                pass

# API Endpoints
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")

# Update your favicon endpoint
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse('src/app/static/favicon.ico')

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
    logger.info("Starting model retraining process")
    
    try:
        # 1. Fetch data from database
        df = fetch_training_data(db)
        
        if len(df) < 100:
            raise HTTPException(400, f"At least 100 records required, got {len(df)}")

        # 2. Preprocess the data with strict type enforcement
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Additional validation
        logger.info("\n=== DATA VALIDATION ===")
        logger.info(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
        logger.info(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
        logger.info(f"Unique y_train values: {np.unique(y_train)}")
        logger.info(f"Number of classes: {len(np.unique(y_train))}")
        logger.info(f"X_train min/max: {X_train.min()}, {X_train.max()}")
        logger.info(f"y_train min/max: {y_train.min()}, {y_train.max()}")
        
        # 3. Train new model with enhanced error handling
        try:
            num_classes = len(np.unique(y_train))
            
            # Clear any existing TensorFlow session
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
            # Simplified model architecture
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_initializer='he_normal'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu',
                     kernel_initializer='he_normal'),
                BatchNormalization(),
                Dense(num_classes, activation='softmax')
            ])
            
            # Use lower learning rate for stability
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
            
            # Train with explicit batch size and validation
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )

            # Generate model_id AFTER successful training
            model_id = str(uuid.uuid4())
            logger.info(f"Generated model ID: {model_id}")
            
            # Store in cache AFTER training completes
            trained_models_cache[model_id] = {
                'model': model,
                'expires': datetime.now() + timedelta(hours=24)
            }

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Model training failed: {str(e)}")

        # 4. Evaluate and save model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Save model with checks
        try:
            model_path = f"models/model_{model_id}.h5"
            model.save(model_path)
            logger.info(f"Model saved successfully to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise HTTPException(500, f"Model saving failed: {str(e)}")
        
        return RetrainResponse(
            metrics=MetricDetail(
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                confusion_matrix=ConfusionMatrix(
                    matrix=metrics["confusion_matrix"],
                    labels=list(VALID_WEATHER_TYPES.keys())
                )
            ),
            model_id=model_id,  # Use the same model_id we generated
            message="Model retrained successfully",
            training_samples=len(X_train),
            test_samples=len(X_test)
        )
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Retraining failed: {str(e)}")
    

@app.post("/save-model/", response_model=SaveResponse)
async def save_model_endpoint(request: SaveRequest, db: Session = Depends(get_db)):
    """Save a trained model from cache to persistent storage"""
    logger.info("Starting model save operation")
    
    try:
        model_id = request.model_id
        if not model_id:
            raise HTTPException(400, "Missing model_id")

        # Check cache
        model_data = trained_models_cache.get(model_id)
        if not model_data:
            logger.error(f"Model not found in cache: {model_id}")
            raise HTTPException(404, "Model not found or expired")

        # Get model from cache
        model = model_data['model']
        
        # Save to persistent storage
        model_path = f"models/model_{model_id}.h5"
        model.save(model_path)
        
        # Remove from cache
        del trained_models_cache[model_id]
        
        logger.info(f"Successfully saved model to {model_path}")
        return SaveResponse(
            success=True,
            message="Model saved successfully",
            model_path=model_path
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