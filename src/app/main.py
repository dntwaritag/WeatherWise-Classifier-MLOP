from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os
import io
import sys
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, DataError
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import logging
from typing import List, Dict, Optional
from sklearn.model_selection import train_test_split

# Local imports
from .database import get_db
from .models import WeatherData
from src.preprocessing import preprocess_data, generate_lag_features
from src.model import train_logistic_regression, train_random_forest, train_neural_network
from src.prediction import predict_weather, load_latest_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("weather_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
EXPECTED_COLUMNS = ["date", "precipitation", "temp_max", "temp_min", "wind", "weather"]
VALID_WEATHERS = {'rain', 'sun', 'fog', 'drizzle', 'snow', 'storm'}
MODEL_DIR = Path("models")
os.makedirs(MODEL_DIR, exist_ok=True)
MAX_FILE_SIZE = 1024 * 1024 * 10  # 10MB
trained_models_cache = {}

# Pydantic Models
class WeatherInput(BaseModel):
    precipitation: float
    temp_max: float
    temp_min: float
    wind: float
    date: datetime

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: dict

class RetrainResponse(BaseModel):
    best_model_type: str
    metrics: ModelMetrics
    model_version: str
    training_samples: int
    test_samples: int

class UploadResponse(BaseModel):
    message: str
    invalid_entries: int
    duplicates_skipped: int = 0
    records_inserted: int

# Helper Functions
def fetch_historical_data(db: Session, days: int = 7) -> List[WeatherData]:
    cutoff_date = datetime.now() - timedelta(days=days)
    return db.query(WeatherData).filter(WeatherData.date >= cutoff_date).all()

def validate_weather_data(df: pd.DataFrame) -> None:
    """Validate weather data DataFrame"""
    # Check required columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Validate numeric columns
    numeric_cols = ["precipitation", "temp_max", "temp_min", "wind"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} contains non-numeric values")
        if df[col].isnull().any():
            raise ValueError(f"Column {col} contains missing values")

    # Validate date format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        invalid_dates = df[df['date'].isnull()]['date'].tolist()
        raise ValueError(f"Invalid date values detected: {invalid_dates[:5]}")

    # Validate weather categories
    invalid_entries = df[~df['weather'].isin(VALID_WEATHERS)]
    if not invalid_entries.empty:
        invalid_values = invalid_entries['weather'].unique().tolist()
        raise ValueError(f"Invalid weather values: {invalid_values}")

# API Endpoints
@app.post("/upload-weather-data/", response_model=UploadResponse)
async def upload_weather_data(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload weather data to database"""
    try:
        # Validate file
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(400, "Only CSV files are supported")

        # Read and parse file
        try:
            contents = await file.read()
            if len(contents) == 0:
                raise HTTPException(400, "Empty file uploaded")
                
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

        # Validate data
        try:
            validate_weather_data(df)
        except ValueError as e:
            raise HTTPException(400, str(e))

        # Process records
        inserted_count = 0
        duplicate_count = 0
        invalid_count = 0
        
        for record in df.to_dict('records'):
            try:
                # Prepare record
                if isinstance(record['date'], str):
                    try:
                        record['date'] = pd.to_datetime(record['date']).to_pydatetime()
                    except ValueError:
                        invalid_count += 1
                        continue
                
                # Insert record
                try:
                    weather_data = WeatherData(**record)
                    db.add(weather_data)
                    db.commit()
                    inserted_count += 1
                except IntegrityError:
                    db.rollback()
                    duplicate_count += 1
                except (DataError, ValueError) as e:
                    db.rollback()
                    invalid_count += 1
                    logger.warning(f"Invalid record: {record} - Error: {str(e)}")
            except Exception as e:
                db.rollback()
                logger.error(f"Unexpected error processing record: {str(e)}")
                invalid_count += 1

        if inserted_count == 0 and invalid_count == 0 and duplicate_count == 0:
            raise HTTPException(400, "No valid records found in file")

        return UploadResponse(
            message=f"Processed {len(df)} records",
            records_inserted=inserted_count,
            invalid_entries=invalid_count,
            duplicates_skipped=duplicate_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/predict-weather/")
async def predict_weather_endpoint(input_data: WeatherInput):
    try:
        # Load necessary artifacts
        model, scaler, encoder = load_latest_model()
        
        # Create DataFrame with historical context
        input_df = pd.DataFrame([input_data.dict()])
        input_df = generate_lag_features(input_df)
        
        # Preprocess and predict
        processed_input = preprocess_data(input_df, scaler=scaler)
        prediction = predict_weather(processed_input, model, encoder)
        
        return {
            "prediction": prediction,
            "model_version": model.metadata['version'],
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate weather prediction"
        )

@app.post("/retrain-model/", response_model=RetrainResponse)
async def retrain_model(db: Session = Depends(get_db)):
    """Retrain weather classification models"""
    try:
        # Fetch and prepare data
        records = fetch_historical_data(db)
        if len(records) < 100:
            raise HTTPException(
                status_code=400,
                detail="Minimum 100 records required for retraining"
            )

        df = pd.DataFrame([{
            "date": r.date,
            "precipitation": r.precipitation,
            "temp_max": r.temp_max,
            "temp_min": r.temp_min,
            "wind": r.wind,
            "weather": r.weather
        } for r in records])

        # Preprocess data
        processed_df = preprocess_data(df)
        processed_df = generate_lag_features(processed_df)
        
        # Split data
        train_df, test_df = train_test_split(
            processed_df,
            test_size=0.2,
            stratify=processed_df['weather_encoded'],
            random_state=42
        )

        # Train models
        logreg_model, logreg_metrics = train_logistic_regression(train_df, test_df)
        rf_model, rf_metrics = train_random_forest(train_df, test_df)
        nn_model, nn_metrics = train_neural_network(train_df, test_df)

        # Select best model
        models = {
            "logistic_regression": (logreg_model, logreg_metrics),
            "random_forest": (rf_model, rf_metrics),
            "neural_network": (nn_model, nn_metrics)
        }
        best_model_type = max(models, key=lambda k: models[k][1]['accuracy'])
        best_model, best_metrics = models[best_model_type]

        # Version and save model
        model_version = f"{best_model_type}_{datetime.now().strftime('%Y%m%d%H%M')}"
        best_model.metadata = {
            "version": model_version,
            "training_date": datetime.now(),
            "metrics": best_metrics,
            "model_type": best_model_type
        }
        joblib.dump(best_model, MODEL_DIR / f"{model_version}.pkl")

        return RetrainResponse(
            best_model_type=best_model_type,
            metrics=ModelMetrics(**best_metrics),
            model_version=model_version,
            training_samples=len(train_df),
            test_samples=len(test_df)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Model retraining failed"
        )

@app.get("/model-performance/")
async def get_model_performance():
    """Get performance metrics for all trained models"""
    try:
        models = []
        for model_file in MODEL_DIR.glob("*.pkl"):
            try:
                model = joblib.load(model_file)
                if hasattr(model, 'metadata'):
                    models.append({
                        "version": model.metadata.get('version'),
                        "type": model.metadata.get('model_type'),
                        "accuracy": model.metadata.get('metrics', {}).get('accuracy'),
                        "training_date": model.metadata.get('training_date')
                    })
            except Exception as e:
                logger.warning(f"Failed to load model {model_file}: {str(e)}")
                continue
                
        return {"models": sorted(models, key=lambda x: x['training_date'], reverse=True)}
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model performance data"
        )

@app.get("/")
async def root():
    return {"message": "Weather Classification API", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)