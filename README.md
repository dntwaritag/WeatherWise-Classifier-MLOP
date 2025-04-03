# WeatherWise-Classifier MLOP Project

![Weather Classification](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)
![MLOP](https://img.shields.io/badge/Approach-MLOP-orange)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [API Documentation](#api-documentation)
- [Data Formats](#data-formats)
- [Deployment](#deployment)
- [Technical Stack](#technical-stack)
- [License](#license)
- [Contact](#contact)

## Project Overview

A production-ready weather classification system implementing MLOP best practices. The system:

- Classifies weather conditions (rain, sun, fog, etc.) from meteorological data
- Supports continuous model improvement through automated retraining
- Provides comprehensive monitoring of model performance
- Offers RESTful API endpoints for integration

**Live Demo**: [https://weatherwise-classifier.onrender.com](https://weatherwise-classifier.onrender.com)

## Features

- **Real-time Prediction**: Classify weather conditions from input features
- **Bulk Processing**: Upload CSV datasets for batch predictions
- **Model Management**: 
  - Version tracking
  - Performance metrics storage
  - Retraining triggers
- **Monitoring Dashboard**: Track model drift and data quality
- **CI/CD Ready**: Dockerized with automated testing

## Project Structure

```text
WeatherWise-Classifier-MLOP/
├── .github/                  # GitHub workflows
├── data/                     # Sample datasets
│   ├── sample_weather.csv    # Example weather data
├── models/                   # Trained model artifacts
├── notebooks/                # Exploration notebooks
├── src/
│   ├── app/                  # FastAPI application
│   │   ├── __init__.py
│   │   ├── database.py       # DB configuration
│   │   ├── main.py           # API endpoints
│   │   ├── models.py         # SQLAlchemy models
│   ├── config/               # Configuration files
│   ├── model/                # ML model code
│   ├── preprocessing/        # Data processing
│   ├── prediction/           # Inference logic
├── tests/                    # Unit and integration tests
├── .env.example              # Environment template
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Multi-container setup
├── requirements.txt          # Python dependencies
├── Makefile                  # Development commands
```

## Getting Started

### Prerequisites

- Python 3.8+
- Docker 20.10+
- PostgreSQL 12+
- Git 2.25+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/WeatherWise-Classifier-MLOP.git
   cd WeatherWise-Classifier-MLOP
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Initialize database**
   ```bash
   python src/app/database.py
   ```

6. **Run the application**
   ```bash
   uvicorn src.app.main:app --reload
   ```

## API Documentation

Interactive documentation available at `http://localhost:8000/docs`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload-weather-data/` | POST | Upload CSV weather data |
| `/predict-weather/` | POST | Single prediction |
| `/batch-predict/` | POST | Bulk predictions |
| `/retrain-model/` | POST | Trigger retraining |
| `/model-metrics/` | GET | Performance metrics |
| `/healthcheck` | GET | Service status |

## Data Formats

### Single Prediction Request
```json
{
  "date": "2023-01-01T00:00:00",
  "precipitation": 0.5,
  "temp_max": 10.0,
  "temp_min": 5.0,
  "wind": 15.0
}
```

### Bulk Upload CSV Format
```csv
date,precipitation,temp_max,temp_min,wind,weather
2023-01-01,0.5,10.0,5.0,15.0,rain
2023-01-02,0.0,15.0,8.0,10.0,sun
```

## Deployment

### Docker Setup
```bash
docker-compose up --build
```

## Technical Stack

**Backend**
- FastAPI
- Uvicorn
- SQLAlchemy
- Alembic (migrations)

**Machine Learning**
- Scikit-learn
- XGBoost
- Joblib

**Data Processing**
- Pandas
- NumPy

**Infrastructure**
- Docker
- PostgreSQL
- Redis (caching)

**Monitoring**
- Prometheus
- Grafana

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For support or questions:
- Email: dntwaritag@alustudent.com
- GitHub: [@dntwaritag](https://github.com/dntwaritag/WeatherWise-Classifier-MLOP/)
```
