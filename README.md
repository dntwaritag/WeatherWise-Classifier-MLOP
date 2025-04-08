---
# WeatherWise-Classifier MLOP Project 🌤️  
**Weather Classification using FastAPI, Docker, and MLOps Best Practices**

## Project Overview  
This project presents a weather classification system leveraging Machine Learning and MLOps principles. It enables users to classify weather conditions (e.g., rain, sun, fog) using meteorological data via a FastAPI backend, supported by model monitoring, versioning, and retraining workflows.

## Accessing the Site  
- **Backend (FastAPI)**: [https://weatherwise-backend-ok73.onrender.com](https://weatherwise-backend-ok73.onrender.com)  
- **Frontend**: [https://weatherwise-frontend.onrender.com](https://weatherwise-frontend.onrender.com)  
- **Demo Video**: *[Insert YouTube demo video link here]*  
- **GitHub Repo**: *[Insert  GitHub repository link here]*  

## Features  
✅ Real-time single and batch weather classification  
📁 Upload CSV datasets for bulk predictions  
📈 Monitoring for model drift and retraining triggers  
📊 Model performance metrics available via endpoints  
🚀 Dockerized for easy deployment and CI/CD compatibility  

## Project Structure  
```
WeatherWise-Classifier-MLOP/
│── data/
│   │── test/
│   │── train/
│── models/
│── notebook/
│   │── WeatherWise_Classifier(Metrix_AI)_MLOP.ipynb
│── src/
│   │── app/
│   │   │── __init__.py
│   │   │── database.py
│   │   │── main.py
│   │   │── models.py
│   │── Front_end/
│   │   │── css/
│   │   │── images/
│   │   │── js/
│   │   │── templates/
│   │   │── __init__.py
│   │── model.py
│   │── prediction.py
│   │── preprocessing.py
│── venv/
│── .env
│── .gitignore
│── docker-compose.yml
│── Dockerfile
│── requirements.txt

```

## Getting Started

### Prerequisites  
- Python 3.8+  
- Docker & Docker Compose  
- PostgreSQL 12+  
- Git

### Installation Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/dntwaritag/WeatherWise-Classifier-MLOP.git
   cd WeatherWise-Classifier-MLOP
   ```

2. **Set Up a Virtual Environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux  
   venv\Scripts\activate         # Windows
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**  
   ```bash
   cp .env.example .env
   # Update .env with your PostgreSQL DB credentials
   ```

5. **Initialize the Database**  
   ```bash
   python src/app/database.py
   ```

6. **Run the Application**  
   ```bash
   uvicorn src.app.main:app --reload
   ```

7. **Run with Docker (Recommended for Prod)**  
   ```bash
   docker-compose up --build
   ```

   App will be live at: `http://localhost:8000`

---

## API Documentation  
Interactive API Docs (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)

### Key Endpoints  

| Endpoint                | Method | Description                      |
|------------------------|--------|----------------------------------|
| /upload-weather-data/  | POST   | Upload CSV file for retraining   |
| /predict-weather/      | POST   | Predict weather from single input|
| /batch-predict/        | POST   | Batch prediction on CSV          |
| /retrain-model/        | POST   | Retrain model with new data      |
| /model-metrics/        | GET    | View model performance           |
| /healthcheck           | GET    | API health check                 |

---

## Data Formats

### Single Prediction Input  
```json
{
  "temp_max": 0,
  "temp_min": 0,
  "wind": 0,
  "lag_wind_1": 0,
  "lag_precipitation_1": 0,
  "lag_temp_max_1": 0,
  "lag_temp_min_1": 0
}

```

### Bulk Upload CSV Format  
```
precipitation,lag_wind_1,lag_precipitation_1,lag_temp_max_1,lag_temp_min_1
0.2,5.0,0.1,30.5,20.1
0.0,3.2,0.0,29.0,21.0

```

---

## Results from Flood Request Simulation  
Refer to the included Jupyter notebook under `/notebooks`:

- All preprocessing steps defined as Python functions  
- Model training and evaluation logic included  
- Prediction and retraining logic modularized  
- Outputs stored as `.pkl` model files

---

## Deployment Package  
- 🌐 Public Backend URL: [https://weatherwise-backend-ok73.onrender.com](https://weatherwise-backend-ok73.onrender.com)  
- 🌐 Public Frontend URL: [https://weatherwise-frontend.onrender.com](https://weatherwise-frontend.onrender.com)  
- 📦 Docker Image: Defined in `Dockerfile`  
- 📱 Optional: Extend to mobile/desktop using Flutter or Electron

---

## Technical Stack  

**Backend:**  
- FastAPI  
- Uvicorn  
- SQLAlchemy  
- Alembic  

**ML/Modeling:**  
- Scikit-learn  
- XGBoost  
- Joblib  

**Data Processing:**  
- Pandas  
- NumPy  

**Infrastructure:**  
- Docker  
- PostgreSQL  
- Redis (optional for caching)  

**Monitoring:**  
- Prometheus  
- Grafana  

**Frontend:**
- HTML
- CSS
- JavaScript

**Containerization:** 
- Docker

**API Docs:**
- Swagger UI (OpenAPI)
---

## License  
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact  
For support or questions:  
📧 Email: [dntwaritag@alustudent.com](mailto:dntwaritag@alustudent.com)  
🐙 GitHub: [@dntwaritag](https://github.com/dntwaritag)

---

