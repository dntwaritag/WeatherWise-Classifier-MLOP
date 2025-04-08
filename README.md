# WeatherWise-Classifier MLOP Project  
**Weather Classification using FastAPI, Docker, and MLOps Best Practices**

## Project Overview  
This project presents a weather classification system leveraging Machine Learning and MLOps principles. It enables users to classify weather conditions (e.g., rain, sun, fog) using meteorological data via a FastAPI backend, supported by model monitoring, versioning, and retraining workflows.

## Accessing the Site  
- **Backend (FastAPI)**: [https://weatherwise-backend-ok73.onrender.com](https://weatherwise-backend-ok73.onrender.com)  
- **Frontend**: [https://weatherwise-frontend.onrender.com ](https://weatherwise-frontend-prjt.onrender.com) 
- **Demo Video**: *[Insert YouTube demo video link here]*  
- **GitHub Repo**: *[Insert GitHub repository link here]*  

## Features  
- Real-time single and batch weather classification  
- Upload CSV datasets for bulk predictions  
- Monitoring for model drift and retraining triggers  
- Model performance metrics available via endpoints  
- Dockerized for easy deployment and CI/CD compatibility  

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

8.  **License**
This project is licensed under the MIT License. See LICENSE for details.

9. **Contact**
For support or questions:
- Email: dntwaritag@alustudent.com
- GitHub: @dntwaritag
