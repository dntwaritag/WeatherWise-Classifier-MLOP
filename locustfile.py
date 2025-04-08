from locust import HttpUser, task, between
import random

class WeatherWiseUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict_single(self):
        payload = {
            "precipitation": random.uniform(0, 10),
            "temp_max": random.uniform(-10, 40),
            "temp_min": random.uniform(-15, 35),
            "wind": random.uniform(0, 30),
            "lag_wind_1": random.uniform(0, 30),
            "lag_precipitation_1": random.uniform(0, 10),
            "lag_temp_max_1": random.uniform(-10, 40),
            "lag_temp_min_1": random.uniform(-15, 35)
        }
        self.client.post("/predict-single/", json=payload)
    
    @task(3)
    def predict_bulk(self):
        # Simulate bulk prediction with a small CSV
        files = {"file": ("test_data.csv", open("data/test.csv", "rb"), "text/csv")}
        self.client.post("/predict-bulk/", files=files)
    
    @task(1)
    def retrain(self):
        # Simulate retraining trigger
        self.client.post("/retrain/")