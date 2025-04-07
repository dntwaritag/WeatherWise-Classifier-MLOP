FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq-dev gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create static directory for favicon
RUN mkdir -p src/app/static

# Copy application
COPY . .

# Verify favicon exists (optional)
RUN if [ -f "src/app/static/favicon.ico" ]; then \
        echo "Favicon found"; \
    else \
        echo "Warning: No favicon.ico found in static directory"; \
    fi
    
ENV PYTHONPATH=/app
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]