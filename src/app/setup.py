from setuptools import setup, find_packages

setup(
    name="weatherwise",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Core Data Processing
        'pandas>=1.3.4',
        'numpy>=1.21.2',
        
        # Machine Learning
        'scikit-learn>=1.0.2',
        'tensorflow>=2.6.0',
        
        # Model Serialization
        'joblib>=1.0.1',
        
        # Database
        'sqlalchemy>=1.4.39',
        'psycopg2-binary>=2.9.3',  # If using PostgreSQL
        
        # API
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'python-dotenv>=0.19.0',
        
        # Visualization
        'matplotlib>=3.4.3',
        'seaborn>=0.11.2',
        
        # Utilities
        'python-dateutil>=2.8.2',
        'python-multipart>=0.0.5'  # For file uploads in FastAPI
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.5',
            'flake8>=4.0.1',
            'black>=22.3.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'weatherwise-train=src.app.main:main',
        ],
    },
)