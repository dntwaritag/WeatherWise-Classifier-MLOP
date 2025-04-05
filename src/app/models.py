from sqlalchemy import Column, Integer, Float, String, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class WeatherData(Base):
    __tablename__ = 'weather_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=True)  # Made nullable
    precipitation = Column(Float)
    temp_max = Column(Float)
    temp_min = Column(Float)
    wind = Column(Float)
    weather = Column(String)  # 'rain', 'sun', 'fog', etc.