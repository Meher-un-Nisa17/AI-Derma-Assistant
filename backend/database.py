from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# This creates the local database file
SQLALCHEMY_DATABASE_URL = "sqlite:///./skin_app.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# This is your "Scans" table
class ScanResult(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    condition = Column(String)
    confidence = Column(Float)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Create the tables
def init_db():
    Base.metadata.create_all(bind=engine)