from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

# --- CONFIGURATION ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./skin_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DATABASE DEPENDENCY (For main.py) ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- TABLES ---

class ChatSession(Base):
    """
    Groups multiple scans into one 'Chat' thread in the History screen.
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="New Analysis") 
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship: One session can have many scan results
    scans = relationship("ScanResult", back_populates="session", cascade="all, delete-orphan")

class ScanResult(Base):
    """
    Stores individual AI analysis results.
    """
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    condition = Column(String)
    confidence = Column(Float)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Link to the ChatSession
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    session = relationship("ChatSession", back_populates="scans")

# --- INITIALIZATION ---
def init_db():
    Base.metadata.create_all(bind=engine)