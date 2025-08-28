import os

DATABASE_URL="postgresql://postgres:secret@localhost:5433/postgres"
print("Using DB URL:", os.getenv("DATABASE_URL"))
OPENAI_API_KEY = "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"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# Add or update these settings
USE_OPENAI_API = False  # Set to False to use local models
LOCAL_MODELS = {
    "summarization": "facebook/bart-large-cnn",
    "response": "facebook/blenderbot-400M-distill"
}

# Model Parameters
MODEL_PARAMS = {
    "max_length": 150,
    "min_length": 50,
    "temperature": 0.7,
    "top_p": 0.9
}

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create database engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()