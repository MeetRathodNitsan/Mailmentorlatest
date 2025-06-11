DATABASE_URL = "postgresql://postgres:admin@localhost:5432/mailmentor"
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