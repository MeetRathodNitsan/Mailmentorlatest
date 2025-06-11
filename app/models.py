from sqlalchemy import Column, String, Text, DateTime
from app.config import Base

class Email(Base):
    __tablename__ = "emails"
    
    id = Column(String, primary_key=True)
    sender = Column(Text)
    recipient = Column(Text)
    subject = Column(Text)
    body = Column(Text)
    timestamp = Column(DateTime)
    category = Column(Text)
    summary = Column(Text)
    ai_response = Column(Text)
    status = Column(String, default='pending')