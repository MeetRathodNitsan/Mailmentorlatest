from sqlalchemy import Column, Integer, Text, inspect
from werkzeug.security import generate_password_hash, check_password_hash
from app.config import Base, SessionLocal, engine

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    email = Column(Text, unique=True, nullable=False)
    password = Column(Text, nullable=False)
    gmail_credentials = Column(Text, nullable=True)  # This stores Gmail credentials
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password, password)

def authenticate_user(email: str, password: str):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user and user.check_password(password):
            return user
        return None
    finally:
        session.close()

# Create all tables if they don't exist
inspector = inspect(engine)
if not inspector.has_table("users"):
    Base.metadata.create_all(engine)
