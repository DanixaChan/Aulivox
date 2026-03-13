from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    original_text = Column(Text)
    summary_text = Column(Text)
    audio_file = Column(String)