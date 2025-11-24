from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from ..database import Base


class PersonaTemplate(Base):
    __tablename__ = "persona_templates"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    prompt_text = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
