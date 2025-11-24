from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ..database import Base


class CorpusTopic(Base):
    __tablename__ = "corpus_topics"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True)
    title = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    domain = Column(String, nullable=True)
    target_tokens_per_session = Column(Integer, nullable=False)
    total_token_target = Column(Integer, nullable=False)
    total_sessions = Column(Integer, default=0)
    total_user_tokens_collected = Column(Integer, default=0)
    min_turns = Column(Integer, default=4)
    max_turns = Column(Integer, default=14)
    persona_id = Column(Integer, ForeignKey("persona_templates.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    persona = relationship("PersonaTemplate")
