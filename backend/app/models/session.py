from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from ..database import Base


class CorpusSession(Base):
    __tablename__ = "corpus_sessions"

    id = Column(Integer, primary_key=True, index=True)
    annotator_id = Column(Integer, ForeignKey("users.id"))
    topic_id = Column(Integer, ForeignKey("corpus_topics.id"))
    status = Column(String, default="active")
    user_token_target = Column(Integer, nullable=False)
    user_token_count = Column(Integer, default=0)
    turn_count = Column(Integer, default=0)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    annotator = relationship("User")
    topic = relationship("CorpusTopic")
    messages = relationship("CorpusMessage", back_populates="session", cascade="all, delete-orphan")


class CorpusMessage(Base):
    __tablename__ = "corpus_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("corpus_sessions.id", ondelete="CASCADE"))
    sender_role = Column(String, nullable=False)
    lang = Column(String, nullable=False)
    text = Column(String, nullable=False)
    token_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("CorpusSession", back_populates="messages")
