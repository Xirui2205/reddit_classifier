from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import CorpusTopic
from ..schemas.topic import TopicCreate, TopicUpdate


def list_topics(db: Session) -> list[CorpusTopic]:
    return db.query(CorpusTopic).all()


def create_topic(db: Session, topic_in: TopicCreate) -> CorpusTopic:
    existing = db.query(CorpusTopic).filter(CorpusTopic.code == topic_in.code).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Topic code already exists")
    topic = CorpusTopic(**topic_in.dict())
    db.add(topic)
    db.commit()
    db.refresh(topic)
    return topic


def update_topic(db: Session, topic_id: int, topic_in: TopicUpdate) -> CorpusTopic:
    topic = db.query(CorpusTopic).filter(CorpusTopic.id == topic_id).first()
    if not topic:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found")
    for field, value in topic_in.dict(exclude_unset=True).items():
        setattr(topic, field, value)
    db.commit()
    db.refresh(topic)
    return topic


def pick_next_topic(db: Session) -> CorpusTopic | None:
    eligible = db.query(CorpusTopic).filter(
        CorpusTopic.is_active.is_(True),
        CorpusTopic.total_user_tokens_collected < CorpusTopic.total_token_target,
    )
    topic = eligible.order_by(CorpusTopic.total_user_tokens_collected.asc(), func.random()).first()
    return topic
