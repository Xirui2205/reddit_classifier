from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas.topic import TopicCreate, TopicOut, TopicUpdate
from ..security import get_current_user, get_db, require_admin
from ..services import topic as topic_service

router = APIRouter(prefix="/api/topics", tags=["topics"])


@router.get("", response_model=list[TopicOut])
def list_topics(db: Session = Depends(get_db), _: Depends = Depends(get_current_user)):
    return topic_service.list_topics(db)


@router.post("", response_model=TopicOut)
def create_topic(topic_in: TopicCreate, db: Session = Depends(get_db), admin=Depends(require_admin)):
    return topic_service.create_topic(db, topic_in)


@router.put("/{topic_id}", response_model=TopicOut)
def update_topic(topic_id: int, topic_in: TopicUpdate, db: Session = Depends(get_db), admin=Depends(require_admin)):
    return topic_service.update_topic(db, topic_id, topic_in)
