from datetime import datetime
from pydantic import BaseModel

from .topic import TopicOut
from .user import UserOut


class MessageOut(BaseModel):
    id: int
    sender_role: str
    lang: str
    text: str
    token_count: int
    created_at: datetime

    class Config:
        orm_mode = True


class SessionOut(BaseModel):
    id: int
    status: str
    user_token_target: int
    user_token_count: int
    turn_count: int
    started_at: datetime
    completed_at: datetime | None
    topic: TopicOut

    class Config:
        orm_mode = True


class SessionDetail(SessionOut):
    messages: list[MessageOut]


class SessionReply(BaseModel):
    text: str
