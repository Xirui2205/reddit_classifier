from datetime import datetime
from pydantic import BaseModel


class TopicBase(BaseModel):
    code: str
    title: str | None = None
    description: str | None = None
    domain: str | None = None
    target_tokens_per_session: int
    total_token_target: int
    min_turns: int | None = 4
    max_turns: int | None = 14
    persona_id: int | None = None
    is_active: bool | None = True


class TopicCreate(TopicBase):
    pass


class TopicUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    domain: str | None = None
    target_tokens_per_session: int | None = None
    total_token_target: int | None = None
    min_turns: int | None = None
    max_turns: int | None = None
    persona_id: int | None = None
    is_active: bool | None = None


class TopicOut(TopicBase):
    id: int
    total_sessions: int
    total_user_tokens_collected: int
    created_at: datetime

    class Config:
        orm_mode = True
