from datetime import datetime
from pydantic import BaseModel


class PersonaBase(BaseModel):
    code: str
    name: str | None = None
    description: str | None = None
    prompt_text: str
    is_active: bool | None = True


class PersonaCreate(PersonaBase):
    pass


class PersonaUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    prompt_text: str | None = None
    is_active: bool | None = None


class PersonaOut(PersonaBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
