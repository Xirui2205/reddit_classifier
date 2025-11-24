from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas.persona import PersonaCreate, PersonaOut, PersonaUpdate
from ..security import get_current_user, get_db, require_admin
from ..services import persona as persona_service

router = APIRouter(prefix="/api/personas", tags=["personas"])


@router.get("", response_model=list[PersonaOut])
def list_personas(db: Session = Depends(get_db), _: Depends = Depends(get_current_user)):
    return persona_service.list_personas(db)


@router.post("", response_model=PersonaOut)
def create_persona(persona_in: PersonaCreate, db: Session = Depends(get_db), admin=Depends(require_admin)):
    return persona_service.create_persona(db, persona_in)


@router.put("/{persona_id}", response_model=PersonaOut)
def update_persona(persona_id: int, persona_in: PersonaUpdate, db: Session = Depends(get_db), admin=Depends(require_admin)):
    return persona_service.update_persona(db, persona_id, persona_in)
