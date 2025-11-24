from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..models import PersonaTemplate
from ..schemas.persona import PersonaCreate, PersonaUpdate


def list_personas(db: Session) -> list[PersonaTemplate]:
    return db.query(PersonaTemplate).all()


def create_persona(db: Session, persona_in: PersonaCreate) -> PersonaTemplate:
    existing = db.query(PersonaTemplate).filter(PersonaTemplate.code == persona_in.code).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Persona code already exists")
    persona = PersonaTemplate(**persona_in.dict())
    db.add(persona)
    db.commit()
    db.refresh(persona)
    return persona


def update_persona(db: Session, persona_id: int, persona_in: PersonaUpdate) -> PersonaTemplate:
    persona = db.query(PersonaTemplate).filter(PersonaTemplate.id == persona_id).first()
    if not persona:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Persona not found")
    for field, value in persona_in.dict(exclude_unset=True).items():
        setattr(persona, field, value)
    db.commit()
    db.refresh(persona)
    return persona
