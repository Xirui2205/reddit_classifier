from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas.auth import LoginRequest, Token
from ..schemas.user import UserCreate, UserOut
from ..security import get_db, get_current_user, require_admin
from ..services import auth as auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, payload.email, payload.password)
    return auth_service.issue_token(user)


@router.post("/register", response_model=UserOut)
def register(user_in: UserCreate, db: Session = Depends(get_db), admin=Depends(require_admin)):
    user = auth_service.create_user(db, user_in, requester=admin)
    return user


@router.get("/me", response_model=UserOut)
def me(current_user=Depends(get_current_user)):
    return current_user
