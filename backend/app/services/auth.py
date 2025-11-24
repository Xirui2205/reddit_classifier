from datetime import timedelta
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..models import User
from ..schemas.user import UserCreate, UserOut
from ..schemas.auth import Token
from ..security import create_access_token
from ..utils.crypto import hash_password, verify_password
from ..config import get_settings

settings = get_settings()


def create_user(db: Session, user_in: UserCreate, requester: User | None = None) -> User:
    if requester and requester.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    existing = db.query(User).filter(User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        password_hash=hash_password(user_in.password),
        role=user_in.role or "annotator",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not verify_password(password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")
    return user


def issue_token(user: User) -> Token:
    token = create_access_token({"sub": user.email}, expires_delta=timedelta(minutes=settings.access_token_expire_minutes))
    return Token(access_token=token)
