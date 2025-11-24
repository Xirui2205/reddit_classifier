import hashlib
import hmac
import os

from ..config import get_settings


settings = get_settings()


def hash_password(password: str) -> str:
    salt = settings.jwt_secret.encode()
    return hmac.new(salt, password.encode(), hashlib.sha256).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hmac.compare_digest(hash_password(password), password_hash)
