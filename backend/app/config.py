from dataclasses import dataclass
import os


@dataclass
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./corpus.db")
    jwt_secret: str = os.getenv("JWT_SECRET", "super-secret-key")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE", "60"))


def get_settings() -> Settings:
    return Settings()
