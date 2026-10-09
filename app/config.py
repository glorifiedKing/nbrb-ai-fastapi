from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    GEMINI_API_KEY: Optional[str] = ""
    DATABASE_URL: str = "postgresql://root_user:12345678@127.0.0.1:5432/nbrb_ai"
    REDIS_URL: str = "redis://localhost:6379/0"
    LARAVEL_WEBHOOK_URL: str = "http://localhost:8000/api/internal/compliance-webhook"
    INTERNAL_API_SECRET: str = "bims-secure-internal-secret"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
