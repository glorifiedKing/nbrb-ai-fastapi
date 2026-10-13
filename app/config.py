from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Path(__file__).resolve().parent is the directory where config.py lives (i.e. 'app')
ENV_PATH = Path(__file__).resolve().parent / ".env"

class Settings(BaseSettings):
    GEMINI_API_KEY: Optional[str] = ""
    DATABASE_URL: str
    REDIS_URL: str
    LARAVEL_WEBHOOK_URL: str
    INTERNAL_API_SECRET: str

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()