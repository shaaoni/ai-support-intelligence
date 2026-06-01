"""
config.py — single source of truth for all environment variables.

Import this anywhere in the project:
    from src.config import settings
    print(settings.openai_api_key)
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (works regardless of where you run from)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    # LLM
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # LangSmith
    langchain_tracing: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "ai-support-intelligence")

    # App
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Paths (always absolute, resolved from project root)
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_dir: Path = ROOT_DIR / "data" / "raw"
    chroma_dir: Path = ROOT_DIR / os.getenv("CHROMA_PERSIST_DIR", "data/chroma")

    # ML
    classifier_api_url: str = os.getenv("CLASSIFIER_API_URL", "http://localhost:8000")

    def validate(self) -> list[str]:
        """Returns a list of missing required config values."""
        missing = []
        if not self.openai_api_key or self.openai_api_key == "your-openai-key-here":
            missing.append("OPENAI_API_KEY")
        if not self.langchain_api_key or self.langchain_api_key == "your-langsmith-key-here":
            missing.append("LANGCHAIN_API_KEY (optional but recommended)")
        return missing


settings = Settings()