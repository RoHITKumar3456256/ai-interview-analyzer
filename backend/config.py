import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

class Settings(BaseSettings):
    APP_NAME: str = "AI Interview Analyzer API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Paths
    TRANSCRIPTS_PATH: Path = DATA_DIR / "transcripts.json"
    INTERVIEW_GUIDE_PATH: Path = DATA_DIR / "interview_guide.json"
    
    # LLM Settings & Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_LLM_PROVIDER: str = "auto"  # "gemini", "openai", "local"
    
    # Retrieval Hyperparameters
    TOP_K_CHUNKS: int = 6
    SIMILARITY_THRESHOLD: float = 0.25
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
