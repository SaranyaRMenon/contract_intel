import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _here = Path(__file__).resolve().parents[2]
    for env_path in (_here / ".env", _here.parent / ".env", _here / ".env.example"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONTRACTS_DIR = DATA_DIR / "contracts"
CHROMA_DIR = DATA_DIR / "chroma"

CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")

DEFAULT_MODEL = os.environ.get("CI_MODEL", "llama-3.3-70b-versatile")
SMART_MODEL = os.environ.get("CI_SMART_MODEL", "llama-3.3-70b-versatile")
FAST_MODEL = os.environ.get("CI_FAST_MODEL", "llama-3.1-8b-instant")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

COLLECTION_NAME = "contracts"

