import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
# The .env file is in the parent of app/ directory or current working directory
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DB_DIR = BASE_DIR / "chroma_db"

# Create directories if they do not exist
DB_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# RAG Configuration
CHUNK_SIZE_MIN = 800
CHUNK_SIZE_MAX = 1200
CHUNK_OVERLAP = 200

# Default model selections
DEFAULT_EMBEDDING_PROVIDER = "gemini"  # Options: gemini, openai, huggingface
DEFAULT_LLM_PROVIDER = "groq" if os.getenv("GROQ_API_KEY") else "gemini"       # Options: gemini, openai, anthropic, groq
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant" if os.getenv("GROQ_API_KEY") else "gemini-2.5-flash"
