import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama3.2")
FAST_MODEL = os.getenv("FAST_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CONTEXT_WINDOW_TOKENS = int(os.getenv("CONTEXT_WINDOW_TOKENS", "8192"))
CONTEXT_WARNING_THRESHOLD = float(os.getenv("CONTEXT_WARNING_THRESHOLD", "0.75"))