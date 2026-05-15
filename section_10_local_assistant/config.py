from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
PROMPTS_DIR = BASE_DIR / "prompts"

DATA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
PROMPTS_DIR.mkdir(exist_ok=True)

CHECKPOINTS_DB = str(DATA_DIR / "checkpoints.db")
TRACES_DB       = str(DATA_DIR / "traces.db")
TASKS_DB        = str(DATA_DIR / "tasks.db")
MEMORY_DB       = str(DATA_DIR / "memory.db")

PRIMARY_MODEL   = os.getenv("PRIMARY_MODEL",   "qwen2.5:7b")
FAST_MODEL      = os.getenv("FAST_MODEL",      "qwen2.5:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MAX_ITERATIONS             = int(os.getenv("MAX_ITERATIONS", "15"))
MAX_RETRIEVAL_STEPS        = int(os.getenv("MAX_RETRIEVAL_STEPS", "4"))
MAX_CONTEXT_TOKENS         = int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
CONTEXT_WARNING_THRESHOLD  = float(os.getenv("CONTEXT_WARNING_THRESHOLD", "0.75"))

MAX_MEMORY_ITEMS                 = int(os.getenv("MAX_MEMORY_ITEMS", "10"))
CONVERSATION_SUMMARY_THRESHOLD   = int(os.getenv("CONVERSATION_SUMMARY_THRESHOLD", "8"))

EVAL_PASS_THRESHOLD = float(os.getenv("EVAL_PASS_THRESHOLD", "0.70"))
EVAL_JUDGE_MODEL    = PRIMARY_MODEL

API_HOST          = os.getenv("API_HOST", "0.0.0.0")
API_PORT          = int(os.getenv("API_PORT", "8000"))
TRACE_VIEWER_PORT = int(os.getenv("TRACE_VIEWER_PORT", "8001"))

ENABLE_HITL        = os.getenv("ENABLE_HITL",        "true").lower() == "true"
ENABLE_TRACING     = os.getenv("ENABLE_TRACING",     "true").lower() == "true"
ENABLE_MULTI_AGENT = os.getenv("ENABLE_MULTI_AGENT", "true").lower() == "true"