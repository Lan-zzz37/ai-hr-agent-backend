"""
项目全局配置文件
面向智能制造转型的AI人岗匹配与技能重塑Agent
"""
import os
from pathlib import Path

# ==================== 项目路径 ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"

# 确保目录存在
for d in [DATA_DIR, UPLOAD_DIR, KNOWLEDGE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== 数据库配置 ====================
SQLITE_DB_PATH = str(DATA_DIR / "smart_manufacturing.db")

# ==================== Chroma 向量数据库配置 ====================
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "skill_knowledge"

# ==================== 大模型配置 ====================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai / zhipu / qwen
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ==================== Embedding 配置 ====================
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", LLM_API_BASE)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

# ==================== FastAPI 配置 ====================
API_TITLE = "AI人岗匹配与技能重塑Agent"
API_VERSION = "1.0.0"
API_DESCRIPTION = "面向智能制造转型的大模型Agent系统，提供人岗匹配、技能差距分析和技能重塑路径规划"
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# ==================== 医配引擎配置 ====================
MATCHING_TOP_K = int(os.getenv("MATCHING_TOP_K", "5"))
SKGAP_THRESHOLD = float(os.getenv("SKGAP_THRESHOLD", "0.3"))