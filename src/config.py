"""
config.py — Central configuration for Spiritual AI.
All paths, model names, and tuning parameters live here.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent   # Spiritual_AI/

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR      = ROOT_DIR / "data"
PDFS_DIR      = DATA_DIR / "pdfs"
VECTORSTORE   = DATA_DIR / "vectorstore" / "chroma_db"
CHUNKS_PATH   = DATA_DIR / "chunks.json"

# ── Model config ──────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL     = "openai/gpt-oss-120b"

# ── Retrieval tuning ──────────────────────────────────────────────────────────
RETRIEVAL_K   = 8      # candidates fetched from each retriever
FINAL_K       = 5      # kept after contextual compression
SIM_THRESHOLD = 0.10   # minimum cosine similarity to keep a chunk
MMR_FETCH_K   = 20     # candidates for MMR diversity pass
