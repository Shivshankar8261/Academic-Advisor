"""Central configuration. Every path and tunable lives here."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
for _d in (PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# --- Source documents (Section 2 "Data Provided" of the assignment) ---
SOURCES = {
    "handbook": RAW / "student_handbook.pdf",
    "sop": RAW / "sop_students.pdf",
    "structure": RAW / "semester_spread_structures.xlsx",
    "minors": RAW / "minor_courses.xlsx",
}

# --- Derived artefacts ---
COURSES_CSV = PROCESSED / "courses.csv"
MINORS_CSV = PROCESSED / "minors.csv"
BASKETS_CSV = PROCESSED / "baskets.csv"
CHUNKS_JSONL = PROCESSED / "chunks.jsonl"
STUDENTS_JSON = PROCESSED / "synthetic_students.json"
TESTSET_JSON = PROCESSED / "testset.json"

# --- LLM: Groq primary, Gemini fallback ---------------------------------
def _secret(name: str) -> str:
    """Environment first, then .env, then Streamlit secrets."""
    if os.getenv(name):
        return os.getenv(name, "")
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"')
    try:
        import streamlit as st
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-20b"]   # separate per-model quota
GROQ_REASONING_EFFORT = "low"      # gpt-oss is a reasoning model; low keeps TPM usage sane
GROQ_TPM = 8000                    # free-tier tokens/minute for this model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
TEMPERATURE = 0.0                  # deterministic: required for reproducible evaluation
MAX_OUTPUT_TOKENS = 1000           # includes gpt-oss reasoning tokens
# How long to wait for Groq's per-minute window before falling back to Gemini.
# Evaluation waits (comparability: one model answers every case); the app does not.
GROQ_MAX_WAIT_S = float(os.getenv("GROQ_MAX_WAIT_S", "8"))

# --- Retrieval ---------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"          # 384-d sentence transformer
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTORSTORE_DIR = PROCESSED / "vectorstore"     # FAISS index + metadata
CANDIDATES = 30            # per retriever, before fusion
RERANK_POOL = 20           # fused candidates sent to the cross-encoder
TOP_K = 4                  # passages given to the generator
RERANK_MIN = -6.97          # abstain below this cross-encoder logit (calibrated)
CHUNK_MAX_WORDS = 220      # ~300 tokens: inside bge-small's 512-token window

CURRENT_TERM = "Semester 5 (Odd), AY 2026-27"
