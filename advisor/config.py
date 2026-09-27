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

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))   # quota meters CONTENTS, not calls
EMBED_PAUSE = float(os.getenv("EMBED_PAUSE", "11"))  # seconds between batches (free tier)
TEMPERATURE = 0.0          # deterministic: required for reproducible evaluation
MAX_OUTPUT_TOKENS = 1200

# --- Retrieval ---
TOP_K = 6                  # chunks passed to the generator
BM25_WEIGHT = 0.5          # hybrid fusion weight (lexical vs dense)
MIN_DENSE_SIM = 0.62       # abstain below this raw cosine similarity (absolute, not ranked)
MIN_TERM_COVERAGE = 0.34   # ...and below this share of query words found in any chunk
CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 150

CURRENT_TERM = "Semester 5 (Odd), AY 2026-27"
