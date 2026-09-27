"""AI Academic Advisor - DATA308 Generative AI Assignment #1."""
import os as _os

# FAISS and PyTorch each ship their own OpenMP runtime. On macOS loading both
# into one process segfaults (exit 139) unless duplicate runtimes are allowed.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

__version__ = "2.0.0"
