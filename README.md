# AI Academic Advisor — DATA308 Generative AI, Assignment #1

> *"Design, Build, Deploy and Evaluate an AI Academic Advisor for University that knows when
> to answer, what to answer, when to ask and when to say it does not have enough information."*

A retrieval-augmented academic advisor built on four real university documents, with a
deterministic rule engine for eligibility, automatic conflict detection, and a 34-case
verified evaluation across four system variants.

**Core design decision — numbers are computed, prose is retrieved, the model only explains
and cites.** Prerequisites, credits, attendance and progression are evaluated in Python and
injected into the prompt as verified fact the model may not recompute.

## Quick start

```bash
pip install -r requirements.txt

# keys go in .env (git-ignored):
#   GROQ_API_KEY=gsk_...          primary   (Groq gpt-oss-120b, then gpt-oss-20b)
#   GOOGLE_API_KEY=...            fallback  (Gemini 3.6 Flash)
#   GOOGLE_API_KEY_2=...          optional second Gemini key

python3 -m advisor.ingest                 # 4 documents -> catalogue + structure-aware chunks
python3 -m advisor.synthetic              # 10 anonymised student profiles
python3 -m advisor.conflicts              # mine curriculum contradictions
python3 -m advisor.retriever              # build the FAISS vector DB + smoke test
python3 scripts/calibrate_abstention.py   # fit the abstention gate
python3 scripts/run_evaluation.py         # 34 cases x 4 variants -> results/
python3 scripts/build_report.py           # -> report/REPORT.md
python3 scripts/build_ppt.py              # -> report/*.pptx

streamlit run app.py                      # the prototype
```

The notebook `notebooks/GenAI_A1_AI_Academic_Advisor.ipynb` walks Phases 1–5 cell by cell.

## Retrieval stack

| Stage | Component |
|---|---|
| Chunking | structure-aware: clause blocks packed within one top-level clause (≤ ~300 tokens), contents pages dropped, contextual header per chunk |
| Embeddings | `BAAI/bge-small-en-v1.5` sentence transformer (384-d, local) |
| Vector DB | FAISS `IndexFlatIP` (exact cosine), persisted in `data/processed/vectorstore/` |
| Hybrid | BM25 + dense, Reciprocal Rank Fusion, metadata filter on the student's batch |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-20 → top-4 |
| Abstain | gate on the cross-encoder's absolute score (calibrated) |
| LLM | Groq `gpt-oss-120b` → Groq `gpt-oss-20b` → Gemini `gemini-3.6-flash` |

## Deploy (Streamlit Community Cloud)

1. <https://share.streamlit.io> → *Create app* → repo `Shivshankar8261/Academic-Advisor`, branch `main`, file `app.py`.
2. *Advanced settings → Secrets*:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GOOGLE_API_KEY = "..."
   GOOGLE_API_KEY_2 = "..."
   ```
3. Deploy. `.env` and `.streamlit/secrets.toml` are git-ignored — never commit keys.

## Layout

| Path | What it is |
|---|---|
| `advisor/ingest.py` | PDF + Excel → structured tables and clause-cited chunks |
| `advisor/retriever.py` | BM25 + FAISS → RRF → cross-encoder rerank; abstention gate |
| `advisor/llm.py` | Groq → Groq → Gemini router with token-window throttle |
| `advisor/eligibility.py` | deterministic rule engine, every rule clause-cited |
| `advisor/conflicts.py` | dangling / ordering / cross-batch conflict mining |
| `advisor/prompts.py` | the four variants, exactly one ingredient apart |
| `advisor/advisor.py` | orchestration, follow-up protocol, abstention |
| `advisor/synthetic.py` | 10 anonymised synthetic student profiles |
| `advisor/testset.py` | 34 verified test cases with hallucination traps |
| `advisor/evaluate.py` | scoring rules + the eight graded metrics |
| `app.py` | Streamlit prototype |

## The four variants

| Variant | Adds |
|---|---|
| V1 Basic LLM | nothing — the question, sent to the model |
| V2 Structured Prompting | role, context, constraints, delimiters, output contract |
| V3 RAG | retrieved evidence, citation duty, abstention duty |
| V4 RAG + Structured Student Data | verified rule-engine facts, conflicts, follow-up protocol |

## Two findings worth knowing

**Table 2 of the handbook is an image.** The letter-grade → grade-point scale never reaches
the PDF text layer, so no grade-point arithmetic is possible. The system says so instead of
reciting the grade points common at Indian universities. This is test case `M01`.

**The curriculum contradicts itself.** 30 conflicts were mined from the provided
spreadsheet — most sharply, `DATA301` requires `DATA202+MATH301` for the 2022/2024 batches
but `DATA206+MATH301` for 2025/2026-DS, and `DATA206` is not offered anywhere in the 2025
spread. The same question has two different correct answers depending on the student's
batch, which is why the advisor asks before answering.

## Privacy

No real student data is used. All profiles are synthetic, identified by opaque tokens
(`STU-A`…`STU-J`), with no names, enrolment numbers, contact details or dates of birth.
The interface states on every screen that advice is advisory and must be confirmed with the
Faculty Advisor, matching the SOP.
