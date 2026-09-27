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
export GOOGLE_API_KEY=...            # free key: https://aistudio.google.com/apikey

python3 -m advisor.ingest            # 4 documents -> catalogue + cited chunks
python3 -m advisor.synthetic         # 10 anonymised student profiles
python3 -m advisor.conflicts         # mine curriculum contradictions
python3 scripts/run_evaluation.py    # 34 cases x 4 variants -> results/
python3 scripts/build_report.py      # -> report/REPORT.md   (with real numbers)
python3 scripts/build_ppt.py         # -> report/*.pptx      (with real numbers)

streamlit run app.py                 # the prototype
```

The notebook `notebooks/GenAI_A1_AI_Academic_Advisor.ipynb` walks Phases 1–5 cell by cell.

## Deploy (free public demo link)

1. Push this folder to a **public** GitHub repo.
2. <https://share.streamlit.io> → *New app* → main file `app.py`.
3. *Advanced settings → Secrets*:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```
4. Put the URL on the report cover page and the demo slide.

`.gitignore` already excludes `.streamlit/secrets.toml`. Never commit the key.

## Layout

| Path | What it is |
|---|---|
| `advisor/ingest.py` | PDF + Excel → structured tables and clause-cited chunks |
| `advisor/retriever.py` | BM25 ⊕ dense hybrid; **absolute-score** abstention |
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
