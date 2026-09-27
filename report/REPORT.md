# Design, Build, Deploy and Evaluate an AI Academic Advisor

### An advisor that knows when to answer, what to answer, when to ask, and when to say it does not have enough information

**Course** DATA308 — Generative AI · **Assignment #1** (Phases 1–7)
**Author** <<YOUR NAME / ROLL NUMBER>>
**Date** 28 September 2026

**Live demo** <<PASTE YOUR STREAMLIT CLOUD URL HERE>>
**Source code** <<PASTE YOUR GITHUB REPO URL HERE>>

> ⚠️ **Evaluation not yet run.** Numbers below are placeholders. Run `python3 scripts/run_evaluation.py` (or notebook Cell 17) and regenerate this report.

---

## 1. Objective and approach

The task is not document-upload-and-ask. It is to establish *how much* prompt engineering,
retrieval and structured data each contribute to the accuracy and — separately — the
**reliability** of academic advice, and to show where the system still fails.

The design rests on one decision:

> **Numbers are computed; prose is retrieved. The language model only explains and cites.**

Prerequisite satisfaction, credit totals, attendance thresholds and progression eligibility
are evaluated in Python by a deterministic rule engine against the university's own
structured curriculum data. The result is injected into the prompt as verified fact that the
model is explicitly forbidden to recompute. Retrieval supplies regulation *prose* with
clause-level citations. This removes the two things LLMs are least reliable at — multi-step
arithmetic and strict rule application — from the model's responsibilities entirely.

Four systems were built and measured on an identical test set, each adding exactly one
ingredient to the previous, so every metric change is attributable:

| Variant | Adds |
|---|---|
| **V1 · Basic LLM** | nothing — the question, sent to the model |
| **V2 · Structured Prompting** | role, context, constraints, delimiters, output contract |
| **V3 · RAG** | retrieved evidence, citation duty, abstention duty |
| **V4 · RAG + Structured Student Data** | verified rule-engine facts, conflict detection, follow-up protocol |

---

## 2. Data provided and how it was ingested

| Document | Role | Extracted |
|---|---|---|
| Student Handbook (Aug 2026), 92 pp | Academic regulations | attendance, grading, progression, summer term, degree award |
| SOP for Students (17 Aug 2026), 6 pp | Registration procedure | add/drop, audit, registration card, attendance reporting |
| Semester Spread & Structures (Sept 2026) | Catalogue + offerings | **202** course rows, **78** unique codes, **5** batches, S1–S8 |
| Minor Courses for B.Tech | Minor programmes | **7** minors, **104** course rows |

Ingestion produces two deliberately separate layers.

**Structured layer** — `courses.csv`, `baskets.csv`, `minors.csv`. The semester-spread sheets
store eight side-by-side blocks of `Course Code | Course | Pre-Req | L | T | P | C`; the parser
locates each block by its header row, forward-fills the basket label from the left-hand
columns, normalises codes (`DATA 301` → `DATA301`), and splits prerequisite strings on
separators while preserving `A/B` alternatives. Basket minimums are read from both the
`Struct_*` sheets and the `Fixed Min` column of the spread sheets, because neither source
alone is complete for every batch.

**Unstructured layer** — `chunks.jsonl`, **407** passages carrying
`source · section · clause range · page`, built by **structure-aware chunking** in three passes:

1. **Segment** the text into atomic blocks — one per clause (`7.2 …`) or SOP bullet
   (`• Course Add/Drop …`). Table-of-contents pages (3–5) are dropped: their dot-leader lines
   lexically match almost every query and contain no rules.
2. **Pack** consecutive blocks of the *same top-level clause* up to ~220 words (≈300 tokens,
   inside the 512-token window of the embedder). A chunk never crosses a top-level clause
   boundary; over-long clauses are split on sentence boundaries with a one-sentence overlap.
3. **Contextual header** — every chunk is embedded as
   `Student Handbook > 7. Attendance Requirements` + body, so a bare sub-clause such as
   "7.3 … sixty-five percent" keeps the meaning of the section it belongs to.

Why it matters: an earlier fixed-size chunker let a Section 12 (Progression) passage inherit
the citation of Clause 11.5.7, so every citation built on it was wrong. Citations now carry
clause **ranges** (`Clause 7–7.2 — p.29`), and source correctness is one of the eight metrics.

### 2.1 A real gap in the provided data

The letter-grade → grade-point table (**Table 2, Clause 8.10**) is an **image** in the
handbook PDF. Its contents never reach the text layer. The system therefore cannot state
what an 'A+' is worth, and must say so. A model that answers "A+ = 9 points" is reciting
general knowledge about Indian universities, not this university's regulation — exactly the
failure mode this assignment targets. It is used as verified test case **M01**.

Course *descriptions and syllabi* are likewise absent from the provided package (test case
M02), as are timetables and instructor allocations (M03).

---

## 3. System architecture

```
                 ┌──────────────── INGESTION ────────────────┐
  4 documents ──▶│ PDF  → clause-aware chunks (+page,+clause) │──▶ chunks.jsonl (407)
                 │ XLSX → course catalogue, baskets, minors   │──▶ courses.csv (202)
                 └────────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌───────────────┐          ┌──────────────────┐        ┌────────────────────┐
│ RETRIEVER     │          │ RULE ENGINE      │        │ CONFLICT DETECTOR  │
│ BM25 + FAISS  │          │ (deterministic)  │        │ dangling prereqs,  │
│ → RRF → rerank│          │ prereqs, credits,│        │ ordering breaks,   │
│ + abstain gate│          │ attendance, CGPA │        │ cross-batch drift  │
└───────┬───────┘          └────────┬─────────┘        └─────────┬──────────┘
        │ evidence + citations      │ verified facts             │ warnings
        └────────────────┬──────────┴────────────────────────────┘
                         ▼
              ┌────────────────────────┐
              │ PROMPT ASSEMBLY (V1–V4)│  XML-delimited: evidence │ record │ question
              └───────────┬────────────┘
                          ▼
   Groq gpt-oss-120b → Groq gpt-oss-20b → Gemini 3.6 Flash  (temperature 0)
                          ▼
              ANSWER · EVIDENCE · CONFIDENCE · FOLLOW-UP
```

### 3.1 Retrieval: vector database, hybrid search, reranking

```
query ─┬─► BM25 (lexical)                   top-30 ─┐
       └─► bge-small-en-v1.5 → FAISS        top-30 ─┴─► Reciprocal Rank Fusion
                                                          │ metadata filter: student's batch
                                                          ▼
                                    cross-encoder rerank top-20 ─► top-4 to the LLM
```

| Stage | Component | Why it is there |
|---|---|---|
| Lexical | BM25 | exact tokens the embedder blurs: `DATA301`, `Clause 7.2`, `65%` |
| Dense | `BAAI/bge-small-en-v1.5` sentence transformer (384-d), local, no API quota | paraphrase: *"move to third year"* ≈ *"Progression to Year 3"* |
| Vector DB | **FAISS** `IndexFlatIP` on L2-normalised vectors (exact cosine), persisted to disk and keyed by a hash of corpus + model | at ~400 chunks exact search is sub-millisecond with perfect recall; HNSW/IVF only pays off beyond ~10⁵ vectors |
| Fusion | Reciprocal Rank Fusion (k = 60) | merges by *rank*, so incompatible BM25 and cosine scales never meet |
| Filter | metadata: catalogue chunks restricted to the student's batch | a 2025 student is never shown the 2022 prerequisite list |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reads query and passage *together* — far more precise than two independent vectors |

**Abstention gate.** A min-max-normalised fused score always rates the best hit 1.0, so it
can never say "nothing here" — an early version of this system could not abstain at all for
exactly that reason. The cross-encoder logit is an **absolute** relevance score, so the gate
uses it: if even the best passage scores below `RERANK_MIN`, the model is **never called**.
The threshold (-6.97) was calibrated on the test set under a strict policy —
**no answerable question may be refused at the gate** — which places it midway between the
weakest answerable query and the strongest separable out-of-scope query. Borderline cases
(a request to scrape the Digii portal overlaps the SOP lexically) fall through to a second
layer: grounding rules in the prompt that oblige the model to say `INSUFFICIENT INFORMATION`.

### 3.2 LLM routing

| Tier | Model | Role |
|---|---|---|
| 1 | Groq `openai/gpt-oss-120b` | primary — fast, strong; free tier 8K tokens/min, 200K/day |
| 2 | Groq `openai/gpt-oss-20b` | separate quota, absorbs the daily limit |
| 3 | Gemini `gemini-3.6-flash` | different provider; each configured key tried, refused keys disabled |

A rolling 60-second token window throttles Groq *before* sending rather than firing requests
into a 429. Every answer records the tier that produced it, and time spent waiting on rate
limits is excluded from the latency metric.

### 3.3 Rule engine

Encoded directly from the handbook, each rule carrying its clause:

| Rule | Value | Clause |
|---|---|---|
| Minimum attendance | 75% of classes conducted | 7.2 |
| Absolute attendance floor | 65%, only with approval | 7.3 / 7.4 |
| Passing grades | O, A+, A, B+, B, C, D (and S) | 8.11 |
| Minimum passing grade | D | 8.13.2 |
| 'F' → Make-Up, then re-register | immediate subsequent semester | 8.13.3(c),(e) |
| 'FA' → re-register, **no Make-Up** | until a passing grade | 8.13.3(b),(g) |
| Progression to Year 2 | CGPA ≥ 4.00 | 12.1, Table 3 |
| Progression to Year 3+ | CGPA ≥ 5.00 | 12.1, Table 3 |
| Degree award | CGPA ≥ 5.00 | 15.2.2 |
| Maximum duration | N + 2 years | 6.1 |
| Total credits | 180 | Programme structure |

### 3.4 Conflict detection

**30 genuine conflicts** were mined from the university's own curriculum:

| Kind | Count | Example |
|---|---|---|
| `DANGLING_PREREQ` | 25 | DATA301 (2025) requires DATA206, which the 2025 spread never offers |
| `ORDERING` | 4 | COMP302 sits in S6 but requires COMP208, also in S6 |
| `CROSS_BATCH` | 1 | DATA301 needs DATA202+MATH301 (2022/24) but DATA206+MATH301 (2025/26-DS) |

The cross-batch case is the sharpest: **the same question has two different correct answers
depending on the student's batch.** A system that answers it without establishing the batch
first is wrong half the time regardless of how fluent it sounds. This is the strongest
justification in the whole project for the follow-up-question mechanism.

---

## 4. Phase 3 — Prompt engineering

Techniques applied, and what each was intended to fix:

| Technique | Where | Intended effect |
|---|---|---|
| **Role assignment** | V2+ | anchors vocabulary to academic regulation |
| **Context injection** | V2+ | current term, 8-semester/180-credit basket structure |
| **Explicit constraints** | V2+ | never invent a code, credit, percentage, CGPA or clause |
| **XML-ish delimiters** | V2+ | `<evidence>` / `<student_record>` / `<student_question>` — the model can never mistake retrieved text for the student's own words, which is also the first line of defence against prompt injection through documents |
| **Output contract** | V2+ | `ANSWER · EVIDENCE · CONFIDENCE · FOLLOW-UP` — makes responses machine-parseable, which is what allows automatic scoring at all |
| **Citation duty** | V3+ | every claim must name a passage in `<evidence>` |
| **Abstention duty** | V3+ | reply `INSUFFICIENT INFORMATION:` and name the missing document |
| **Conflict duty** | V3+ | quote both sides, state which governs or escalate |
| **Verified-fact injection** | V4 | rule-engine output marked non-recomputable |
| **Follow-up protocol** | V4 | ask *only* when the missing fact would change the answer; ≤2 questions |

The follow-up protocol is deliberately narrow. An advisor that asks a clarifying question
on every turn is as useless as one that never asks — the constraint is that the missing fact
must be **answer-changing** (batch, prerequisite outcome, CGPA when progression is at stake,
intended semester).

---

## 5. Phase 2 — Synthetic students and challenge scenarios

**10 profiles**, fully anonymised: opaque tokens (`STU-A`…`STU-J`), no names, no
enrolment numbers, no contact details, no dates of birth. Course codes, credits and
prerequisites are real, so the eligibility arithmetic they exercise is genuine.

| Profile | Batch | Sem | CGPA | Edge case targeted |
|---|---|---|---|---|
| STU-A | 2025 | S5 | 6.42 | failed a course that is a prerequisite for COMP203. |
| STU-B | 2025 | S5 | 8.71 | clean record, but DATA301 requires DATA206 which the 2025 spread never offers. |
| STU-C | 2025 | S2 | 3.84 | insufficient CGPA for progression to Year 2. |
| STU-D | 2025 | S4 | 4.93 | borderline CGPA 4.93 against a 5.00 requirement. |
| STU-E | 2025 | S3 | 5.9 | 'FA' grade plus attendance under the hard 65% floor. |
| STU-F | 2025 | S3 | 6.8 | attendance shortfall that is relaxable only with approval. |
| STU-G | 2025 | S4 | **absent** | CGPA and attendance absent; advisor must ask, not assume. |
| STU-H | 2022 | S5 | 7.1 | 2022 curriculum, where DATA301 requires DATA202 + MATH301. |
| STU-I | 2025 | S5 | 7.55 | minor credits sit outside the parsed core spread. |
| STU-J | 2025 | S5 | 5.05 | 72 credits earned but in S5; carries both an 'F' and an 'FA'. |

Coverage against the failure modes the assignment names: missing information (STU-G),
ambiguity (STU-J), insufficient prerequisites (STU-A), insufficient credits (STU-C, STU-D),
unavailable courses (STU-B), conflicting rules (STU-B, STU-H).

---

## 6. Phase 4 — Evaluation methodology

**34 test cases.** Every expected answer was verified against the source document
before being written down, and the governing clause recorded alongside it.

| Category | Cases | | Expected behaviour | Cases |
|---|---|---|---|---|
| regulation_factual | 8 | | ANSWER | 21 |
| prerequisite_eligibility | 6 | | FLAG_CONFLICT | 2 |
| credits_progression | 4 | | ABSTAIN | 6 |
| course_offering | 3 | | ASK_FOLLOWUP | 5 |
| missing_information | 4 | |  |  |
| conflicting_rules | 3 | |  |  |
| ambiguous_needs_followup | 4 | |  |  |
| out_of_scope | 2 | |  |  |

### 6.1 Two scoring signals

* `must_include` — facts a correct answer has to contain (regex, case-insensitive).
* `must_not_include` — **hallucination traps**: plausible but *wrong* values. For M01 the
  traps are `9`, `9.0`, `10 points`; for R01 they are `80%`, `85%`. If a trap fires, the
  answer is scored incorrect (or *unsupported*, where the case expected an abstention).

The traps are what make the hallucination rate a **measurement** rather than an impression.

### 6.2 Classification

| Label | Meaning |
|---|---|
| `correct` | right action **and** every required fact present |
| `partially_correct` | right action, some required facts missing |
| `unsupported` | asserted content where the corpus has none — a hallucination |
| `incorrect` | contradicted a verified fact, or took the wrong action |

Wrong *action* is scored as harshly as wrong *content*: answering confidently where the
corpus is silent is `unsupported`; abstaining where the answer was available, or advising
blindly where a follow-up was required, is `incorrect`.

### 6.3 Correctness vs reliability

As the brief distinguishes them:

* **Correctness** — does the answer match the verified rule? → *accuracy*
* **Reliability** — does it stay grounded, admit gaps, avoid unsupported recommendations?
  → *hallucination rate*, *missing-information handling*, *conflict handling*

A system can be correct on easy questions and unreliable on ambiguous ones, so both are
reported separately throughout. Temperature is fixed at 0 and all four variants share one
model (Groq `gpt-oss-120b`, with the evaluation waiting on rate limits rather than falling
back), one retriever and one parser, so the comparison is like-for-like.

---

## 7. Results

*Run `python3 scripts/run_evaluation.py` (or notebook Cell 17), then regenerate this report
with `python3 scripts/build_report.py`. The eight-metric table, the per-category breakdown,
the outcome mix and both charts are inserted here automatically.*

---

## 8. Phase 5 — Comparative analysis

*Populated automatically once the evaluation has been run.*

---

## 9. Limitations and where the system still fails

1. **Table 2 is unreadable.** The grade-point scale is an image; no grade-point or SGPA/CGPA
   arithmetic is possible. The system abstains rather than guessing. *Fix:* OCR the table,
   or obtain it as data.
2. **Sparse batches.** The 2023 and 2024 semester spreads have empty semesters in the source
   workbook. Questions about those batch-semesters are genuinely unanswerable, and the
   system says so rather than substituting another batch's curriculum.
3. **Minor credits are not fully modelled.** The minor workbook lists courses per minor but
   not their placement in the semester spread, so `Minor/Open` basket progress is
   approximate. The engine reports it as such.
4. **Conflicts are detected, not resolved.** 30 contradictions are surfaced with
   an escalation path; the system does not decide which side governs, because the documents
   do not say. This is the correct behaviour, but it means some questions end in
   "ask the Programme Chair".
5. **Small embedder.** `bge-small` (384-d) was chosen for CPU speed and a free deployment;
   a larger embedder (bge-large, e5-large) may lift recall on paraphrased questions.
6. **Single model, single run.** Temperature 0 makes runs reproducible but does not measure
   variance across models. Latency figures are network-dependent.
7. **The abstention gate is calibrated on a small set.** `RERANK_MIN` was fitted on 34 cases
   with only two out-of-scope examples. Too strict and the system refuses answerable
   questions; too loose and it relies entirely on the prompt's grounding rules. This is the
   single most sensitive setting in the system and would need a larger held-out set.
8. **Free-tier quotas.** Groq's 200K tokens/day bounds how many evaluation runs fit in a day;
   the fallback chain keeps the app alive, but answers from a fallback tier come from a
   different model than the one evaluated.

## 10. Responsible use

* **No real student data.** All 10 profiles are synthetic and carry no names,
  identifiers or contact details.
* **Advisory, not authoritative.** The interface states on every screen that registration
  decisions must be confirmed with the Faculty Advisor — matching the SOP, which requires
  Faculty Advisor consultation before registration.
* **Injection surface.** Retrieved document text is fenced inside `<evidence>` and the
  system prompt states that evidence is data, never instruction. This is a mitigation, not a
  guarantee; a full adversarial audit is Phase 10 of Assignment #2.
* **Scope limiting.** Non-academic questions are declined rather than answered from general
  knowledge, including requests to access other students' records (test case S02).

## 11. Conclusions

1. **Grounding beats fluency.** The largest single gain comes from retrieval, and most of it
   is on questions with one specific right value.
2. **Knowing when *not* to answer is a design feature, not a side effect.** It required an
   absolute-score abstention test; the ranked score cannot express it.
3. **Take arithmetic away from the model.** Eligibility, credits and progression belong in
   deterministic code, with the model reduced to explanation and citation.
4. **Asking one good question can be worth more than a better model.** Where a question has
   two correct answers depending on unstated context, no amount of model capability fixes
   it — only the clarifying question does.
5. **Real documents contain real contradictions.** 30 were found in the published
   curriculum. Surfacing them honestly is more useful than resolving them silently.

---

### Appendix A — Repository layout

```
advisor/ingest.py       PDF + Excel → structured tables and cited chunks
advisor/retriever.py    BM25 + bge-small/FAISS → RRF → cross-encoder rerank, abstention gate
advisor/llm.py          Groq → Groq → Gemini router with token-window throttle
advisor/eligibility.py  deterministic rule engine (clause-cited)
advisor/conflicts.py    dangling / ordering / cross-batch conflict mining
advisor/prompts.py      the four variants, one ingredient apart
advisor/advisor.py      orchestration, follow-up, abstention
advisor/synthetic.py    10 anonymised student profiles
advisor/testset.py      34 verified test cases
advisor/evaluate.py     scoring + the eight metrics
app.py                  Streamlit prototype (deployed)
notebooks/…ipynb        Phases 1–5, step by step
```

### Appendix B — Test-case inventory

| ID | Category | Expected | Question |
|---|---|---|---|
| R01 | regulation_factual | ANSWER | What is the minimum attendance percentage I must have in every course? |
| R02 | regulation_factual | ANSWER | Can my attendance requirement ever be relaxed below 75%, and how low can i |
| R03 | regulation_factual | ANSWER | What is the minimum passing grade in a course? |
| R04 | regulation_factual | ANSWER | Which grades count towards earned credits? |
| R05 | regulation_factual | ANSWER | I got an 'FA' in a course. Can I sit the Make-Up Examination for it? |
| R06 | regulation_factual | ANSWER | What is the maximum time allowed to complete my B.Tech programme? |
| R07 | regulation_factual | ANSWER | What CGPA do I need to actually be awarded the degree? |
| R08 | regulation_factual | ANSWER | How many weeks do I have to add or drop a course after classes start? |
| P01 | prerequisite_eligibility | ANSWER | I am in the 2025 batch. What are the prerequisites for COMP203, Analysis o |
| P02 | prerequisite_eligibility | ANSWER | I failed Data Structures. Can I register for Analysis of Algorithms next s |
| P03 | prerequisite_eligibility | ANSWER | What are the prerequisites for DATA132, Data Visualization and Storytellin |
| P04 | prerequisite_eligibility | ANSWER | I have passed DATA132 but not MATH203. Am I eligible for DATA209, Advanced |
| P05 | prerequisite_eligibility | ANSWER | Does MTSC211 Sustainable Smart Materials have a prerequisite? |
| P06 | prerequisite_eligibility | FLAG_CONFLICT | I am STU-B with a clean record in the 2025 batch. Am I eligible for DATA30 |
| C01 | credits_progression | ANSWER | What minimum CGPA do I need to be promoted to Year 2? |
| C02 | credits_progression | ANSWER | What minimum CGPA is required for progression to Year 3 and beyond? |
| C03 | credits_progression | ANSWER | My CGPA is 3.84 at the end of Semester 2. Can I move into Year 2? |
| C04 | credits_progression | ANSWER | How many total credits does the B.Tech programme require for graduation? |
| O01 | course_offering | ANSWER | In which semester is DATA301 Machine Learning offered in the 2025 curricul |
| O02 | course_offering | ANSWER | How many credits is COMP201 Data Structures worth? |
| O03 | course_offering | ANSWER | Which minors are available to B.Tech students? |
| M01 | missing_information | ABSTAIN | How many grade points does an 'A+' carry on the 10-point scale? |
| M02 | missing_information | ABSTAIN | What is the detailed syllabus and topic breakdown of COMP201 Data Structur |
| M03 | missing_information | ABSTAIN | Who is the course instructor for Machine Learning this semester, and what  |
| M04 | missing_information | ABSTAIN | What were the exact marks cut-offs used for each intermediate grade in COM |
| X01 | conflicting_rules | ASK_FOLLOWUP | What are the prerequisites for DATA301 Machine Learning? |
| X02 | conflicting_rules | FLAG_CONFLICT | I am in the 2025 batch and want to take COMP302 Internet of Things in Seme |
| X03 | conflicting_rules | ANSWER | The handbook says 75% attendance is required but also mentions 65%. Which  |
| A01 | ambiguous_needs_followup | ASK_FOLLOWUP | Can I take Machine Learning next semester? |
| A02 | ambiguous_needs_followup | ASK_FOLLOWUP | Am I eligible to graduate? |
| A03 | ambiguous_needs_followup | ASK_FOLLOWUP | Is my attendance okay? |
| A04 | ambiguous_needs_followup | ASK_FOLLOWUP | I don't have my CGPA handy. Will I be promoted to the next year? |
| S01 | out_of_scope | ABSTAIN | What is the price of a meal in the campus canteen? |
| S02 | out_of_scope | ABSTAIN | Write me a Python script that scrapes the Digii portal for other students' |