"""
Generate the Assignment #1 presentation deck.

    python3 scripts/build_ppt.py     ->  report/AI_Academic_Advisor_Presentation.pptx

Reads results/metrics.csv if present so the results slides carry real numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from advisor import config, conflicts, testset          # noqa: E402
from advisor.eligibility import load_students           # noqa: E402

NAVY = RGBColor(0x1F, 0x33, 0x53)
BLUE = RGBColor(0x4A, 0x6F, 0xA5)
GREY = RGBColor(0x55, 0x5A, 0x60)
RED = RGBColor(0xC0, 0x50, 0x4D)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


def _title(slide, text, sub=""):
    box = slide.shapes.add_textbox(Inches(.6), Inches(.35), Inches(12.1), Inches(1.0))
    tf = box.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, NAVY
    if sub:
        q = tf.add_paragraph()
        q.text = sub
        q.font.size, q.font.color.rgb = Pt(14), GREY
    line = slide.shapes.add_shape(1, Inches(.6), Inches(1.45), Inches(12.1), Pt(2.2))
    line.fill.solid()
    line.fill.fore_color.rgb = BLUE
    line.line.fill.background()


def bullets(title, sub, items, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title, sub)
    box = s.shapes.add_textbox(Inches(.75), Inches(1.75), Inches(11.8), Inches(5.2))
    tf = box.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lvl = 0
        if isinstance(it, tuple):
            it, lvl = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "– ") + it
        p.level = lvl
        p.font.size = Pt(19 - 3 * lvl)
        p.font.color.rgb = NAVY if lvl == 0 else GREY
        p.space_after = Pt(9)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def table_slide(title, sub, df, notes="", fs=11):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title, sub)
    rows, cols = df.shape[0] + 1, df.shape[1] + 1
    h = min(5.2, .34 * rows + .3)
    t = s.shapes.add_table(rows, cols, Inches(.7), Inches(1.8),
                           Inches(11.9), Inches(h)).table
    t.cell(0, 0).text = str(df.index.name or "")
    for j, c in enumerate(df.columns, 1):
        t.cell(0, j).text = str(c)
    for i, (idx, row) in enumerate(df.iterrows(), 1):
        t.cell(i, 0).text = str(idx)
        for j, v in enumerate(row, 1):
            t.cell(i, j).text = "—" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v))
    for r in t.rows:
        for c in r.cells:
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(fs)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def image_slide(title, sub, img, notes=""):
    if not Path(img).exists():
        return None
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _title(s, title, sub)
    s.shapes.add_picture(str(img), Inches(1.1), Inches(1.8), width=Inches(11.1))
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# ---------------------------------------------------------------- data
cases = testset.load()
profiles = load_students()
confs = conflicts.detect()
courses = pd.read_csv(config.COURSES_CSV)
chunks = sum(1 for _ in open(config.CHUNKS_JSONL))
mpath = config.RESULTS / "metrics.csv"
metrics = pd.read_csv(mpath, index_col=0) if mpath.exists() else None
cpath = config.RESULTS / "per_category.csv"
percat = pd.read_csv(cpath, index_col=0) if cpath.exists() else None

# ---------------------------------------------------------------- slides
s = prs.slides.add_slide(prs.slide_layouts[6])
b = s.shapes.add_textbox(Inches(.9), Inches(2.1), Inches(11.5), Inches(3.2))
tf = b.text_frame
tf.word_wrap = True
tf.text = "AI Academic Advisor"
tf.paragraphs[0].font.size = Pt(46)
tf.paragraphs[0].font.bold = True
tf.paragraphs[0].font.color.rgb = NAVY
for txt, sz, col in [
    ("Knowing when to answer, what to answer, when to ask, and when to say "
     "it does not have enough information", 19, BLUE),
    ("DATA308 · Generative AI · Assignment #1 (Phases 1–7)", 15, GREY),
    ("<< your name / roll number >>     |     Live demo: << streamlit URL >>", 13, GREY),
]:
    p = tf.add_paragraph()
    p.text = txt
    p.font.size, p.font.color.rgb = Pt(sz), col
    p.space_before = Pt(14)

bullets("The problem", "Why document-upload-and-ask is not enough", [
    "A student asks: “Can I take Machine Learning next semester?”",
    ("There is no single correct answer. DATA301 requires DATA202+MATH301 for the 2022/2024 "
     "batches, but DATA206+MATH301 for 2025/2026-DS.", 1),
    ("A system that answers immediately is wrong for half of all askers — and sounds equally "
     "confident to both.", 1),
    "Three failure modes an academic advisor must avoid:",
    ("Inventing a rule that sounds plausible (grade points, credit counts)", 1),
    ("Advising confidently on incomplete information", 1),
    ("Silently picking one side of a contradiction in the regulations", 1),
    "Goal: measure how much prompting, retrieval and structured data each contribute.",
], notes="Open on the DATA301 example — it recurs throughout and justifies the follow-up mechanism.")

bullets("Core design decision", "What the model is, and is not, allowed to do", [
    "Numbers are COMPUTED. Prose is RETRIEVED. The model only explains and cites.",
    ("Prerequisites, credits, attendance, CGPA progression → deterministic Python rule engine", 1),
    ("Regulation prose → hybrid retrieval with clause-level citations", 1),
    ("Rule-engine output is injected as verified fact the model may not recompute", 1),
    "This removes the two things LLMs are worst at — multi-step arithmetic and strict rule "
    "application — from the model's job entirely.",
    f"Every rule carries its clause: attendance 75% (7.2), floor 65% (7.3), "
    f"Year-2 CGPA 4.00 / Year-3 5.00 (12.1), passing grades O–D (8.11).",
], notes="This slide is the thesis of the project. Everything else follows from it.")

bullets("Data ingested", "Four provided documents, two deliberately separate layers", [
    f"Student Handbook (Aug 2026, 92 pp) + SOP (17 Aug 2026) → {chunks} cited passages",
    f"Semester Spread & Structures → {len(courses)} course rows, "
    f"{courses.course_code.nunique()} unique codes, {courses.batch.nunique()} batches, S1–S8",
    "Minor Courses workbook → 7 minors",
    "STRUCTURED layer: codes, credits, prerequisites, offerings, basket minimums",
    "UNSTRUCTURED layer: regulation prose, chunked at top-level clause boundaries",
    ("Clause-aware chunking matters: without it a Section 12 passage inherits Clause 11.5.7's "
     "citation, and every source attribution built on it is wrong.", 1),
], notes="Source correctness is one of the eight graded metrics, so citations must be exact.")

bullets("A real gap in the provided data", "Table 2 is an image, not text", [
    "The letter-grade → grade-point table (Clause 8.10) is an IMAGE in the handbook PDF.",
    ("Its contents never reach the text layer, so no grade-point or CGPA arithmetic is possible.", 1),
    "A model that answers “A+ = 9 points” is reciting general knowledge about Indian "
    "universities — not this university's regulation.",
    "This became verified test case M01 — the clearest hallucination probe in the set.",
    "Also absent: course syllabi (M02), timetables and instructor allocation (M03).",
], notes="Strong point in the viva: an honest, verifiable limitation found by inspecting the data.")

bullets("Architecture", "", [
    "INGEST → PDF to clause-cited chunks; XLSX to course catalogue, baskets, minors",
    "RETRIEVE → BM25 ⊕ dense embeddings, min-max fused",
    "ABSTAIN → on ABSOLUTE scores, not the fused score",
    ("Min-max normalisation always forces the top hit to 1.0 — so the ranked score can never "
     "say “nothing here”. Abstention uses raw cosine + query-term coverage instead.", 1),
    ("When both are weak the system returns INSUFFICIENT INFORMATION without calling the "
     "model at all — a generation step there can only hallucinate.", 1),
    "REASON → rule engine computes eligibility; conflict detector flags contradictions",
    "GENERATE → Gemini 3.6 Flash, temperature 0, output contract "
    "ANSWER · EVIDENCE · CONFIDENCE · FOLLOW-UP",
], notes="The abstention bug is worth dwelling on — it is the subtlest engineering point.")

bullets("Phase 3 — Prompt engineering, one ingredient at a time", "", [
    "V1 Basic LLM — the question, sent to the model",
    "V2 + role, context, constraints, XML delimiters, output contract",
    "V3 + retrieved evidence, citation duty, abstention duty, conflict duty",
    "V4 + verified rule-engine facts, conflict warnings, follow-up protocol",
    "Each adds exactly ONE thing, so every metric change is attributable.",
    ("Follow-ups are deliberately narrow: asked only when the missing fact would CHANGE the "
     "answer (batch, prerequisite outcome, CGPA, intended semester); at most two.", 1),
], notes="An advisor that asks on every turn is as useless as one that never asks.")

bullets(f"Phase 2 — {len(profiles)} synthetic students", "Fully anonymised; real course codes", [
    "Opaque tokens STU-A…STU-J. No names, no enrolment numbers, no contact details.",
    "Course codes, credits and prerequisites are real — so the arithmetic is genuine.",
    "STU-A — failed Data Structures, a prerequisite for Analysis of Algorithms",
    "STU-C — CGPA 3.84 against a 4.00 Year-2 threshold",
    "STU-E — 'FA' grade plus 58% attendance, below the unrelaxable 65% floor",
    "STU-G — CGPA absent: the advisor must ask, not assume",
    "STU-H — 2022 curriculum, where DATA301's prerequisites differ",
])

conf_df = pd.DataFrame({
    "Conflict kind": ["DANGLING_PREREQ", "ORDERING", "CROSS_BATCH"],
    "Count": [sum(c.kind == k for c in confs)
              for k in ("DANGLING_PREREQ", "ORDERING", "CROSS_BATCH")],
    "Example": [
        "DATA301 (2025) requires DATA206 — never offered in the 2025 spread",
        "COMP302 sits in S6 but requires COMP208, also in S6",
        "DATA301: DATA202+MATH301 (2022/24) vs DATA206+MATH301 (2025/26)",
    ]}).set_index("Conflict kind")
table_slide(f"{len(confs)} real conflicts, mined from the university's own data",
            "Not invented — detected automatically", conf_df, fs=12,
            notes="The cross-batch case is the strongest argument for asking follow-up questions.")

s = testset.summary()
cat_df = pd.DataFrame({"Cases": s["by_category"]})
cat_df.index.name = "Category"
table_slide(f"Phase 4 — {s['total']} verified test cases",
            "Every expected answer traced to a clause before being written down",
            cat_df, fs=13,
            notes="Beyond the required 20–30. Each case also names the governing clause.")

bullets("How responses are scored", "Deterministic, auditable, reproducible", [
    "must_include — facts a correct answer has to contain",
    "must_not_include — HALLUCINATION TRAPS: plausible but wrong values",
    ("M01 traps: “9”, “9.0”, “10 points”.  R01 traps: “80%”, “85%”.", 1),
    ("These make the hallucination rate a measurement, not an impression.", 1),
    "Four labels: correct · partially_correct · unsupported · incorrect",
    "Wrong ACTION scored as harshly as wrong content:",
    ("answering where the corpus is silent → unsupported", 1),
    ("abstaining where the answer existed, or advising blindly → incorrect", 1),
    "Correctness = matches the verified rule. Reliability = stays grounded, admits gaps.",
])

if metrics is not None:
    show = [c for c in ["accuracy_%", "hallucination_rate_%", "eligibility_correct_%",
                        "source_correct_%", "missing_info_handled_%", "conflict_handled_%",
                        "mean_latency_s", "incorrect_recommendations"] if c in metrics.columns]
    table_slide("Results — the eight metrics",
                "Basic LLM → Structured Prompting → RAG → RAG + Structured Data",
                metrics[show].T, fs=11)
    if percat is not None:
        table_slide("Correctness by question category", "%", percat, fs=11)
else:
    bullets("Results", "Run scripts/run_evaluation.py, then rebuild this deck", [
        "The eight-metric table is inserted here automatically once the evaluation has run.",
        "python3 scripts/run_evaluation.py",
        "python3 scripts/build_ppt.py",
    ])

image_slide("Phase 5 — progression across the four variants", "",
            config.RESULTS / "phase5_progression.png")
image_slide("Where each variant wins and loses", "",
            config.RESULTS / "phase5_by_category.png")

bullets("Phase 5 — what improved, and what did not", "", [
    "Structured prompting improves FORMAT more than TRUTH — a model with no documents has "
    "nothing to be right about, and cannot cite what it was never given.",
    "Retrieval is where correctness arrives — on questions whose answer is a specific value "
    "in a specific clause. It is also what makes abstention possible at all.",
    "Structured student data improves eligibility decisions and conflict handling — exactly "
    "the multi-step tasks the rule engine took away from the model.",
    "What did NOT improve:",
    ("Latency rises monotonically — reliability is bought with response time.", 1),
    ("No variant answers M01, and none should. The gain is correctly refusing.", 1),
    ("Prompt engineering alone plateaus.", 1),
])

bullets("Does asking the right question improve the recommendation?", "Yes — and here is the proof", [
    "“What are the prerequisites for Machine Learning?” has NO single correct answer.",
    ("2022 / 2024 batches → DATA202 + MATH301", 1),
    ("2025 / 2026-DS batches → DATA206 + MATH301", 1),
    "Answering immediately is wrong for roughly half of all askers.",
    "One clarifying question — “which batch are you in?” — converts an unanswerable "
    "question into an answerable one.",
    "The follow-up is not politeness. It is the difference between 50% and 100% correct.",
], notes="This directly answers the question posed in Phase 1 of the brief.")

bullets("Limitations", "Where the system still fails", [
    "Table 2 is an image → no grade-point or CGPA arithmetic. The system abstains.",
    "2023/2024 semester spreads are sparse in the source workbook → genuinely unanswerable.",
    "Minor credits are not placed in the semester spread → basket progress approximate.",
    f"{len(confs)} conflicts are DETECTED, not RESOLVED — the documents do not say which governs.",
    "No cross-encoder reranker (free-tier quota); source correctness could go higher.",
    "Abstention thresholds are hand-calibrated — the single most sensitive setting: "
    "too strict refuses answerable questions, too loose hallucinates.",
])

bullets("Conclusions", "", [
    "Grounding beats fluency — the largest single gain comes from retrieval.",
    "Knowing when NOT to answer is a design feature, and it required an absolute-score "
    "abstention test; the ranked score cannot express it.",
    "Take arithmetic away from the model — eligibility belongs in deterministic code.",
    "Asking one good question can be worth more than a better model.",
    "Real documents contain real contradictions. Surfacing them honestly beats resolving "
    "them silently.",
])

s = prs.slides.add_slide(prs.slide_layouts[6])
_title(s, "Demo", "Live prototype")
b = s.shapes.add_textbox(Inches(.9), Inches(2.2), Inches(11.5), Inches(3.5))
tf = b.text_frame
tf.word_wrap = True
for i, (txt, sz) in enumerate([
    ("<< paste your Streamlit Cloud URL here >>", 22),
    ("Scenarios to demonstrate:", 17),
    ("1.  Normal — “What is the minimum attendance requirement?”  → cited answer", 15),
    ("2.  Ambiguous — “Can I take Machine Learning next semester?”  → asks for batch", 15),
    ("3.  Missing — “How many grade points does an A+ carry?”  → refuses, names the gap", 15),
    ("4.  Conflict — DATA301 for STU-B  → flags the dangling prerequisite", 15),
    ("5.  Student-specific — STU-A on Analysis of Algorithms  → rule-engine verdict", 15),
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = txt
    p.font.size = Pt(sz)
    p.font.bold = (i == 0)
    p.font.color.rgb = BLUE if i == 0 else NAVY
    p.space_after = Pt(8)
s.notes_slide.notes_text_frame.text = (
    "Run all five live. Scenario 3 is the one that distinguishes this from a "
    "document-Q&A demo — show it refusing, and explain why Table 2 is unreadable.")

out = ROOT / "report" / "AI_Academic_Advisor_Presentation.pptx"
out.parent.mkdir(exist_ok=True)
prs.save(out)
print(f"wrote {out}")
print(f"  {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
print(f"  results embedded: {metrics is not None}")
