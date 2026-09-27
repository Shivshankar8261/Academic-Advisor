"""
AI Academic Advisor -- deployable prototype (Phase 6).

Run locally :  streamlit run app.py
Deploy      :  push to GitHub -> share.streamlit.io -> add GOOGLE_API_KEY to Secrets
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from advisor import config, conflicts, prompts                      # noqa: E402
from advisor.advisor import AcademicAdvisor                          # noqa: E402
from advisor.eligibility import EligibilityEngine, load_students     # noqa: E402
from advisor.llm import LLMClient                                    # noqa: E402
from advisor.retriever import HybridRetriever                        # noqa: E402

st.set_page_config(page_title="AI Academic Advisor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1150px;}
  .ev {background:#f5f7fa; border-left:3px solid #4a6fa5; padding:.6rem .8rem;
       margin:.35rem 0; font-size:.82rem; border-radius:0 4px 4px 0;}
  .ev b {color:#2c4a70;}
  .warn {background:#fff6e5; border-left:3px solid #d98324; padding:.6rem .8rem;
         margin:.35rem 0; font-size:.84rem; border-radius:0 4px 4px 0;}
  .pill {display:inline-block; padding:.12rem .55rem; border-radius:10px;
         font-size:.72rem; font-weight:600; margin-right:.3rem;}
  .p-hi {background:#d8f0dc; color:#1d6b2b;} .p-md {background:#fdf0cd; color:#8a6100;}
  .p-lo {background:#fadadd; color:#a3262f;} .p-ab {background:#e4e6eb; color:#4a4a4a;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing university documents…")
def _bootstrap(api_key: str):
    if not config.CHUNKS_JSONL.exists():
        from advisor.ingest import build_all
        build_all(verbose=False)
    if not config.STUDENTS_JSON.exists():
        from advisor.synthetic import main as gen
        gen()
    return HybridRetriever(api_key=api_key), EligibilityEngine()


def _key() -> str:
    return (st.session_state.get("api_key")
            or os.getenv("GOOGLE_API_KEY", "")
            or st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else "")


# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🎓 Academic Advisor")
    st.caption("DATA308 · Generative AI · Assignment #1")

    default_key = os.getenv("GOOGLE_API_KEY", "")
    try:
        default_key = default_key or st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    st.session_state["api_key"] = st.text_input(
        "Google Gemini API key", value=default_key, type="password",
        help="Get one free at aistudio.google.com/apikey")

    st.divider()
    variant = st.selectbox(
        "System variant", prompts.VARIANTS, index=3,
        format_func=lambda v: prompts.VARIANT_LABELS[v],
        help="Switch between the four versions compared in Phase 5.")

    students = {s.student_id: s for s in load_students()} if config.STUDENTS_JSON.exists() else {}
    ids = ["(no profile — anonymous question)"] + list(students)
    picked = st.selectbox("Synthetic student profile", ids,
                          help="Only used by V4. All profiles are synthetic and anonymised.")
    student = students.get(picked)

    if student:
        st.markdown(f"**{student.student_id}** · batch {student.batch} · S{student.current_semester}")
        st.markdown(f"CGPA **{student.cgpa if student.cgpa is not None else '—'}** · "
                    f"passed **{len(student.passed())}** · failed **{len(student.failed())}**")
        if student.failed():
            st.warning("Failing grades: " +
                       ", ".join(f"{c} ({g})" for c, g in student.failed().items()))
        st.caption(student.notes)

    st.divider()
    show_ev = st.checkbox("Show retrieved evidence", value=True)
    show_facts = st.checkbox("Show rule-engine working", value=False)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Sources: Student Handbook (Aug 2026) · SOP for Students (17 Aug 2026) · "
               "Programme Structure & Semester Spread (Sept 2026) · Minor Courses list.")
    st.caption("⚠️ Advisory only. Confirm with your Faculty Advisor before registering.")

# --------------------------------------------------------------------------
api_key = st.session_state.get("api_key", "")
if not api_key:
    st.title("AI Academic Advisor")
    st.info("Enter a Google Gemini API key in the sidebar to begin. "
            "A free key is available at **aistudio.google.com/apikey**.")
    st.subheader("What this system does")
    st.markdown("""
- Answers academic questions **only** from the four provided university documents, with a clause-level citation for each claim.
- Computes prerequisites, credits, attendance and progression with a **deterministic rule engine** — the model never does the arithmetic.
- **Asks a follow-up question** when the answer would change depending on information you have not given.
- Says **"insufficient information"** instead of guessing when the documents do not cover something.
- **Flags contradictions** in the curriculum rather than silently picking a side.
""")
    st.stop()

retriever, engine = _bootstrap(api_key)
llm = LLMClient(api_key=api_key)
advisor = AcademicAdvisor(
    variant,
    retriever=retriever if variant in ("V3_rag", "V4_rag_student") else None,
    engine=engine if variant == "V4_rag_student" else None,
    llm=llm)

st.title("AI Academic Advisor")
st.caption(f"{prompts.VARIANT_LABELS[variant]} · retrieval index: **{retriever.mode}** · "
           f"{len(retriever.chunks)} indexed passages · {config.CURRENT_TERM}")

EXAMPLES = [
    "Can I take Machine Learning next semester?",
    "What is the minimum attendance requirement?",
    "I failed Data Structures — can I register for Analysis of Algorithms?",
    "What are the prerequisites for DATA301?",
    "How many grade points does an A+ carry?",
    "What CGPA do I need to progress to Year 3?",
]
cols = st.columns(3)
clicked = None
for i, ex in enumerate(EXAMPLES):
    if cols[i % 3].button(ex, key=f"ex{i}", use_container_width=True):
        clicked = ex

st.session_state.setdefault("messages", [])
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for block in m.get("extras", []):
            st.markdown(block, unsafe_allow_html=True)

prompt = st.chat_input("Ask about courses, prerequisites, credits, attendance or progression…")
question = clicked or prompt

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the regulations…"):
            ans = advisor.answer(question, student=student)

        if not ans.ok:
            st.error(f"The model call failed: {ans.error}")
            st.stop()

        badge = {"High": "p-hi", "Medium": "p-md", "Low": "p-lo"}.get(ans.confidence, "p-ab")
        head = f'<span class="pill {badge}">{ans.confidence or "—"} confidence</span>'
        if ans.abstained:
            head += '<span class="pill p-ab">insufficient information</span>'
        if ans.asked_followup:
            head += '<span class="pill p-md">follow-up asked</span>'
        head += f'<span class="pill p-ab">{ans.latency_s:.1f}s</span>'
        st.markdown(head, unsafe_allow_html=True)
        st.markdown(ans.text)

        extras = []
        if getattr(advisor, "last_conflicts", None):
            body = "".join(
                f'<div class="warn">⚠️ <b>{c.kind}</b> — {c.detail}<br><i>{c.advice}</i></div>'
                for c in advisor.last_conflicts)
            st.markdown("**Conflicts detected in the curriculum**", unsafe_allow_html=True)
            st.markdown(body, unsafe_allow_html=True)
            extras.append(body)
            advisor.last_conflicts = []

        if show_ev and ans.hits:
            with st.expander(f"Retrieved evidence ({len(ans.hits)} passages)"):
                for h in ans.hits:
                    st.markdown(
                        f'<div class="ev"><b>{h.chunk.citation()}</b> · '
                        f'score {h.score:.2f} · cosine {h.dense:.2f}<br>'
                        f'{h.chunk.text[:600]}…</div>', unsafe_allow_html=True)

        if show_facts and ans.engine_facts:
            with st.expander("Rule-engine working (verified, deterministic)"):
                st.code(ans.engine_facts, language="text")

        st.session_state.messages.append(
            {"role": "assistant", "content": ans.text, "extras": extras})
